"""SFT com LoRA do aluno sobre o JSONL exportado pela API.

    python -m treino.sft_lora --perfil colab-t4 --dataset treino.jsonl \
        --held-out held_out.jsonl --dataset-versao ab12cd34ef56 --saida saida/

    python -m treino.sft_lora --perfil cpu-local --dataset treino.jsonl --max-exemplos 50 --saida saida/
    python -m treino.sft_lora --perfil colab-t4 --dataset treino.jsonl --dry-run   # so' valida e planeja

E' a destilacao BLACK-BOX da pesquisa: o aluno aprende a imitar pergunta ->
raciocinio -> resposta do professor. Mesmo script nos dois ambientes; o que
muda e' o `--perfil` (ver `preparacao.PERFIS`). Produz:

    saida/adapter/      LoRA treinado (pequeno, versionavel)
    saida/merged/       base + LoRA fundidos (para converter em GGUF)
    saida/run.json      o corpo de `POST /treinos`

Dependencias pesadas (torch, transformers, trl, peft, datasets) sao
importadas DENTRO de `treinar`: o `--dry-run` e os testes da preparacao
rodam sem elas.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

from treino.preparacao import (
    MODULOS_LORA_QWEN,
    PERFIS,
    PREFIXO_SEM_PENSAR_QWEN3,
    Perfil,
    RegistroDoRun,
    carregar_jsonl,
    modelo_local_id_sugerido,
    preparar,
)


def _argumentos(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--perfil", choices=sorted(PERFIS), required=True)
    ap.add_argument("--dataset", type=Path, required=True, help="treino.jsonl exportado pela API")
    ap.add_argument("--held-out", type=Path, help="held_out.jsonl: so' para eval loss; NUNCA entra no treino")
    ap.add_argument("--dataset-versao", default="", help="versao do manifesto (vai para o run.json)")
    ap.add_argument("--modelo-base", help="sobrescreve o modelo-base do perfil")
    ap.add_argument("--saida", type=Path, default=Path("saida"))
    ap.add_argument("--epocas", type=float)
    ap.add_argument("--max-exemplos", type=int, help="corta o treino (validacao da receita na CPU)")
    ap.add_argument("--max-eval", type=int, default=100, help="itens do held-out usados no eval loss")
    ap.add_argument("--prefixo-do-assistente", default=PREFIXO_SEM_PENSAR_QWEN3,
                    help="anteposto a resposta; '' para modelo que nao usa bloco <think>")
    ap.add_argument("--sem-merge", action="store_true", help="nao funde o LoRA no base (economiza disco)")
    ap.add_argument("--dry-run", action="store_true", help="valida dados e escreve o plano, sem treinar")
    return ap.parse_args(argv)


def _perfil_efetivo(args: argparse.Namespace) -> Perfil:
    perfil = PERFIS[args.perfil]
    alteracoes = {}
    if args.modelo_base:
        alteracoes["modelo_base"] = args.modelo_base
    if args.epocas is not None:
        alteracoes["epocas"] = args.epocas
    return Perfil(**{**perfil.como_dict(), **alteracoes}) if alteracoes else perfil


def _carregar(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    treino = preparar(carregar_jsonl(args.dataset), args.prefixo_do_assistente)
    if args.max_exemplos:
        treino = treino[: args.max_exemplos]
    held_out: list[dict] = []
    if args.held_out:
        held_out = preparar(carregar_jsonl(args.held_out), args.prefixo_do_assistente)[: args.max_eval]
    ids_treino = {l["id"] for l in treino}
    vazados = [l["id"] for l in held_out if l["id"] in ids_treino]
    if vazados:
        raise SystemExit(f"ERRO: {len(vazados)} id(s) do held-out tambem estao no treino: {vazados[:3]}")
    return treino, held_out


def _carregar_modelo(classe, nome: str, dtype, **kwargs):
    """`from_pretrained` com o nome do parametro de precisao que a versao
    instalada aceita: `dtype` (transformers >= 4.56) ou `torch_dtype`
    (anteriores). O Colab instala a versao que quiser; a receita nao pode
    quebrar por um rename de kwarg."""

    try:
        return classe.from_pretrained(nome, dtype=dtype, **kwargs)
    except TypeError:
        return classe.from_pretrained(nome, torch_dtype=dtype, **kwargs)


def _prompt_e_completion(linha: dict) -> dict:
    """`messages` [system, user, assistant] -> {"prompt": [system, user],
    "completion": [assistant]}, o formato conversacional prompt/completion do
    TRL (perda so' na completion)."""

    mensagens = linha["messages"]
    return {"prompt": mensagens[:2], "completion": mensagens[2:]}


def treinar(perfil: Perfil, treino: list[dict], held_out: list[dict], saida: Path, sem_merge: bool) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    usa_gpu = torch.cuda.is_available()
    if not usa_gpu and perfil.precisao != "fp32":
        print("AVISO: sem GPU; forcando fp32 e desligando 4-bit.", file=sys.stderr)
        perfil = Perfil(**{**perfil.como_dict(), "precisao": "fp32", "quatro_bits": False})

    tokenizer = AutoTokenizer.from_pretrained(perfil.modelo_base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[perfil.precisao]
    kwargs: dict = {}
    if perfil.quatro_bits:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True
        )
    if usa_gpu:
        kwargs["device_map"] = "auto"
    modelo = _carregar_modelo(AutoModelForCausalLM, perfil.modelo_base, dtype, **kwargs)

    lora = LoraConfig(
        r=perfil.lora_r, lora_alpha=perfil.lora_alpha, lora_dropout=perfil.lora_dropout,
        target_modules=list(MODULOS_LORA_QWEN), task_type="CAUSAL_LM",
    )
    config = SFTConfig(
        output_dir=str(saida / "checkpoints"),
        num_train_epochs=perfil.epocas,
        learning_rate=perfil.taxa_de_aprendizado,
        per_device_train_batch_size=perfil.tamanho_do_lote,
        per_device_eval_batch_size=perfil.tamanho_do_lote,
        gradient_accumulation_steps=perfil.acumulacao,
        max_length=perfil.comprimento_maximo,
        gradient_checkpointing=perfil.checkpointing,
        fp16=perfil.precisao == "fp16",
        bf16=perfil.precisao == "bf16",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if held_out else "no",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",
        seed=42,
    )
    # Formato prompt/completion conversacional: o TRL calcula a perda SO' na
    # completion (o turno do assistente). E' o que faz o aluno aprender a
    # responder, nao a repetir a pergunta e o system prompt. Preferido a
    # `assistant_only_loss`, que exige marcadores {% generation %} no template
    # de chat -- o do Qwen3 nao os tem.
    trainer = SFTTrainer(
        model=modelo,
        args=config,
        train_dataset=Dataset.from_list([_prompt_e_completion(l) for l in treino]),
        eval_dataset=Dataset.from_list([_prompt_e_completion(l) for l in held_out]) if held_out else None,
        processing_class=tokenizer,
        peft_config=lora,
    )
    resultado = trainer.train()
    metricas = {"train_loss": resultado.training_loss, **{k: v for k, v in resultado.metrics.items() if isinstance(v, (int, float))}}
    if held_out:
        metricas.update({k: v for k, v in trainer.evaluate().items() if isinstance(v, (int, float))})

    trainer.model.save_pretrained(saida / "adapter")
    tokenizer.save_pretrained(saida / "adapter")
    if not sem_merge and not perfil.quatro_bits:
        # Fundir e' o que permite converter para GGUF; com 4-bit o merge
        # exigiria recarregar o base em precisao cheia -- fica para o Colab.
        base = _carregar_modelo(AutoModelForCausalLM, perfil.modelo_base, dtype)
        fundido = PeftModel.from_pretrained(base, saida / "adapter").merge_and_unload()
        fundido.save_pretrained(saida / "merged", safe_serialization=True)
        tokenizer.save_pretrained(saida / "merged")
    return metricas


def main(argv: list[str] | None = None) -> int:
    args = _argumentos(argv)
    perfil = _perfil_efetivo(args)
    treino, held_out = _carregar(args)
    if not treino:
        print("ERRO: dataset de treino vazio", file=sys.stderr)
        return 1

    run = RegistroDoRun(
        dataset_versao=args.dataset_versao, modelo_base=perfil.modelo_base, perfil=perfil.nome,
        hiperparametros={**perfil.como_dict(), "lora_target_modules": list(MODULOS_LORA_QWEN), "max_exemplos": args.max_exemplos},
        modelo_local_id=modelo_local_id_sugerido(perfil.modelo_base, args.dataset_versao or "semversao"),
        exemplos_de_treino=len(treino), exemplos_de_held_out=len(held_out),
        prefixo_do_assistente=args.prefixo_do_assistente,
    )
    print(f"perfil={perfil.nome} base={perfil.modelo_base} treino={len(treino)} held_out={len(held_out)} "
          f"precisao={perfil.precisao} seq={perfil.comprimento_maximo} epocas={perfil.epocas}")

    if args.dry_run:
        run.status = "dry_run"
        run.observacoes = "plano validado; nenhum treino executado"
        run.gravar(args.saida / "run.json")
        print(f"dry-run ok -> {args.saida / 'run.json'}")
        return 0

    inicio = time.perf_counter()
    try:
        run.metricas = treinar(perfil, treino, held_out, args.saida, args.sem_merge)
        run.status = "concluido"
        run.artefato = str((args.saida / ("adapter" if args.sem_merge else "merged")).resolve())
    except Exception as exc:  # o run.json registra a falha; o traceback vai ao stderr
        run.status = "falhou"
        run.observacoes = f"{type(exc).__name__}: {exc}"[:500]
        raise
    finally:
        run.duracao_s = round(time.perf_counter() - inicio, 1)
        run.concluido_em = dt.datetime.now(dt.timezone.utc).isoformat()
        run.gravar(args.saida / "run.json")
        print(f"run.json -> {args.saida / 'run.json'} ({run.status}, {run.duracao_s}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
