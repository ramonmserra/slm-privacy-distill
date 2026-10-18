"""DPO com LoRA sobre os pares (chosen, rejected) do on-policy.

    python -m treino.dpo_lora --perfil colab-t4 --pares dpo.jsonl --adapter saida/adapter \
        --dataset-versao ab12cd34ef56 --saida saida-dpo/

Continua o aluno ja treinado por SFT (`--adapter` aponta para o LoRA do run
anterior; sem ele, parte do modelo-base do perfil). O arquivo `dpo.jsonl`
vem de `GET /on-policy/jobs/{id}/arquivo/dpo`: `prompt` (system+user),
`chosen` e `rejected` (um turno de assistente cada), o formato conversacional
que o `DPOTrainer` do TRL consome.

Mesma disciplina do `sft_lora.py`: preparacao pura e testavel, dependencias
pesadas dentro de `treinar`, `--dry-run`, `run.json` = corpo de `POST /treinos`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

from treino.preparacao import MODULOS_LORA_QWEN, PERFIS, PREFIXO_SEM_PENSAR_QWEN3, Perfil, RegistroDoRun, carregar_jsonl, modelo_local_id_sugerido


def validar_par(par: dict) -> None:
    for campo in ("prompt", "chosen", "rejected"):
        if not isinstance(par.get(campo), list) or not par[campo]:
            raise ValueError(f"par {par.get('id')!r}: campo {campo!r} ausente ou vazio")
    if [m.get("role") for m in par["prompt"]] != ["system", "user"]:
        raise ValueError(f"par {par.get('id')!r}: prompt deve ser [system, user]")
    for campo in ("chosen", "rejected"):
        if par[campo][0].get("role") != "assistant" or not str(par[campo][0].get("content", "")).strip():
            raise ValueError(f"par {par.get('id')!r}: {campo} deve ser um turno de assistente nao vazio")


def _com_prefixo(turno: list[dict], prefixo: str) -> list[dict]:
    conteudo = turno[0]["content"]
    if prefixo and not conteudo.startswith(prefixo):
        conteudo = prefixo + conteudo
    return [{"role": "assistant", "content": conteudo}]


def preparar_pares(pares: list[dict], prefixo_do_assistente: str) -> list[dict]:
    """Valida e aplica o mesmo prefixo do SFT aos dois lados: chosen e
    rejected precisam diferir pelo CONTEUDO, nunca pelo formato."""

    saida = []
    for par in pares:
        validar_par(par)
        saida.append({
            "prompt": [{"role": m["role"], "content": m["content"]} for m in par["prompt"]],
            "chosen": _com_prefixo(par["chosen"], prefixo_do_assistente),
            "rejected": _com_prefixo(par["rejected"], prefixo_do_assistente),
        })
    return saida


def _argumentos(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--perfil", choices=sorted(PERFIS), required=True)
    ap.add_argument("--pares", type=Path, required=True, help="dpo.jsonl do job on-policy")
    ap.add_argument("--adapter", type=Path, help="LoRA do SFT anterior (continua dele)")
    ap.add_argument("--dataset-versao", default="")
    ap.add_argument("--modelo-base")
    ap.add_argument("--saida", type=Path, default=Path("saida-dpo"))
    ap.add_argument("--epocas", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.1, help="forca da preferencia (DPO beta)")
    ap.add_argument("--max-pares", type=int)
    ap.add_argument("--prefixo-do-assistente", default=PREFIXO_SEM_PENSAR_QWEN3)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def treinar(perfil: Perfil, pares: list[dict], adapter: Path | None, saida: Path, beta: float, epocas: float) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    usa_gpu = torch.cuda.is_available()
    if not usa_gpu and perfil.precisao != "fp32":
        print("AVISO: sem GPU; forcando fp32.", file=sys.stderr)
        perfil = Perfil(**{**perfil.como_dict(), "precisao": "fp32", "quatro_bits": False})
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[perfil.precisao]

    tokenizer = AutoTokenizer.from_pretrained(perfil.modelo_base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"device_map": "auto"} if usa_gpu else {}
    try:
        modelo = AutoModelForCausalLM.from_pretrained(perfil.modelo_base, dtype=dtype, **kwargs)
    except TypeError:
        modelo = AutoModelForCausalLM.from_pretrained(perfil.modelo_base, torch_dtype=dtype, **kwargs)
    peft_config = None
    if adapter is not None:
        # Continua o LoRA do SFT: os pesos do adapter sao treinaveis e o
        # DPOTrainer usa o modelo sem adapter como referencia implicita.
        modelo = PeftModel.from_pretrained(modelo, adapter, is_trainable=True)
    else:
        peft_config = LoraConfig(r=perfil.lora_r, lora_alpha=perfil.lora_alpha, lora_dropout=perfil.lora_dropout,
                                 target_modules=list(MODULOS_LORA_QWEN), task_type="CAUSAL_LM")

    config = DPOConfig(
        output_dir=str(saida / "checkpoints"), num_train_epochs=epocas, learning_rate=5e-6, beta=beta,
        per_device_train_batch_size=1, gradient_accumulation_steps=perfil.acumulacao,
        max_length=perfil.comprimento_maximo, gradient_checkpointing=perfil.checkpointing,
        fp16=perfil.precisao == "fp16", bf16=perfil.precisao == "bf16", logging_steps=10, save_strategy="epoch",
        report_to="none", seed=42,
    )
    trainer = DPOTrainer(model=modelo, args=config, train_dataset=Dataset.from_list(pares), processing_class=tokenizer, peft_config=peft_config)
    resultado = trainer.train()
    trainer.model.save_pretrained(saida / "adapter")
    tokenizer.save_pretrained(saida / "adapter")
    try:
        fundido = trainer.model.merge_and_unload()
        fundido.save_pretrained(saida / "merged", safe_serialization=True)
        tokenizer.save_pretrained(saida / "merged")
    except Exception as exc:  # merge e' conveniencia; o adapter ja esta' salvo
        print(f"AVISO: merge nao realizado ({exc}); use saida/adapter.", file=sys.stderr)
    return {"train_loss": resultado.training_loss, **{k: v for k, v in resultado.metrics.items() if isinstance(v, (int, float))}}


def main(argv: list[str] | None = None) -> int:
    args = _argumentos(argv)
    perfil = PERFIS[args.perfil]
    if args.modelo_base:
        perfil = Perfil(**{**perfil.como_dict(), "modelo_base": args.modelo_base})
    pares = preparar_pares(carregar_jsonl(args.pares), args.prefixo_do_assistente)
    if args.max_pares:
        pares = pares[: args.max_pares]
    if not pares:
        print("ERRO: nenhum par", file=sys.stderr)
        return 1
    run = RegistroDoRun(
        dataset_versao=args.dataset_versao, modelo_base=perfil.modelo_base, perfil=perfil.nome, receita="dpo_lora",
        hiperparametros={**perfil.como_dict(), "beta": args.beta, "epocas_dpo": args.epocas, "adapter_inicial": str(args.adapter) if args.adapter else None, "pares": len(pares)},
        modelo_local_id=modelo_local_id_sugerido(perfil.modelo_base, args.dataset_versao or "semversao").replace("-lgpd-", "-lgpd-dpo-"),
        exemplos_de_treino=len(pares), prefixo_do_assistente=args.prefixo_do_assistente,
    )
    print(f"perfil={perfil.nome} base={perfil.modelo_base} pares={len(pares)} beta={args.beta} adapter={args.adapter}")
    if args.dry_run:
        run.status, run.observacoes = "dry_run", "plano validado; nenhum treino executado"
        run.gravar(args.saida / "run.json")
        return 0
    inicio = time.perf_counter()
    try:
        run.metricas = treinar(perfil, pares, args.adapter, args.saida, args.beta, args.epocas)
        run.status = "concluido"
        run.artefato = str((args.saida / "merged").resolve())
    except Exception as exc:
        run.status, run.observacoes = "falhou", f"{type(exc).__name__}: {exc}"[:500]
        raise
    finally:
        run.duracao_s = round(time.perf_counter() - inicio, 1)
        run.concluido_em = dt.datetime.now(dt.timezone.utc).isoformat()
        run.gravar(args.saida / "run.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
