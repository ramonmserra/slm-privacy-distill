"""Resumo legivel de um run de treino, a partir do `run.json`, para o TCC.

    python -m treino.resumo saida/run.json                  # imprime
    python -m treino.resumo saida/run.json saida/resumo.txt # grava

So' reorganiza o que ja' esta' no run.json: nenhum numero e' calculado alem
de conversoes de unidade (segundos -> minutos, bytes -> MB).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_METRICAS = (
    ("train_loss", "Perda de treino (media)"),
    ("eval_loss", "Perda no held-out (final)"),
    ("eval_mean_token_accuracy", "Acuracia por token no held-out"),
    ("eval_entropy", "Entropia no held-out"),
    ("train_runtime", "Tempo so' de treino (s)"),
    ("train_samples_per_second", "Exemplos por segundo"),
    ("train_steps_per_second", "Passos por segundo"),
    ("eval_runtime", "Tempo da avaliacao final (s)"),
    ("epoch", "Epocas concluidas"),
)


def _duracao(segundos: float) -> str:
    minutos = segundos / 60
    return f"{segundos:.0f} s ({minutos:.1f} min)" if minutos < 60 else f"{segundos:.0f} s ({minutos / 60:.2f} h)"


def _numero(valor) -> str:
    return f"{valor:.4f}" if isinstance(valor, float) else str(valor)


def resumir(run: dict, arquivos: dict[str, int] | None = None) -> str:
    hiper, hardware, metricas = run.get("hiperparametros", {}), run.get("hardware", {}), run.get("metricas", {})
    linhas = [
        "RESUMO DO RUN DE TREINO",
        "=======================",
        f"Status:              {run.get('status', '')}",
        f"Modelo-base:         {run.get('modelo_base', '')}",
        f"Modelo no Ollama:    {run.get('modelo_local_id', '')}",
        f"Dataset (versao):    {run.get('dataset_versao', '')}",
        f"Exemplos de treino:  {run.get('exemplos_de_treino', '')}",
        f"Exemplos de held-out usados na perda de validacao: {run.get('exemplos_de_held_out', '')}",
        f"Receita / perfil:    {run.get('receita', '')} / {run.get('perfil', '')}",
        f"Prefixo do assistente: {run.get('prefixo_do_assistente', '')!r}",
        "",
        "TEMPO",
        f"Inicio (UTC):        {run.get('iniciado_em', '')}",
        f"Fim (UTC):           {run.get('concluido_em', '')}",
        f"Duracao total:       {_duracao(float(run.get('duracao_s') or 0))}",
        "",
        "METRICAS",
    ]
    linhas += [f"{rotulo}: {_numero(metricas[chave])}" for chave, rotulo in _METRICAS if chave in metricas]
    linhas += [
        "Obs.: 'acuracia por token' e' a fracao de tokens da resposta previstos",
        "corretamente no held-out durante o treino. NAO e' a acuracia juridica do",
        "modelo; essa vem dos runs de avaliacao da API (passo de avaliacao).",
        "",
        "HIPERPARAMETROS",
    ]
    linhas += [f"{chave}: {valor}" for chave, valor in hiper.items()]
    linhas += ["", "HARDWARE"] + [f"{chave}: {valor}" for chave, valor in hardware.items()]
    if arquivos:
        linhas += ["", "ARQUIVOS"] + [f"{nome}: {tamanho / 1024**2:.1f} MB" for nome, tamanho in sorted(arquivos.items())]
    if run.get("observacoes"):
        linhas += ["", "OBSERVACOES", str(run["observacoes"])]
    return "\n".join(linhas) + "\n"


def main(argv: list[str] | None = None) -> int:
    argumentos = sys.argv[1:] if argv is None else argv
    if len(argumentos) not in (1, 2):
        print("uso: python -m treino.resumo <run.json> [resumo.txt]", file=sys.stderr)
        return 2
    origem = Path(argumentos[0])
    run = json.loads(origem.read_text(encoding="utf-8"))
    destino = Path(argumentos[1]) if len(argumentos) == 2 else None
    ignorados = {"desktop.ini", ".ds_store", "thumbs.db"}
    vizinhos = {p.name: p.stat().st_size for p in origem.parent.iterdir()
                if p.is_file() and p != destino and p.name.lower() not in ignorados}
    texto = resumir(run, vizinhos)
    if destino:
        destino.write_text(texto, encoding="utf-8")
        print(f"resumo -> {destino}")
    else:
        print(texto, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
