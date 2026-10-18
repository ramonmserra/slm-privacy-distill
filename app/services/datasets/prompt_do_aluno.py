"""O system prompt do ALUNO -- byte a byte o mesmo no treino e na avaliacao.

Regra herdada do projeto de referencia: o exemplo de treino precisa ser
exatamente o prompt que o modelo vai receber em uso. Se o dataset fosse
gerado com uma copia da montagem do prompt, treinaria-se uma distribuicao
que nunca ocorre, e a degradacao apareceria so' na avaliacao.

Curto de proposito: o aluno e' pequeno, e instrucao longa no system e'
contexto que ele nao usa bem. O comportamento (citar ids, abster-se quando a
lei nao resolve) vem dos exemplos, nao da instrucao.
"""

from __future__ import annotations

import hashlib

PROMPT_DO_ALUNO = (
    "Voce e' um assistente juridico especializado na Lei Geral de Protecao de Dados (LGPD, Lei "
    "13.709/2018). Responda em portugues do Brasil, de forma fundamentada, citando os dispositivos "
    "pelo id canonico (ex.: LGPD:art.7:inc.IX). Quando a lei nao resolver o ponto ou faltar "
    "informacao, diga isso e indique o que faltaria, sem inventar prazo, valor ou regra."
)

VERSAO_DO_PROMPT_DO_ALUNO = hashlib.sha256(PROMPT_DO_ALUNO.encode("utf-8")).hexdigest()[:12]


def resposta_de_treino(raciocinio: str, resposta: str, incluir_raciocinio: bool) -> str:
    """O turno do assistente no exemplo de treino. Com raciocinio, o aluno
    aprende a cadeia antes da conclusao (e' a hipotese da pesquisa); sem,
    aprende so' a conclusao (condicao de controle)."""

    if incluir_raciocinio and raciocinio.strip():
        return f"Raciocinio:\n{raciocinio.strip()}\n\nResposta:\n{resposta.strip()}"
    return resposta.strip()
