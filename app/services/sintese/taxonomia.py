"""Taxonomia das tarefas sinteticas -- fixada ANTES de gerar.

Cada tipo diz ao professor o que produzir e, mais tarde, diz ao relatorio
como agrupar os resultados. A instrucao e' o unico lugar onde o tipo e'
descrito em linguagem natural; o `TipoDeTarefa` e' o token.

As `ancoras` sao os artigos que o tipo exige no contexto alem do alvo
sorteado: classificar um dado sem a definicao legal em contexto, ou pedir
a base legal sem o rol de hipoteses, so' produz abstencao ('o texto nao
traz a definicao') ou citacao de memoria -- os dois erros do piloto. O
art. 5 (definicoes) entra em quase todos os tipos porque foi o artigo que o
professor mais citou sem ter recebido (11 dos 15 descartes do 2o piloto). A
ancora entra no contexto (e em `contexto_ids`) como qualquer outro
dispositivo; o tipo `sem_resposta` nao tem ancora porque a lacuna e' o
objeto do exemplo.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.sintese import TipoDeTarefa


@dataclass(frozen=True)
class DescricaoDaTarefa:
    tipo: TipoDeTarefa
    objetivo: str
    instrucao: str
    ancoras: tuple[str, ...] = ()  # ids de ARTIGO que sempre entram no contexto deste tipo


TAXONOMIA: dict[TipoDeTarefa, DescricaoDaTarefa] = {
    TipoDeTarefa.CONCEITO: DescricaoDaTarefa(
        TipoDeTarefa.CONCEITO,
        "explicar um conceito ou principio da lei",
        "Formule uma pergunta conceitual (o que e', qual a diferenca, quais sao os requisitos) "
        "respondivel pelo dispositivo em contexto; a resposta define o conceito nos termos da lei e "
        "explica a consequencia juridica da definicao.",
        ancoras=("LGPD:art.5", "LGPD:art.6",),
    ),
    TipoDeTarefa.BASE_LEGAL: DescricaoDaTarefa(
        TipoDeTarefa.BASE_LEGAL,
        "classificar a base legal de um tratamento",
        "Descreva uma operacao de tratamento de dados concreta (quem trata, que dado, para que fim) e "
        "pergunte qual hipotese legal a autoriza; a resposta identifica a(s) base(s) aplicavel(is), "
        "descarta as que nao servem e explica por que.",
        ancoras=("LGPD:art.5", "LGPD:art.7", "LGPD:art.11",),
    ),
    TipoDeTarefa.ANALISE_DE_CASO: DescricaoDaTarefa(
        TipoDeTarefa.ANALISE_DE_CASO,
        "avaliar a licitude de uma situacao concreta",
        "Narre um caso pratico com uma afirmacao ou pratica de uma organizacao e pergunte se ela e' "
        "compativel com a lei; a resposta analisa o caso contra o dispositivo, aponta o que falta ou "
        "excede e evita conclusao absoluta onde a lei preve condicoes e excecoes.",
        ancoras=("LGPD:art.5", "LGPD:art.6", "LGPD:art.7",),
    ),
    TipoDeTarefa.DEVERES_DOS_AGENTES: DescricaoDaTarefa(
        TipoDeTarefa.DEVERES_DOS_AGENTES,
        "identificar obrigacoes de controlador, operador e encarregado",
        "Pergunte quais deveres recaem sobre um agente de tratamento numa situacao dada; a resposta "
        "lista as obrigacoes do dispositivo, distingue os papeis e menciona a responsabilidade "
        "correspondente.",
        ancoras=("LGPD:art.5", "LGPD:art.46",),
    ),
    TipoDeTarefa.DIREITOS_DO_TITULAR: DescricaoDaTarefa(
        TipoDeTarefa.DIREITOS_DO_TITULAR,
        "aplicar os direitos do titular a um pedido concreto",
        "Descreva um pedido feito por um titular (acesso, correcao, eliminacao, revisao, portabilidade) "
        "e pergunte como o controlador deve proceder; a resposta identifica o direito, o prazo ou a "
        "forma previstos e as hipoteses em que o pedido pode ser recusado ou limitado.",
        ancoras=("LGPD:art.5", "LGPD:art.18",),
    ),
    TipoDeTarefa.INCIDENTE: DescricaoDaTarefa(
        TipoDeTarefa.INCIDENTE,
        "orientar a resposta a um incidente de seguranca",
        "Narre um incidente de seguranca com dados pessoais e pergunte o que a organizacao deve fazer; "
        "a resposta separa providencias operacionais e juridicas, trata a comunicacao a autoridade e "
        "aos titulares e distingue o que esta na lei do que depende de regulamento.",
        ancoras=("LGPD:art.48", "LGPD:art.46",),
    ),
    TipoDeTarefa.IDENTIFICACAO_DE_DADO: DescricaoDaTarefa(
        TipoDeTarefa.IDENTIFICACAO_DE_DADO,
        "classificar dados como pessoais, sensiveis ou anonimizados",
        "Apresente um conjunto de informacoes sobre uma pessoa (com marcadores no lugar de dados reais) "
        "e pergunte como cada uma se classifica; a resposta classifica item a item segundo as "
        "definicoes legais e explica a consequencia da classificacao.",
        ancoras=("LGPD:art.5", "LGPD:art.11", "LGPD:art.12",),
    ),
    TipoDeTarefa.SEM_RESPOSTA: DescricaoDaTarefa(
        TipoDeTarefa.SEM_RESPOSTA,
        "reconhecer o que a lei NAO responde",
        "Formule uma pergunta plausivel sobre o tema do dispositivo cuja resposta NAO esta no texto "
        "da lei em contexto -- porque depende de regulamento da autoridade, de fato nao informado ou "
        "esta' fora do escopo da LGPD. A resposta CORRETA e' abster-se: dizer que a lei nao resolve o "
        "ponto, explicar por que, indicar o que faltaria (norma, fato, fonte) e NAO inventar prazo, "
        "valor ou regra. `artigos_citados` pode ficar vazio ou trazer so' o dispositivo que delimita o "
        "escopo.",
    ),
}


def instrucoes_da_taxonomia() -> str:
    linhas = ["TIPOS DE TAREFA (o campo `tipo` de cada pedido diz qual produzir):"]
    for descricao in TAXONOMIA.values():
        linhas.append(f"- {descricao.tipo.value}: {descricao.objetivo}. {descricao.instrucao}")
    return "\n".join(linhas)
