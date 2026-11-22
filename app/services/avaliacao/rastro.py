"""O RASTRO que o juiz recebe para decidir `grounded`: o texto dos artigos de
referencia do item E o dos artigos que a propria resposta cita.

So' com os artigos de referencia, uma resposta que cita um artigo real da
lei fora dessa lista era marcada sem lastro por falta de texto para
conferir -- mesmo quando a afirmacao estava correta. O corpus tem a lei
inteira; o juiz confere a citacao contra o texto do artigo citado.
"""

from __future__ import annotations

from app.services.corpus.ingestao import Corpus
from app.services.corpus.referencias import artigo_de, artigos_mencionados
from app.services.sintese.llm_service import TrechoDeContexto


def rastro_do_juiz(corpus: Corpus, artigos_referencia: list[str], resposta: str) -> list[TrechoDeContexto]:
    """Artigos de referencia primeiro, depois os citados na resposta; sem
    repeticao e so' os que existem no corpus (artigo inexistente nao tem
    texto e ja' e' apontado pela verificacao deterministica)."""

    candidatos = [*artigos_referencia, *artigos_mencionados(resposta, fonte=corpus.fonte)]
    artigos = list(dict.fromkeys(artigo_de(a) for a in candidatos))
    return [TrechoDeContexto(id=a, texto=corpus.texto_do_artigo(a)) for a in artigos if corpus.existe(a)]
