"""`scripts/baixar_lgpd.py`: a limpeza do HTML do Planalto, sem rede."""

from __future__ import annotations

from scripts.baixar_lgpd import extrair_texto, limpar, recortar_corpo_da_lei

_HTML = """
<html><body>
<p>Presidência da República</p>
<p align="center">CAPÍTULO I<br/>DISPOSIÇÕES PRELIMINARES</p>
<p class="Artigo"><span>Art. 1º Esta Lei dispõe sobre o tratamento.</span></p>
<p class="Artigo"><span>Parágrafo único. Normas gerais.&nbsp;<a href="#art65">Vigência</a></span></p>
<p class="Artigo"><span style="color: black; text-decoration:line-through">VIII - redação antiga;</span></p>
<p class="Artigo"><span>VIII - redação nova; <a href="x">(Redação dada pela Lei nº 13.853, de 2019)</a></span></p>
<p class="Artigo"><strike>§ 1º texto revogado inteiro.</strike></p>
<p class="Artigo"><span style="text-decoration: line-through">§ 2º (VETADO).</span></p>
<p class="Artigo"><span>§ 2º .</span></p>
<p class="Artigo"><span>Art. 5 7. (VETADO).</span></p>
<p class="Artigo"><span>I-A – dia 1º de agosto de 2021;</span></p>
<p class="Artigo"><span>(Convertida na Lei nº 14.058, de 2020)</span></p>
<p>Brasília , 14 de agosto de 2018; 197º da Independência.</p>
<p>MICHEL TEMER</p>
<script>var x = 1;</script>
</body></html>
""".encode("windows-1252")


def test_extrai_limpa_e_recorta():
    paragrafos = recortar_corpo_da_lei(extrair_texto(_HTML))
    assert paragrafos == [
        "CAPÍTULO I DISPOSIÇÕES PRELIMINARES",
        "Art. 1º Esta Lei dispõe sobre o tratamento.",
        "Parágrafo único. Normas gerais.",
        "VIII - redação nova;",
        "Art. 57. (VETADO).",
        "I-A – dia 1º de agosto de 2021;",
    ]


def test_limpar_remove_anotacoes_e_marcadores_orfaos():
    assert limpar(["§ 3º . Vigência", "(Vide Lei nº 1)", "Art. 2º Texto. (Incluído pela Lei nº 2)", "V -;"]) == ["Art. 2º Texto."]


def test_texto_riscado_nunca_vaza():
    paragrafos = extrair_texto(_HTML)
    assert not any("antiga" in p or "revogado" in p for p in paragrafos)
