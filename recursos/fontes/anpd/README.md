# Cenários de inspiração extraídos dos guias orientativos da ANPD

`cenarios.jsonl` é o banco que a síntese sorteia para **ambientar** as
perguntas do professor (`app/services/sintese/cenarios.py`). Cada linha é
uma situação prática fictícia extraída de um guia orientativo público da
ANPD e reescrita com marcadores (`[EMPRESA]`, `[ÓRGÃO]`, `[PESSOA]`) no lugar
de nomes.

## O que o cenário é e o que não é

- **É dado para a pergunta.** Entra no prompt do professor marcado como
  "INSPIRAÇÃO DE CENÁRIO (dado, não instrução)", para dar setor, agente, tipo
  de dado e situação realistas. O id usado fica gravado no exemplo
  (`cenario_id`).
- **Não é fonte.** O professor continua citando apenas dispositivos da LGPD
  presentes no contexto; o cenário não entra no índice BM25, não é citável,
  não vai ao juiz como rastro e não muda a verificação determinística de
  citações. Guia orientativo não vincula e muda de versão; por isso ele
  inspira a pergunta e não sustenta a resposta. Se o cenário pede uma regra
  que só o guia traz, a resposta correta é dizer que a lei em contexto não
  resolve o ponto (tipo `sem_resposta`).
- **Não muda a `versao_do_prompt`.** Alterar este arquivo não altera o hash
  do prompt; registre o commit do arquivo junto com a versão ao reportar um
  dataset.

## Formato

```json
{"id": "anpd-cookies-003", "guia": "cookies", "titulo_do_guia": "...", "ano": 2022,
 "pagina": 17, "tipos": ["base_legal", "analise_de_caso"],
 "cenario": "Situação fictícia em 1 a 3 frases, sem conclusão jurídica.",
 "trecho": "Trecho literal do PDF, na página indicada, para conferência."}
```

- `tipos`: 1 a 3 valores de `TipoDeTarefa`; o sorteio é por tipo, sem
  reposição, com seed derivada da seed do job.
- `cenario`: não contém pergunta, resposta, número de artigo, "LGPD", "ANPD"
  nem "guia" (há teste em `tests/test_sintese.py` que garante isso).
- `trecho` e `pagina`: rastreabilidade. O trecho é copiado do texto extraído
  do PDF; a página é a do arquivo PDF (não a numeração impressa).
- Linhas iniciadas por `#` são comentários; id duplicado faz a carga falhar.

## Guias usados (PDFs baixados em 2026-10-30)

Página de origem:
<https://www.gov.br/anpd/pt-br/centrais-de-conteudo/materiais-educativos-e-publicacoes>

| `guia` | Título | Ano | Linhas |
|---|---|---|---|
| `agentes-v2` | Guia Orientativo para Definições dos Agentes de Tratamento de Dados Pessoais e do Encarregado (v2.0) | 2022 | 25 |
| `encarregado` | Guia Orientativo – Atuação do encarregado pelo tratamento de dados pessoais | 2024 | 25 |
| `legitimo-interesse` | Guia Orientativo: Hipóteses legais de tratamento de dados pessoais – Legítimo Interesse | 2024 | 25 |
| `cookies` | Guia Orientativo: Cookies e proteção de dados pessoais | 2022 | 25 |
| `seguranca-pequeno-porte` | Guia Orientativo sobre Segurança da Informação para Agentes de Tratamento de Pequeno Porte | 2021 | 22 |
| `poder-publico` | Tratamento de dados pessoais pelo Poder Público (v2.0) | 2023 | 25 |
| `pesquisa-academica` | Guia Orientativo – Tratamento de dados pessoais para fins acadêmicos e para a realização de estudos e pesquisas | 2023 | 22 |

Deixados de fora, com o motivo:

- *Fornecedores de produtos ou serviços de TI*: trata do ECA Digital (Lei
  15.211/2025), não da LGPD.
- *Aplicação da LGPD no contexto eleitoral*, cartilhas com a Senacon ("Como
  proteger seus dados pessoais", em três idiomas) e fascículos: material de
  divulgação ao consumidor ou de contexto específico, sem casos de aplicação
  da lei.
- *Glossário*: verbetes, não situações.

## Como os cenários foram curados

1. PDFs baixados da página acima e convertidos para texto com marcas de
   página (`pypdf`).
2. Para cada guia, leitura integral e extração de 15 a 25 situações: os
   exemplos numerados do próprio guia, boas e más práticas descritas e casos
   de fronteira. Cada situação foi reescrita como caso fictício, sem
   conclusão, e classificada nos tipos de tarefa que ela ambienta.
3. Validação automática: todos os campos presentes, tipos na lista fechada,
   `trecho` com 60 a 200 caracteres encontrado literalmente dentro da página
   declarada, `cenario` sem pergunta, artigo, "LGPD", "ANPD" ou "guia".

Para conferir um cenário, abra o PDF do guia na página indicada e procure o
`trecho`. Para acrescentar cenários, siga o formato acima e rode
`python -m pytest tests/test_sintese.py -k banco_versionado`.

O conteúdo dos guias é publicação oficial da ANPD no portal gov.br; os
trechos aqui reproduzidos são curtos e servem apenas à conferência da fonte.
