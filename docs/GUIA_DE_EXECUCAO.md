# Guia de execução — passo a passo pelo Swagger e pelo painel

Este guia diz **o que rodar, em que ordem, o que deve aparecer e o que o
grupo precisa avaliar** em cada passo. Toda chamada à API pode ser feita
pelo Swagger em `http://localhost:8000/docs`; a leitura humana dos
resultados é feita no painel em `http://localhost:8000/ui/`. Só três coisas
acontecem fora deles: subir os serviços (terminal), treinar (Google Colab)
e colocar o modelo treinado no Ollama (terminal).

## Como usar o Swagger

1. Abra `http://localhost:8000/docs`.
2. Clique em **Authorize** (canto superior direito), cole a chave que está em
   `AUTH_API_KEYS` no `.env` e confirme. Sem isso toda rota responde 401.
3. As rotas ficam agrupadas por tag: `usage`, `corpus`, `sintese`, `filtro`,
   `datasets`, `treinos`, `avaliacao`, `on-policy`, `relatorios`.
4. Clique na rota, em **Try it out**, preencha o corpo ou os parâmetros,
   **Execute**, e leia o **Response body**.
5. Jobs (síntese, filtro, avaliação, on-policy) respondem `202` na hora e
   rodam em segundo plano. Para acompanhar, chame o `GET .../{id}` da mesma
   tag até `status` deixar de ser `executando`.

## Como usar o painel (`/ui/`)

Abra `http://localhost:8000/ui/`, cole a mesma chave no campo do topo e
clique **Entrar**. A chave fica só na sessão da aba. Três telas:

| Tela | Para quê | Controles |
|---|---|---|
| **Painel** | saúde e provedores, custo por etapa, lista de todos os jobs, runs de avaliação com métricas, comparação pareada, tabela das condições | botão **atualizar**; selects **A (referência)** e **B (candidato)** + **Comparar**; caixa de seleção por run + campos de rótulo + **Gerar tabela** (HTML e Markdown) |
| **Inspeção** | ler o conteúdo que cada job produziu, com paginação | select **Fonte** (síntese exemplos/descartados, filtro aprovados/reprovados, on-policy amostras/reprovadas, avaliação erros/todos), select **Job / run**, **Por página**, setas |
| **Anotação** | calibração humana do juiz | select **Run**, campo **Anotador**, caixa "pular itens que já anotei", caixa "mostrar veredito do juiz" (deixe desmarcada), **Carregar itens**, **Salvar e próximo**, **Recalcular** kappa |

## Como baixar os dados de cada etapa

Tudo o que um passo produz pode ser baixado de três jeitos. Cada passo
abaixo tem um bloco **Download** dizendo qual usar.

1. **Botão Download do Swagger.** Toda resposta JSON no Swagger tem um
   botão **Download** logo abaixo do **Response body**. Serve para guardar o
   manifesto, o job, o run ou uma página de itens como `.json`.
2. **Rotas de arquivo.** Datasets e on-policy têm rotas que devolvem o
   JSONL inteiro; o Swagger mostra um link **Download file**. Para as rotas
   paginadas (exemplos, decisões, itens), o limite é 500 por página; use
   `offset` para as páginas seguintes, ou a função abaixo.
3. **Arquivos em disco.** A API grava tudo em `DATA_DIR` (default `data/`
   na raiz do projeto). É a cópia completa e serve para backup e para os
   gráficos do grupo:

   | Etapa | Arquivo em `DATA_DIR` |
   |---|---|
   | Corpus | `corpus/lgpd.jsonl` e `corpus/lgpd.manifesto.json` |
   | Síntese | `sintese/jobs/<id>.json`, `sintese/<id>/exemplos.jsonl`, `sintese/<id>/descartados.jsonl` |
   | Filtro | `filtro/jobs/<id>.json`, `filtro/<id>/decisoes.jsonl` |
   | Dataset | `datasets/<versao>/manifesto.json`, `treino.jsonl`, `held_out.jsonl` |
   | Treino | `treinos/<id>.json` (o `run.json` registrado) |
   | Avaliação | `avaliacao/runs/<id>.json`, `avaliacao/<id>/itens.jsonl`, `avaliacao/<id>/anotacoes.jsonl` |
   | On-policy | `on_policy/jobs/<id>.json`, `on_policy/<id>/amostras.jsonl`, `sft.jsonl`, `dpo.jsonl` |
   | Custo | `llm_usage.json` |

Para baixar pelo terminal, defina uma vez por sessão:

```powershell
$B = "http://localhost:8000/api/v1"
$H = @{ "X-API-Key" = "SUA_CHAVE" }
# rota de arquivo (JSONL inteiro)
function Baixar($rota, $destino) { Invoke-WebRequest -Headers $H -OutFile $destino "$B/$rota" }
# rota paginada: percorre as paginas de 500 e grava um JSON com tudo
function BaixarPaginado($rota, $destino) {
  $sep = if ($rota.Contains("?")) { "&" } else { "?" }
  $tudo = @(); $offset = 0
  do { $pag = @(Invoke-RestMethod -Headers $H "$B/$rota${sep}limite=500&offset=$offset"); $tudo += $pag; $offset += 500 } while ($pag.Count -eq 500)
  $tudo | ConvertTo-Json -Depth 20 | Out-File $destino -Encoding utf8
}
```

A função aceita rotas com filtro (por exemplo `decisoes?aprovado=false`).

Convenção: **ID** entre colchetes (por exemplo `[ID_SINTESE_1]`) é um valor
copiado da resposta de um passo anterior. Anote todos: são reutilizados.

---

## Ordem de execução (resumo)

| # | Passo | Onde roda | Custo | Pré-requisito |
|---|---|---|---|---|
| 1 | Subir Ollama e API | terminal | 0 | `.env` preenchido |
| 2 | Checar saúde e provedores | Swagger / Painel | 0 | 1 |
| 3 | Ping nos três modelos | Swagger | centavos | 2 |
| 4 | Ingerir o corpus da LGPD | Swagger | 0 | 2 |
| 5 | Piloto de síntese (200) e leitura pelo grupo | Swagger + Inspeção | ~0,5 USD | 4 |
| 6 | Síntese em volume (2 jobs de 2.000) | Swagger | 10 a 20 USD | 5 aprovado |
| 7 | Filtro e leitura das reprovações | Swagger + Inspeção | 20 a 40 USD | 6 concluído |
| 8 | Datasets (filtrado e sem filtro) e download | Swagger | 0 | 7 concluído |
| 9 | Treinar no Colab (C3, C2, C5) | Colab | 0 | 8 |
| 10 | Servir no Ollama e registrar o run | terminal + Swagger | 0 | 9 |
| 11 | Avaliar cada condição | Swagger + Painel + Inspeção | 5 a 30 USD | 4 e 10 |
| 12 | Comparar item a item | Painel ou Swagger | 0 | 11 |
| 13 | Calibração humana do juiz | Anotação | 0 | 11 |
| 14 | On-policy (C4) | Swagger + Inspeção + Colab | 10 a 20 USD | 10 e 11 |
| 15 | Tabelas e custos do TCC | Painel ou Swagger | 0 | 11 |

Condições experimentais que os passos produzem:

| Condição | O que é | Como se obtém |
|---|---|---|
| C0 | Professor (teto) | avaliar `modelo: "teacher"` |
| C1 | Qwen3-1.7B base, sem treino | avaliar `modelo: "student"` |
| C2 | SFT no dataset **sem filtro** | treinar no Colab com `v1-sem-filtro` e avaliar `local:<id>` |
| C3 | SFT no dataset **filtrado** (principal) | treinar no Colab com `v1-filtrado` e avaliar `local:<id>` |
| C4 | C3 + rodada on-policy | job on-policy sobre C3, treinar de novo no Colab, avaliar |
| C5 | Manacá-1B com a receita de C3 | treinar no Colab com `--modelo-base menezesbruno/manaca-1b-base` |
| C6 | Qwen base + RAG, sem treino | avaliar `modelo: "student"` com `com_rag: true` |

---

## Passo 1 — Subir Ollama e API (terminal)

Uma vez por máquina:

```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
copy .env.example .env
```

No `.env`, preencha `AUTH_API_KEYS`, `OPENAI_API_KEY` (professor) e
`ANTHROPIC_API_KEY` (juiz). Os demais valores do exemplo servem.

Toda sessão de trabalho, em dois terminais:

```powershell
# terminal 1: aluno local
docker compose up -d ollama
docker compose exec ollama ollama pull qwen3:1.7b      # só na primeira vez

# terminal 2: API
.venv\Scripts\Activate.ps1
uvicorn main:app --reload
```

**Resultado esperado:** o terminal 2 mostra `Application startup complete`
e nenhuma linha `ERROR` sobre `AUTH_MODE`. `http://localhost:8000/docs`
abre o Swagger e `http://localhost:8000/ui/` abre o painel.

**O que avaliar:** nada ainda. Só confirme que `python -m pytest` passa
(testes sem rede, não gastam nada).

---

## Passo 2 — Checar saúde e provedores (0 custo)

**Swagger:**

1. Tag `health` → `GET /health/ready`.

   **Esperado:** `200`, `"status": "ok"`. Em `checks`:
   `credencial_professor: "ok"`, `credencial_juiz: "ok"`,
   `servidor_local_do_aluno: "ok"`, `avisos: []`.

   Se vier `503` com `"status": "degraded"`, o próprio `checks` diz o que
   falta: `auth_mode: "unset"` é `.env` sem `AUTH_MODE`;
   `credencial_professor: "falha"` é `OPENAI_API_KEY` vazia;
   `servidor_local_do_aluno: "indisponivel"` é Ollama fora do ar (a API
   funciona, mas o ping do aluno e a avaliação do aluno não).

2. Tag `usage` → `GET /api/v1/usage/providers`.

   **Esperado:** três entradas em `papeis`, todas com
   `credencial_presente: true`, e `avisos` vazio:

   | papel | provedor | modelo |
   |---|---|---|
   | teacher | openai | gpt-5.6-luna |
   | judge | anthropic | claude-sonnet-5 |
   | student | local | qwen3:1.7b |

**No front:** tela **Painel**, bloco **Saúde e provedores**. Deve mostrar
os mesmos três papéis com credencial presente e nenhum aviso. Um aviso de
"professor e juiz na mesma família" significa `.env` errado: a pesquisa
exige famílias diferentes.

**O que avaliar:** os três modelos listados são exatamente os que vão para
o texto do TCC. Anote provedor e modelo de cada papel agora; eles entram
na seção de reprodutibilidade.

---

## Passo 3 — Ping nos três modelos (Swagger, centavos)

Tag `usage` → `POST /api/v1/usage/ping/{papel}`. Rode três vezes, com
`papel` igual a `teacher`, depois `judge`, depois `student`. Sem corpo.

**Esperado em cada um:** `200` e um corpo como:

```json
{
  "papel": "teacher",
  "modelo": "openai:gpt-5.6-luna",
  "latencia_ms": 1800,
  "resposta": { "saudacao": "Olá!", "numero": 42 },
  "custo_estimado_usd": 0.0002
}
```

O que importa é `resposta.numero` igual a `42`. O ping do aluno em CPU
pode levar de 10 a 60 segundos.

**Se falhar:** `503` diz que a credencial do papel está ausente. Se o ping
do aluno responder erro de saída inválida, o Ollama está numa versão sem
suporte a `json_schema`; não bloqueia a avaliação, que usa texto livre.

**No front:** tela **Painel**, bloco **Custo por etapa**: deve aparecer a
etapa `diagnostico` com custo de centavos. É a prova de que a telemetria
de custo funciona.

**O que avaliar:** `latencia_ms` do aluno. Se passar de 60 segundos num
ping de 256 tokens, a avaliação do aluno (respostas de até 1.024 tokens)
vai precisar de `concorrencia: 1` e `LLM_TIMEOUT_S` maior no `.env`.

---

## Passo 4 — Ingerir o corpus (Swagger, 0 custo)

1. Tag `corpus` → `POST /api/v1/corpus/ingerir` → corpo:

   ```json
   { "fonte": "LGPD" }
   ```

   **Esperado:** `200` com `artigos: 79` e `dispositivos: 452`.

2. Tag `corpus` → `GET /api/v1/corpus/dispositivos/{id_}` com
   `id_ = LGPD:art.7:inc.IX`.

   **Esperado:** `200`; `dispositivo.texto` fala de "legítimo interesse" e
   `artigo_completo` traz o art. 7º inteiro.

3. Tag `corpus` → `GET /api/v1/corpus/buscar` com `q = incidente de
   seguranca` e `k = 3`.

   **Esperado:** 3 itens; o primeiro com `artigo_id: "LGPD:art.48"`.

O corpus fica gravado em `DATA_DIR`; não precisa reingerir a cada subida
da API. Se algum job responder `409` dizendo "corpus vazio", repita o
item 1.

**O que avaliar:** faça mais duas ou três buscas com termos do domínio
(`consentimento`, `dados sensiveis`, `encarregado`) e confira se o artigo
certo vem em primeiro (arts. 8º, 11 e 41). O retrieval é o mesmo que a
síntese e a condição C6 usam; se ele errar aqui, erra lá.

**No front:** não há tela de corpus. Tudo pelo Swagger.

**Download:** o texto-fonte já está no repositório em
`recursos/fontes/lgpd.txt`. O corpus chunkado fica em
`data/corpus/lgpd.jsonl` (um dispositivo por linha, com id canônico) e o
resumo em `data/corpus/lgpd.manifesto.json`. Não há rota de arquivo;
`GET /api/v1/corpus` devolve o resumo para o botão **Download** do Swagger.

---

## Passo 5 — Piloto de síntese (~0,5 USD)

Objetivo: gerar 200 exemplos e o grupo ler antes de gastar em volume.

**Swagger:**

1. Tag `sintese` → `POST /api/v1/sintese/jobs` → corpo:

   ```json
   { "quantidade": 200, "tamanho_do_lote": 5, "concorrencia": 3, "seed": 42 }
   ```

   **Esperado:** `202` com `status: "pendente"` e um `id`. Anote como
   `[ID_PILOTO]`.

2. Tag `sintese` → `GET /api/v1/sintese/jobs/{job_id}`. Repita a cada minuto.

   **Esperado ao final (5 a 15 minutos):** `status: "concluido"`,
   `aceitos` entre 150 e 200, `custo_usd` abaixo de 1, `lotes_com_erro`
   zero ou quase. `descartes_por_motivo` diz por que o resto saiu
   (`duplicata_no_job`, `citacoes_invalidas`, `vazamento_do_banco_inicial`).

   `status: "orcamento_excedido"` significa que o job parou no teto
   (`LLM_BUDGET_USD_POR_JOB`, default 5 USD). O que foi gerado está salvo.

**No front:** tela **Inspeção** → Fonte **Síntese — exemplos** → Job
`[ID_PILOTO]` → **Carregar**. Cada cartão mostra tipo, dificuldade,
pergunta, raciocínio, resposta, ids citados e se as citações são válidas.
Depois troque a Fonte para **Síntese — descartados** e leia os motivos.
A tela **Painel**, bloco **Jobs**, mostra o progresso do job sem precisar
do Swagger.

**O que avaliar (o grupo, em 30 a 50 exemplos, de tipos variados):**

| Critério | Onde olhar | Aprova se |
|---|---|---|
| Pergunta autocontida e plausível para o domínio | `pergunta` | um profissional de privacidade faria essa pergunta |
| Raciocínio cita ids que existem | `citacoes_validas`, `artigos_citados_invalidos` | `true` e lista vazia na grande maioria |
| Resposta não inventa prazo, valor ou regra | `resposta` contra os `contexto_ids` | nenhuma afirmação sem dispositivo que a sustente |
| Abstenção correta | exemplos com `tipo: "sem_resposta"` | a resposta diz que não dá para concluir e o que falta |
| Sem dado pessoal real | `pergunta` e `resposta` | só marcadores como `[PESSOA]`, `[CPF]`, `[EMPRESA]` |
| Cobertura de tipos e dificuldades | cabeçalho dos cartões | os 8 tipos aparecem; não é só `conceito` básico |
| Descartes fazem sentido | Fonte **Síntese — descartados** | duplicatas são mesmo duplicatas; vazamento do banco inicial foi pego |

**Download:** para o grupo ler fora do painel (planilha, impressão), use
o botão **Download** do Swagger em `GET /api/v1/sintese/jobs/{job_id}/exemplos`
com `limite = 500`, ou pelo terminal:

```powershell
BaixarPaginado "sintese/jobs/[ID_PILOTO]/exemplos" piloto-exemplos.json
Invoke-RestMethod -Headers $H "$B/sintese/jobs/[ID_PILOTO]/descartados" | ConvertTo-Json -Depth 10 | Out-File piloto-descartados.json -Encoding utf8
```

Os mesmos dados estão em `data/sintese/[ID_PILOTO]/exemplos.jsonl` e
`descartados.jsonl`; o job em `data/sintese/jobs/[ID_PILOTO].json`.

**Critério para avançar:** o grupo aprovou a amostra e registrou a
`versao_do_prompt` que leu (está no job e em cada exemplo). Se reprovou,
ajuste o prompt em `app/services/sintese/llm_service.py` (a versão muda
sozinha), rode `python -m pytest` e repita o passo 5. Não gere em volume
com um prompt que o grupo não leu.

---

## Passo 6 — Síntese em volume (Swagger, 10 a 20 USD)

Dois jobs com seeds diferentes, para cobrir os artigos em ordens distintas.

Tag `sintese` → `POST /api/v1/sintese/jobs`, duas vezes:

```json
{ "quantidade": 2000, "tamanho_do_lote": 5, "concorrencia": 4, "seed": 1, "teto_usd": 15 }
```

```json
{ "quantidade": 2000, "tamanho_do_lote": 5, "concorrencia": 4, "seed": 2, "teto_usd": 15 }
```

Anote os ids como `[ID_SINTESE_1]` e `[ID_SINTESE_2]`. Podem rodar ao mesmo
tempo. Acompanhe com `GET /api/v1/sintese/jobs/{job_id}`.

**Esperado ao final (1 a 3 horas cada):** `status: "concluido"`, `aceitos`
na faixa de 1.700 a 1.900 por job (soma de 3.000 a 4.000, o alvo da
pesquisa), `custo_usd` de 2 a 4 USD por mil exemplos,
`descartes_por_motivo.vazamento_do_banco_inicial` em poucos casos.

**No front:** tela **Painel**, bloco **Jobs**: os dois jobs com status,
contadores e custo, atualizados pelo botão **atualizar**. O bloco **Custo
por etapa** mostra a etapa `sintese` crescendo; compare com o teto que você
definiu.

**O que avaliar:**

- `aceitos` total atingiu a meta de 3.000? Se não, um terceiro job com
  `seed: 3` resolve; a deduplicação entre jobs acontece no filtro e no
  dataset.
- `lotes_com_erro` acima de 5% do total indica instabilidade do provedor;
  releia o `erro` do job antes de seguir.
- Leitura rápida (10 exemplos por job) na tela **Inspeção**, só para
  confirmar que a qualidade é a mesma do piloto. A `versao_do_prompt` deve
  ser a que o grupo aprovou.

**Download:** igual ao passo 5, trocando o id:
`BaixarPaginado "sintese/jobs/[ID_SINTESE_1]/exemplos" sintese1.json`.
Em disco: `data/sintese/[ID_SINTESE_1]/exemplos.jsonl`. Guarde também o
job (`GET /api/v1/sintese/jobs/{job_id}` → **Download**): ele tem a
`versao_do_prompt`, o `modelo` e o `custo_usd` que vão para o TCC.

**Não reinicie a API** com job `executando`: o job morre com o processo e
é marcado `falhou`. O que já foi gravado permanece e pode entrar no filtro.

---

## Passo 7 — Filtro (20 a 40 USD)

O filtro reprova por citação inválida (sem custo), por incerteza do
professor (k reamostragens, julgadas pelo juiz) e pela rubrica do juiz.

**Swagger:**

1. Tag `filtro` → `POST /api/v1/filtro/jobs` → corpo (com os seus ids):

   ```json
   {
     "jobs_de_sintese": ["[ID_SINTESE_1]", "[ID_SINTESE_2]"],
     "k_reamostragens": 3,
     "temperatura_de_reamostragem": 0.8,
     "limiar_de_incerteza": 0.34,
     "teto_usd": 40
   }
   ```

   **Esperado:** `202` com um `id`. Anote como `[ID_FILTRO]`.

   Para um ensaio barato antes do filtro de verdade, rode primeiro sobre
   `[ID_PILOTO]` com `"k_reamostragens": 0` e `"teto_usd": 3`.

2. Tag `filtro` → `GET /api/v1/filtro/jobs/{job_id}`. Repita a cada minutos.

   **Esperado ao final (2 a 5 horas):** `status: "concluido"`;
   `aprovados / avaliados` entre 50% e 85%; `reprovacoes_por_motivo` com
   `sem_lastro_nas_citacoes` e `incerteza_alta` entre os principais. São
   esses motivos que a hipótese da pesquisa quer tirar do treino.

**No front:** tela **Inspeção** → Fonte **Filtro — reprovados** → Job
`[ID_FILTRO]`. Cada cartão mostra o id do exemplo, o tipo, APROVADO ou
REPROVADO, a `incerteza` (score), a `nota` do juiz, os **Motivos**, os
**Problemas (juiz)** e, num bloco expansível, as reamostragens do
professor. O cartão não mostra a pergunta nem a resposta: para lê-las,
troque a Fonte para **Síntese — exemplos** e procure o mesmo id, ou use
`GET /api/v1/filtro/jobs/{job_id}/decisoes` no Swagger, que traz o
veredito completo (`correcao_juridica`, `lastro_nas_citacoes`,
`completude`). Depois Fonte **Filtro — aprovados** para o contraste.

**O que avaliar (o grupo, 20 reprovações e 10 aprovações):**

| Pergunta | Onde olhar | Esperado |
|---|---|---|
| A reprovação está certa? | `motivos` e `veredito.problemas` | em pelo menos 16 das 20 o grupo concorda com o juiz |
| `incerteza_alta` é incerteza mesmo? | as 3 `reamostragens` contra a resposta original | as reamostragens chegam a conclusão diferente; se chegam à mesma com outras palavras, o juiz está rígido demais |
| O filtro deixou passar algo ruim? | 10 aprovados | nenhuma afirmação sem lastro; `nota` 4 ou 5 |
| Taxa de aprovação | `aprovados / avaliados` no job | 50% a 85%; fora disso, calibração errada |
| Reprovação por tipo | `reprovacoes_por_motivo` e tipos nos cartões | `sem_resposta` não pode ter sido reprovado em massa por `abstencao_inadequada` |

**Download:** as decisões completas (veredito do juiz, reamostragens,
motivos) saem de `GET /api/v1/filtro/jobs/{job_id}/decisoes`. Pelo Swagger,
`aprovado = false` e `limite = 500` e o botão **Download**; pelo terminal:

```powershell
BaixarPaginado "filtro/jobs/[ID_FILTRO]/decisoes?aprovado=false" reprovadas.json
BaixarPaginado "filtro/jobs/[ID_FILTRO]/decisoes" decisoes.json
```

Em disco: `data/filtro/[ID_FILTRO]/decisoes.jsonl` tem todas as decisões
numa linha por exemplo, e `data/filtro/jobs/[ID_FILTRO].json` o resumo
com `reprovacoes_por_motivo` e `versao_do_filtro`.

Se quase tudo reprovar por `incerteza_alta`, leia as reamostragens antes
de subir o `limiar_de_incerteza`. Se o juiz estiver errado com frequência,
o problema é o prompt do juiz, e ele precisa ser corrigido antes de
qualquer avaliação, porque é o mesmo juiz do passo 11.

---

## Passo 8 — Datasets e download (Swagger, 0 custo)

Dois datasets: o filtrado (C3) e o sem filtro (C2, controle).

1. Tag `datasets` → `POST /api/v1/datasets` → corpo:

   ```json
   { "nome": "v1-filtrado", "jobs_de_filtro": ["[ID_FILTRO]"] }
   ```

   **Esperado:** `201` com um manifesto. Anote `versao` como
   `[VERSAO_FILTRADO]`.

2. De novo `POST /api/v1/datasets` → corpo:

   ```json
   { "nome": "v1-sem-filtro", "jobs_de_sintese_sem_filtro": ["[ID_SINTESE_1]", "[ID_SINTESE_2]"] }
   ```

   **Esperado:** `201`. Anote `versao` como `[VERSAO_SEM_FILTRO]`.

3. Amostra: `GET /api/v1/datasets/{versao}/amostra` com `split = held_out`
   e `limite = 3`.

   **Esperado:** cada linha tem `id`, `tipo` e `messages` com três turnos
   (`system`, `user`, `assistant`); o `assistant` traz raciocínio e
   conclusão.

4. Download para o treino: `GET /api/v1/datasets/{versao}/arquivo/{split}`
   com `[VERSAO_FILTRADO]` e `split = treino`, depois `split = held_out`.
   O Swagger mostra um link **Download file**; salve como `treino.jsonl` e
   `held_out.jsonl`. Repita para `[VERSAO_SEM_FILTRO]` em outra pasta.

   Pelo terminal, se preferir:

   ```powershell
   $H = @{ "X-API-Key" = "SUA_CHAVE" }
   Invoke-WebRequest -Headers $H -OutFile treino.jsonl   "http://localhost:8000/api/v1/datasets/[VERSAO_FILTRADO]/arquivo/treino"
   Invoke-WebRequest -Headers $H -OutFile held_out.jsonl "http://localhost:8000/api/v1/datasets/[VERSAO_FILTRADO]/arquivo/held_out"
   ```

**O que avaliar, no manifesto de cada dataset:**

| Campo | Esperado |
|---|---|
| `treino.total + held_out.total` | igual a `aprovados` do filtro (filtrado) ou a `aceitos` somados menos duplicatas (sem filtro) |
| `held_out.total` | cerca de 15% do total |
| `held_out.por_tipo` | os 8 tipos presentes; nenhum com menos de 10 itens |
| `duplicados_entre_origens` | pequeno; se for grande, as seeds cobriram os mesmos artigos |
| `versoes_do_prompt_de_sintese` | só a versão que o grupo aprovou |
| `versoes_do_filtro` | uma versão só |

**No front:** não há tela de datasets. O manifesto fica no Swagger.

**Download:** o item 4 acima já é o download dos splits. Guarde também o
manifesto (`GET /api/v1/datasets/{versao}` → **Download**): é o documento
que identifica o dataset no TCC. Em disco, tudo fica em
`data/datasets/[VERSAO]/` (`manifesto.json`, `treino.jsonl`, `held_out.jsonl`).

**Regra:** o `held_out` nunca entra em treino nem no on-policy. O split é
por hash do id; não copie linhas entre os arquivos.

---

## Passo 9 — Treinar no Colab (C3, C2 e C5)

O treino é sempre no Google Colab com GPU T4. Esta máquina não treina.

1. Abra `treino/colab_treino_lgpd.ipynb` no Colab e selecione
   **Ambiente de execução → Alterar tipo → GPU (T4)**. A primeira célula
   (`nvidia-smi`) deve listar uma Tesla T4.
2. Célula 1 (código do projeto): ajuste `REPO_URL` para o repositório do
   grupo antes de rodar. Ela clona e instala `treino/requirements-treino.txt`.
3. Célula 2 (dataset): cole `[VERSAO_FILTRADO]` (ou `[VERSAO_SEM_FILTRO]`)
   em `DATASET_VERSAO` e envie `treino.jsonl` e `held_out.jsonl` quando o
   seletor de arquivos abrir.
4. Célula 3 (dry-run): deve terminar sem erro e imprimir a contagem de
   exemplos e o plano do treino. Não usa GPU.
5. Antes do treino completo, duplique a célula 4 (treino) e acrescente
   `--max-exemplos 8 --saida saida-teste` ao comando. Se terminar em poucos
   minutos, as bibliotecas estão compatíveis. Se falhar por parâmetro
   desconhecido do TRL, acrescente `!pip install -q -U trl` na célula 1 e
   repita.
6. Célula 4 (treino) original, com `--perfil colab-t4`. A célula seguinte
   imprime o resumo do `run.json`.
7. Célula 5 (GGUF) e célula 6 (download de `modelo-q4_k_m.gguf`,
   `Modelfile` e `run.json`).

Um run por condição, 1 a 2 horas cada:

| Run | Dataset enviado | Opções extras |
|---|---|---|
| C3 (principal) | `v1-filtrado` | nenhuma |
| C2 (controle) | `v1-sem-filtro` | nenhuma |
| C5 | `v1-filtrado` | `--modelo-base menezesbruno/manaca-1b-base --prefixo-do-assistente ""` e Modelfile com template próprio |

**Esperado por run:** o dry-run imprime a contagem de exemplos sem erro;
o `run.json` final tem `status: "concluido"`, `metricas.train_loss`,
`metricas.eval_loss` e `modelo_local_id` no formato
`local:qwen3-1.7b-lgpd-xxxxxxxx` (os 8 caracteres vêm da versão do
dataset). Guarde os três arquivos de cada run numa pasta própria com o
nome da condição.

**O que avaliar no `run.json`:**

| Campo | Esperado |
|---|---|
| `dataset_versao` | igual à `versao` do manifesto do passo 8 (senão o registro falha) |
| `exemplos_de_treino` | igual a `treino.total` do manifesto |
| `metricas.eval_loss` | abaixo do `train_loss` do início; se subir entre épocas, overfit |
| `hiperparametros` | iguais entre C2 e C3 (só o dataset muda; é o que isola o efeito do filtro) |
| `duracao_s`, `hardware` | anotar para o TCC |

**No front:** nada até o passo 10.

---

## Passo 10 — Servir no Ollama e registrar o run (terminal + Swagger)

Para cada run do passo 9. O nome no Ollama é o `modelo_local_id` do
`run.json` sem o prefixo `local:`.

Terminal, na pasta com os arquivos do run:

```powershell
docker cp .\modelo-q4_k_m.gguf gena-ollama:/tmp/
docker cp .\Modelfile gena-ollama:/tmp/
docker compose exec -w /tmp ollama ollama create qwen3-1.7b-lgpd-xxxxxxxx -f Modelfile
docker compose exec ollama ollama run qwen3-1.7b-lgpd-xxxxxxxx "O que e dado pessoal sensivel?"
```

**Esperado:** `ollama create` termina com `success`; o `ollama run`
responde em português citando ids no formato `LGPD:art.N`. Em CPU leva de
30 segundos a alguns minutos.

**Swagger:**

1. Tag `treinos` → `POST /api/v1/treinos` → corpo: abra o `run.json` num
   editor, copie o conteúdo inteiro e cole no campo de corpo.

   **Esperado:** `201` com o mesmo conteúdo mais `id` e
   `disponivel_localmente: null`. Anote `id` como `[ID_RUN_C3]` (ou C2, C5).

   `404`: o `dataset_versao` do `run.json` não existe nesta API. `409`: já
   há um run concluído com esse `modelo_local_id`; cada modelo treinado
   precisa de um id novo.

2. Tag `treinos` → `POST /api/v1/treinos/{run_id}/verificar`.

   **Esperado:** `200` com `disponivel: true` e o id dentro de
   `modelos_servidos`. `disponivel: false` significa que o nome do
   `ollama create` não bate com o `modelo_local_id`.

**O que avaliar:** a resposta do `ollama run`. Ela precisa (a) estar em
português, (b) citar ao menos um id `LGPD:art.N` e (c) não terminar
truncada no meio de uma frase. Se vier em inglês ou sem citações, o
Modelfile está com template errado e a avaliação vai medir o template, não
o treino.

**No front:** nada específico de treinos. Os runs aparecem como opções de
modelo quando você cria avaliações.

**Download:** o run registrado sai de `GET /api/v1/treinos/{run_id}`
(**Download**) ou de `GET /api/v1/treinos` para todos de uma vez. Em
disco: `data/treinos/[ID_RUN].json`. O GGUF, o Modelfile e o `run.json`
originais ficam na pasta de cada condição que você baixou do Colab; a API
não os guarda.

---

## Passo 11 — Avaliar cada condição (5 a 30 USD)

Cada avaliação é um run em `POST /api/v1/avaliacao/runs` (tag `avaliacao`).
O modelo responde ao conjunto, a API confere as citações no corpus e o juiz
compara com a referência. Primeiro no **banco inicial** (34 itens, rápido e
barato), depois no **held-out** (volume estatístico).

### 11.1 Banco inicial

Um `POST` por condição. Anote o `id` de cada resposta.

| Condição | Corpo | Anotar como |
|---|---|---|
| C0 | `{ "modelo": "teacher" }` | `[AV_C0]` |
| C1 | `{ "modelo": "student" }` | `[AV_C1]` |
| C6 | `{ "modelo": "student", "com_rag": true, "k_rag": 3 }` | `[AV_C6]` |
| C3 | `{ "modelo": "local:qwen3-1.7b-lgpd-xxxxxxxx" }` | `[AV_C3]` |
| C2 | `{ "modelo": "local:qwen3-1.7b-lgpd-yyyyyyyy" }` | `[AV_C2]` |

Para C3 e C2 use o `modelo_local_id` completo do `run.json`, com `local:`.
`student` resolve para `LOCAL_MODEL_STUDENT` do `.env`, o Qwen base.

Rode **uma avaliação de aluno por vez**: o Ollama em CPU não atende duas
em paralelo com velocidade útil. C0 pode rodar junto com qualquer uma.

Acompanhe com `GET /api/v1/avaliacao/runs/{run_id}`.

**Esperado ao final:** `status: "concluido"`, `itens: 34`,
`metricas.respondidos: 34`, `metricas.julgados: 34`. Em `metricas`:
`acuracia` com `acuracia_ic95`, `taxa_de_alucinacao`,
`itens_com_artigo_inexistente`, `abstencoes_adequadas`, `latencia_p50_ms`,
`custo_usd`. Tempo: C0 em minutos; alunos em CPU de 30 a 90 minutos.

Se houver `erro` de timeout nos itens, crie o run de novo com
`"concorrencia": 1` e aumente `LLM_TIMEOUT_S` no `.env` (exige reiniciar a
API, portanto sem job em curso).

### 11.2 Held-out

Mesmo endpoint, com o conjunto e a versão do dataset filtrado:

| Condição | Corpo |
|---|---|
| C0 | `{ "modelo": "teacher", "conjunto": "held_out", "dataset_versao": "[VERSAO_FILTRADO]", "teto_usd": 20 }` |
| C3 | `{ "modelo": "local:qwen3-1.7b-lgpd-xxxxxxxx", "conjunto": "held_out", "dataset_versao": "[VERSAO_FILTRADO]" }` |
| C1, C2, C6 | igual ao banco inicial, acrescentando `conjunto` e `dataset_versao` |

Anote como `[HO_C0]`, `[HO_C3]` e assim por diante. Para um ensaio antes do
run inteiro, acrescente `"limite": 30`.

**Esperado:** `itens` igual a `held_out.total` do manifesto; o resto como
em 11.1. Um aluno em CPU com 500 itens leva várias horas: deixe rodando de
noite e não reinicie a API.

**No front:**

- Tela **Painel**, bloco **Runs de avaliação**: uma linha por run com
  Id, Modelo, Conjunto (com `+RAG` quando for C6), Status, **n** (itens já
  julgados), Acurácia [IC95], Alucinação [IC95], Abst. (adequadas /
  esperadas), p50 ms, US$/item e Medida (`versao_da_medida`). Clique
  **atualizar** para ver o `n` crescer; é aqui que se acompanha o andamento
  sem o Swagger. Os detalhes de progresso (`respondidos`,
  `lotes_do_juiz_concluidos`) só aparecem no Swagger.
- Tela **Inspeção** → Fonte **Avaliação — itens com erro** → Run
  `[AV_C3]`. Cada cartão mostra pergunta, referência, resposta do modelo,
  artigos mencionados e inexistentes, e o veredito do juiz com
  `categoria` (`hallucination`, `wrong_legal_basis`, `incomplete_answer`,
  `non_answer`, `out_of_scope_answered`, `format_error`) e
  `justificativa`. Fonte **Avaliação — todos os itens** para ver os
  acertos também.

**O que avaliar, por run:**

| Pergunta | Onde olhar | Esperado |
|---|---|---|
| O run é completo? | `respondidos`, `julgados` | ambos iguais a `itens`; itens sem juiz não entram na acurácia |
| A ordem faz sentido? | `acuracia` entre runs | C0 acima de C3, C3 acima de C1; se C1 bater C0, há erro de id de modelo ou de juiz |
| O aluno inventa artigo? | `itens_com_artigo_inexistente` | perto de zero em C0; é a métrica que o filtro deveria baixar entre C2 e C3 |
| O aluno se abstém quando deve? | `abstencoes_adequadas / abstencoes_esperadas` | alto em C0; é onde alunos pequenos mais erram |
| Que tipo de erro domina? | `categorias_de_erro` | anote por condição; vira parágrafo da discussão |
| Os erros do juiz são justos? | 10 itens com erro na Inspeção | o grupo concorda com a `justificativa` em pelo menos 8; se não, vá ao passo 13 antes de tirar conclusões |
| Latência e custo | `latencia_p50_ms`, `custo_usd_por_item` | entram na tabela do TCC como custo do cenário on-premise |

**Download:** por run, dois arquivos. O resumo com as métricas sai de
`GET /api/v1/avaliacao/runs/{run_id}` (**Download**); os itens, com
resposta do modelo e veredito do juiz, de
`GET /api/v1/avaliacao/runs/{run_id}/itens` com `limite = 500`:

```powershell
Invoke-RestMethod -Headers $H "$B/avaliacao/runs/[AV_C3]" | ConvertTo-Json -Depth 10 | Out-File av_c3.json -Encoding utf8
BaixarPaginado "avaliacao/runs/[AV_C3]/itens" av_c3_itens.json
```

Em disco: `data/avaliacao/runs/[AV_C3].json` e
`data/avaliacao/[AV_C3]/itens.jsonl`. Baixe os itens de **todos** os runs:
são a matéria-prima das figuras por tipo e da análise de erros.

Todos os runs do mesmo conjunto devem ter a mesma `versao_da_medida`. Se
você alterar o prompt do juiz no meio, os runs antigos ficam incomparáveis
e precisam ser refeitos.

---

## Passo 12 — Comparar item a item (0 custo)

**No front:** tela **Painel**, bloco **Comparação pareada**: escolha
**A (referência)** e **B (candidato)** e clique **Comparar**. A `leitura`
aparece em texto e os números logo abaixo.

**Swagger:** tag `avaliacao` → `GET /api/v1/avaliacao/comparar` com `a` e `b`.

| Pergunta da pesquisa | A | B |
|---|---|---|
| Quanto o aluno treinado se aproxima do professor? | `[AV_C0]` | `[AV_C3]` |
| O filtro melhora o treino? | `[AV_C2]` | `[AV_C3]` |
| RAG sem treino vale quanto? | `[AV_C1]` | `[AV_C6]` |
| Treino supera RAG? | `[AV_C6]` | `[AV_C3]` |

**Esperado:** `mesma_versao_da_medida: true`, `itens_em_comum: 34` (ou o
tamanho do held-out), `delta_acuracia` com `delta_ic95_bootstrap`,
`p_mcnemar`, listas `melhoraram` e `pioraram` com ids de itens, e uma
`leitura`.

**O que avaliar:**

- `mesma_versao_da_medida` precisa ser `true`; se for `false`, a comparação
  não vale e um dos runs tem de ser refeito.
- No banco inicial o intervalo será largo (34 itens): use só como
  indicação. A conclusão estatística vem do held-out, onde
  `p_mcnemar` abaixo de 0,05 e intervalo que não cruza zero sustentam a
  afirmação.
- Abra na tela **Inspeção** três ids de `pioraram` da comparação C2 → C3:
  são os casos em que o filtro tirou algo que ajudava. Isso vira discussão
  no texto.

---

## Passo 13 — Calibração humana do juiz (0 custo)

Os três autores anotam os mesmos 100 ou mais itens de um run (sugestão:
`[HO_C3]`), sem ver o veredito do juiz.

**No front:** tela **Anotação** → Run `[HO_C3]` → Anotador com o seu nome
(sempre o mesmo) → deixe **"mostrar veredito do juiz"** desmarcada →
**Carregar itens**. Para cada item: leia pergunta, resposta do modelo,
referência e o texto dos artigos; marque **Correto** (Sim/Não) e
**Grounded** (Sim/Não/Não sei); **Salvar e próximo**. O contador no topo
(`12 / 120`) mostra quantos faltam. Ao final, botão **Recalcular** mostra os kappas.

**Swagger, se preferir:** tag `avaliacao` →
`POST /api/v1/avaliacao/runs/{run_id}/anotacoes` → corpo:

```json
{ "item_id": "[ID_DO_ITEM]", "anotador": "ramon", "correto": true, "grounded": true, "comentario": "" }
```

Resumo: `GET /api/v1/avaliacao/runs/{run_id}/calibracao`.

**Esperado:** `itens_anotados` igual ou acima de 100, três nomes em
`anotadores`, `kappa_correto_maioria` (juiz contra a maioria dos humanos),
`kappa_entre_humanos` e a `leitura`.

**O que avaliar:**

| Resultado | Leitura |
|---|---|
| kappa do juiz acima de 0,6 e humanos acima de 0,6 | juiz validado; as conclusões dos passos 11 e 12 valem |
| kappa do juiz abaixo de 0,6 e humanos acima de 0,6 | o juiz está errado; revisar o prompt do juiz e refazer todas as avaliações |
| humanos abaixo de 0,6 | a rubrica está ambígua; alinhar a definição de "correto" e "grounded" entre os autores e reanotar antes de culpar o juiz |

Use a mesma definição da rubrica do juiz: "correto" é chegar à conclusão
jurídica da referência; "grounded" é toda afirmação normativa ter lastro
nos dispositivos de referência.

**Download:** as anotações dos três autores saem de
`GET /api/v1/avaliacao/runs/{run_id}/anotacoes` (sem filtro de
`anotador`, vem tudo) e o resumo dos kappas de `.../calibracao`, ambos
pelo botão **Download**. Em disco: `data/avaliacao/[HO_C3]/anotacoes.jsonl`.
Esse arquivo é a evidência da calibração; anexe ao TCC.

---

## Passo 14 — On-policy, condição C4 (10 a 20 USD)

O aluno C3 gera várias respostas para cada prompt de treino; o juiz
seleciona. Pré-requisito: C3 servido no Ollama (passo 10).

**Swagger:**

1. Tag `on-policy` → `POST /api/v1/on-policy/jobs` → corpo:

   ```json
   {
     "dataset_versao": "[VERSAO_FILTRADO]",
     "aluno": "local:qwen3-1.7b-lgpd-xxxxxxxx",
     "amostras_por_prompt": 4,
     "temperatura": 0.8,
     "teto_usd": 20
   }
   ```

   Anote o `id` como `[ID_ONPOLICY]`. Para um ensaio, acrescente
   `"limite": 20`.

2. `GET /api/v1/on-policy/jobs/{job_id}` até concluir (em CPU, horas).

   **Esperado:** `status: "concluido"`; `prompts_com_aprovada / prompts`
   acima de 50%; `pares_dpo` na casa das centenas; `linhas_sft` próximo de
   `prompts_com_aprovada`.

3. Download: `GET /api/v1/on-policy/jobs/{job_id}/arquivo/{nome}` com
   `nome = sft` e depois `nome = dpo`. Salve como `sft.jsonl` e `dpo.jsonl`.
   Um terceiro valor, `nome = amostras`, baixa todas as amostras com o
   veredito do juiz, para análise. Pelo terminal:

   ```powershell
   Baixar "on-policy/jobs/[ID_ONPOLICY]/arquivo/sft" sft.jsonl
   Baixar "on-policy/jobs/[ID_ONPOLICY]/arquivo/dpo" dpo.jsonl
   Baixar "on-policy/jobs/[ID_ONPOLICY]/arquivo/amostras" amostras.jsonl
   ```

   Em disco: `data/on_policy/[ID_ONPOLICY]/` com os mesmos três arquivos.

4. No Colab (mesmo notebook, GPU T4), uma das receitas ou as duas como
   condições separadas:

   ```powershell
   python -m treino.sft_lora --perfil colab-t4 --dataset sft.jsonl --dataset-versao [VERSAO_FILTRADO] --saida saida-onpolicy
   python -m treino.dpo_lora --perfil colab-t4 --pares dpo.jsonl --adapter saida/adapter --dataset-versao [VERSAO_FILTRADO] --saida saida-dpo
   ```

5. Servir e registrar (passo 10), avaliar (passo 11) e anotar como
   `[AV_C4]`. Comparar `[AV_C3]` contra `[AV_C4]` (passo 12).

**No front:** tela **Inspeção** → Fonte **On-policy — reprovadas** → Job
`[ID_ONPOLICY]`: cada cartão traz a amostra do aluno, o veredito do juiz e
os artigos inexistentes. Fonte **On-policy — amostras** mostra as
aprovadas também, com `similaridade` com a referência.

**O que avaliar:**

| Pergunta | Onde olhar | Esperado |
|---|---|---|
| O aluno é bom o bastante para se autodestilar? | `prompts_com_aprovada / prompts` | acima de 50%; abaixo disso, registre o número e siga sem C4 |
| As aprovadas são boas mesmo? | 10 aprovadas na Inspeção | o grupo concorda; se o juiz aprova resposta sem lastro, o SFT on-policy vai reforçar erro |
| As reprovadas foram reprovadas pelo motivo certo? | 10 reprovadas, `veredito.categoria` | `hallucination` e `wrong_legal_basis` dominam |
| Há pares DPO suficientes? | `pares_dpo` | centenas; com dezenas, só o SFT on-policy vale a pena |

---

## Passo 15 — Tabelas e custos do TCC (0 custo)

**No front:** tela **Painel**. No bloco **Runs de avaliação**, marque a
caixa de seleção de cada run que entra na tabela (só os do mesmo conjunto).
Para cada run marcado aparece um campo de rótulo no bloco **Tabela das
condições**: escreva `C0 professor`, `C3 SFT filtrado` e assim por diante.
Clique **Gerar tabela**. A tabela aparece em HTML e, logo abaixo, em
Markdown pronto para colar no texto. O bloco **Custo por etapa** é o
número de custo do TCC.

**Swagger:** tag `relatorios`.

1. `GET /api/v1/relatorios/condicoes.md` com:

   - `runs` = `[AV_C0],[AV_C1],[AV_C6],[AV_C2],[AV_C3],[AV_C4]`
   - `rotulos` = `[AV_C0]:C0 professor,[AV_C1]:C1 Qwen base,[AV_C6]:C6 Qwen+RAG,[AV_C2]:C2 SFT sem filtro,[AV_C3]:C3 SFT filtrado,[AV_C4]:C4 on-policy`

   **Esperado:** tabela Markdown, uma linha por condição, com acurácia e
   intervalo, alucinação, abstenção, latência e custo. Repita com os ids do
   held-out. A versão JSON, para gráficos, é `GET /api/v1/relatorios/condicoes`.

2. `GET /api/v1/relatorios/pareadas` com `referencia = [AV_C0]` e
   `candidatos = [AV_C1],[AV_C3],[AV_C4]`.

   **Esperado:** uma comparação pareada por candidato, no formato do passo 12.

3. `GET /api/v1/relatorios/custos`.

   **Esperado:** `por_etapa` com tokens e custo de `sintese`, `filtro`,
   `avaliacao`, `on_policy` e `diagnostico`, e o `total`.

**Download:** o `.md` sai direto do Swagger (botão **Download** no
Response body, que é texto). Para guardar tudo de uma vez pelo terminal:

```powershell
$runs = "[AV_C0],[AV_C1],[AV_C6],[AV_C2],[AV_C3],[AV_C4]"
$rotulos = "[AV_C0]:C0 professor,[AV_C1]:C1 Qwen base,[AV_C6]:C6 Qwen+RAG,[AV_C2]:C2 SFT sem filtro,[AV_C3]:C3 SFT filtrado,[AV_C4]:C4 on-policy"
Invoke-RestMethod -Headers $H "$B/relatorios/condicoes.md?runs=$runs&rotulos=$rotulos" | Out-File tabela_banco_inicial.md -Encoding utf8
Invoke-RestMethod -Headers $H "$B/relatorios/condicoes?runs=$runs&rotulos=$rotulos" | ConvertTo-Json -Depth 10 | Out-File tabela_banco_inicial.json -Encoding utf8
Invoke-RestMethod -Headers $H "$B/relatorios/pareadas?referencia=[AV_C0]&candidatos=[AV_C1],[AV_C3],[AV_C4]" | ConvertTo-Json -Depth 10 | Out-File pareadas.json -Encoding utf8
Invoke-RestMethod -Headers $H "$B/relatorios/custos" | ConvertTo-Json -Depth 10 | Out-File custos.json -Encoding utf8
```

A telemetria bruta de custo, por chamada agregada em etapa e modelo, está
em `data/llm_usage.json`. Ao final da pesquisa, copie a pasta `data/`
inteira para o armazenamento do grupo: é o conjunto completo de evidências.

**O que avaliar antes de colar no TCC:**

- Todas as linhas da tabela têm o mesmo conjunto e a mesma
  `versao_da_medida` (a tabela do held-out é separada da do banco inicial).
- Nenhum run com `julgados` menor que `itens`.
- Os rótulos batem com as condições (confira o `modelo` de cada linha).
- O custo total fecha com a soma dos `custo_usd` dos jobs; se
  `modelos_sem_preco` não estiver vazio, há um modelo sem preço na tabela
  de preços e o custo está subestimado.

---

## O que anotar no TCC para reprodutibilidade

| O quê | Onde está |
|---|---|
| Versão do corpus | cabeçalho de `recursos/fontes/lgpd.txt` |
| Versão do prompt de síntese | `versao_do_prompt` em cada exemplo e job |
| Versão do filtro | `versao_do_filtro` em cada decisão |
| Versão do dataset | `versao` do manifesto |
| Receita de treino | `run.json` registrado em `/treinos` |
| Régua da avaliação | `versao_da_medida` em cada run |
| Modelos usados | `GET /usage/providers` no dia dos runs |
| Custo | `GET /relatorios/custos` ao final |

---

## Cuidados que valem para todos os passos

- **Não reinicie a API com job em curso.** Ao subir de novo, o que estava
  aberto é marcado `falhou`; o gravado permanece.
- **Orçamento.** Todo job aceita `teto_usd`; sem ele vale
  `LLM_BUDGET_USD_POR_JOB` (default 5 USD). O job para antes da chamada que
  estouraria e fica `orcamento_excedido`.
- **Held-out é sagrado.** Nunca entra em treino nem no on-policy.
- **Juiz diferente do professor.** `GET /usage/providers` avisa se
  coincidirem. Nunca destile o juiz.
- **Dados pessoais.** Só marcadores. O filtro reprova nome real por
  `dado_pessoal_real`; confira na amostra do piloto.
- **Um id novo por modelo treinado.** `POST /treinos` recusa `modelo_local_id`
  repetido para run concluído.
- **Cegueira na anotação.** Na tela Anotação, a caixa "mostrar veredito do
  juiz" fica desmarcada durante a calibração; marcá-la invalida o kappa.
