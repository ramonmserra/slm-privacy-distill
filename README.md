# GENA — Destilação de SLM em Privacidade e Proteção de Dados

Plataforma FastAPI que executa a pesquisa *Utilização da destilação de
conhecimento e Dados Sintéticos Aplicados a SLMs no Domínio de Privacidade e
Proteção de Dados* (TCC — AKCIT/UFG, 2026): um LLM **professor** acessado por
API gera dados sintéticos e cadeias de raciocínio sobre a LGPD; os dados são
**filtrados** por incerteza e por um LLM **juiz**; o **aluno** (um SLM local)
é treinado fora da API e **avaliado** contra o professor com o mesmo juiz.

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/framework-FastAPI-009688)
![Status](https://img.shields.io/badge/etapa-0%20funda%C3%A7%C3%A3o-orange)

## Sumário

- [Estado atual](#estado-atual)
- [Como funciona](#como-funciona)
- [Requisitos](#requisitos)
- [Rodando localmente](#rodando-localmente)
- [Variáveis de ambiente](#variáveis-de-ambiente)
- [Validando a instalação](#validando-a-instalação)
- [Testes e qualidade](#testes-e-qualidade)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Troubleshooting](#troubleshooting)
- [Documentação](#documentação)

## Estado atual

O projeto é construído em etapas, cada uma validável sozinha:

| Etapa | Conteúdo | Estado |
|---|---|---|
| 0 — Fundação | API, autenticação, camada única de LLM (professor/juiz/aluno), telemetria de custo, orçamento por job, testes, CI | ✅ pronta |
| 1 — Corpus + Síntese | LGPD chunkada por dispositivo com id canônico, BM25, parser de referências, geração sintética pelo professor com dedup e verificação de citações | ✅ pronta |
| 2 — Filtro + Dataset | pré-checagem determinística de citações, self-consistency do professor julgada pelo juiz, rubrica de qualidade, held-out por hash, export JSONL de chat versionado | ✅ pronta |
| 3 — Treino externo | receita SFT+LoRA igual na CPU local (0.6B, validação) e no Colab T4 (1.7B), notebook, GGUF para o Ollama, registro e verificação de runs | ✅ pronta |
| 4 — Avaliação | qualquer modelo responde ao banco inicial ou ao held-out; verificação determinística de citações, juiz cego com rubrica, métricas com intervalo de Wilson, comparação pareada (McNemar, bootstrap), custo e latência | ✅ pronta |
| 5 — On-policy + relatórios | autodestilação on-policy (aluno amostra, juiz seleciona, SFT por rejection sampling e pares DPO), receita DPO, anotação humana com kappa, tabela das condições em JSON/Markdown, comparações pareadas e custos por etapa | ✅ pronta |

Todas as etapas de software estão implementadas. O que resta é execução
com chaves e GPU: gerar, filtrar, treinar no Colab, avaliar e preencher as
tabelas do TCC.

## Como funciona

```
FONTES ──▶ CORPUS ──▶ SÍNTESE ──▶ FILTRO ──▶ DATASET ──▶ TREINO (externo) ──▶ AVALIAÇÃO
LGPD,       artigos    professor    incerteza   held-out    SFT/LoRA na CPU      held-out,
ANPD        chunkados  (OpenAI)     + juiz      por hash    ou Colab T4          juiz + regras,
                                    (Anthropic)                                   custo
```

Três papéis de modelo, cada um com provedor e modelo definidos no `.env`:

| Papel | Função | Default |
|---|---|---|
| `teacher` | gera dados sintéticos e raciocínio | `openai` / `gpt-5.6-luna` |
| `judge` | avalia professor **e** aluno; nunca é destilado | `anthropic` / `claude-sonnet-5` |
| `student` | o SLM medido, servido localmente (Ollama/vLLM) | `local` / `qwen3:1.7b` |

Regra da pesquisa: professor e juiz de **famílias diferentes**. A API avisa
em `GET /api/v1/usage/providers` quando coincidem.

**Onde entra o RAG.** A síntese é *geração ancorada em recuperação*: a LGPD
é quebrada em dispositivos com id canônico (`LGPD:art.7:inc.IX`) e indexada
por BM25. Para cada exemplo pedido, o gerador recupera o artigo-alvo e os
artigos relacionados e os entrega ao professor como contexto; o professor só
pode citar ids que recebeu, e cada id citado é conferido no corpus por
código, não por confiança. O mesmo índice servirá à condição de controle
"aluno com RAG sem treino" e ao rastro de embasamento do juiz.

## Requisitos

- Python 3.11
- Docker (para o Ollama que serve o aluno localmente)
- Chave da OpenAI (professor) e, para o juiz, chave da Anthropic
- Para treinar: CPU local (receita de validação, aluno 0.6B) e Google Colab
  com T4 (runs reais, aluno 1.7B). Ver `treino/` a partir da Etapa 3.

## Rodando localmente

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
copy .env.example .env
```

Edite o `.env`: o bloco **OBRIGATÓRIO** tem `APP_ENV`, `AUTH_MODE`,
`AUTH_API_KEYS` e `OPENAI_API_KEY`. O servidor **não atende sem
autenticação** (`AUTH_MODE=disabled` só vale com `APP_ENV=test`). Cada
variável está comentada no `.env.example`.

### Servidor local do aluno (Ollama)

```powershell
docker compose up -d ollama
docker compose exec ollama ollama pull qwen3:1.7b
```

Sem GPU utilizável nesta máquina (MX450 de 2 GB), o Ollama roda em CPU. É
o cenário on-premise que a pesquisa mede. A API sobe sem ele; só a
avaliação do aluno precisa dele.

### Subir a API

```powershell
uvicorn main:app --reload
```

- Swagger em `http://localhost:8000/docs`. Clique em **Authorize** e informe
  a chave de `AUTH_API_KEYS` no header `X-API-Key`.
- **Painel** em `http://localhost:8000/ui/` (a raiz redireciona para lá):
  front mínimo, sem build, para a parte humana do fluxo. Digite a chave uma
  vez; ela fica só na sessão da aba. Três telas:
  - **Anotação**: calibração do juiz. Mostra pergunta, resposta, referência
    e o texto dos artigos; esconde o veredito do juiz por padrão; pula o que
    você já anotou; mostra o kappa do run.
  - **Inspeção**: exemplos e descartes da síntese, decisões do filtro
    (aprovadas/reprovadas com motivos e reamostragens), amostras do
    on-policy e itens com erro da avaliação, com paginação.
  - **Painel**: saúde e provedores, custo por etapa, todos os jobs, runs de
    avaliação com métricas e intervalo, comparação pareada e a tabela das
    condições com rótulos, em HTML e em Markdown para o TCC.
  Disparar síntese, filtro e treino continua pelo Swagger ou pelo guia:
  são operações caras e raras, e ficam fora do painel de propósito.
- Saúde: `GET /health/live` (processo vivo) e `GET /health/ready` (diz o que
  falta: autenticação, `DATA_DIR`, credencial do professor, Ollama).

## Variáveis de ambiente

Tudo o que é parametrizável mora no `.env` (ignorado pelo git). Inventário
completo e comentado em `.env.example`; as que decidem se o sistema atende:

| Variável | Obrigatória | Efeito |
|---|---|---|
| `APP_ENV` | sim | `test` / `development` / `production`; desconhecido vira `production` |
| `AUTH_MODE` | sim | `api_key` ou `disabled` (só em `test`). Sem ela a API responde 503 |
| `AUTH_API_KEYS` | com `api_key` | `chave[:nome]`, separadas por vírgula |
| `OPENAI_API_KEY` | sim | Credencial do professor |
| `ANTHROPIC_API_KEY` | para o juiz | Credencial do juiz (default Anthropic) |
| `TEACHER_PROVIDER`, `JUDGE_PROVIDER`, `STUDENT_PROVIDER` | não | Provedor de cada papel (`openai`, `anthropic`, `local`) |
| `{PROVEDOR}_MODEL_{PAPEL}` | não | Modelo do papel no provedor, ex.: `OPENAI_MODEL_TEACHER=gpt-5.6-luna` |
| `LOCAL_LLM_BASE_URL`, `LOCAL_MODEL_STUDENT` | não | Endpoint OpenAI-compatible e tag do aluno (`http://localhost:11434/v1`, `qwen3:1.7b`) |
| `LLM_BUDGET_USD_POR_JOB` | não | Teto de custo estimado por job (default 5.0; 0 desliga) |
| `DATA_DIR` | não | Onde ficam datasets, telemetria e artefatos (default `data/`) |

## Validando a instalação

Cada etapa tem uma validação própria. Para a **Etapa 0**:

1. **Suíte sem rede** (não gasta nada):
   ```powershell
   python -m pytest
   python -m pyflakes app/ tests/ scripts/
   ```
2. **Configuração dos papéis** (não gasta nada):
   ```powershell
   curl -H "X-API-Key: SUA_CHAVE" http://localhost:8000/api/v1/usage/providers
   curl http://localhost:8000/health/ready
   ```
   O corpo lista provedor, modelo e origem de cada papel e os `avisos`
   (credencial ausente, professor e juiz na mesma família).
3. **Uma chamada real por papel** (custa centavos; o retorno diz quanto):
   ```powershell
   curl -X POST -H "X-API-Key: SUA_CHAVE" http://localhost:8000/api/v1/usage/ping/teacher
   curl -X POST -H "X-API-Key: SUA_CHAVE" http://localhost:8000/api/v1/usage/ping/judge
   curl -X POST -H "X-API-Key: SUA_CHAVE" http://localhost:8000/api/v1/usage/ping/student
   ```
   Prova credencial, transporte, saída estruturada e telemetria de ponta a
   ponta. Depois, `GET /api/v1/usage/llm` mostra o consumo registrado por
   etapa (`diagnostico`) e por modelo.

Para a **Etapa 1** (corpus e síntese):

1. **Texto da LGPD** já está versionado em `recursos/fontes/lgpd.txt`. Para
   rebaixar do Planalto (e conferir se a lei mudou):
   ```powershell
   python -m scripts.baixar_lgpd
   ```
2. **Ingerir e consultar o corpus** (não gasta nada):
   ```powershell
   curl -X POST -H "X-API-Key: SUA_CHAVE" -H "Content-Type: application/json" -d "{\"fonte\":\"LGPD\"}" http://localhost:8000/api/v1/corpus/ingerir
   curl -H "X-API-Key: SUA_CHAVE" "http://localhost:8000/api/v1/corpus/dispositivos/LGPD:art.7:inc.IX"
   curl -H "X-API-Key: SUA_CHAVE" "http://localhost:8000/api/v1/corpus/buscar?q=incidente%20de%20seguranca&k=3"
   ```
   Esperado: 79 artigos e cerca de 450 dispositivos; a busca por incidente
   devolve o art. 48 em primeiro.
3. **Um job pequeno de síntese** (custa centavos; o job respeita
   `LLM_BUDGET_USD_POR_JOB`):
   ```powershell
   curl -X POST -H "X-API-Key: SUA_CHAVE" -H "Content-Type: application/json" -d "{\"quantidade\":10,\"tamanho_do_lote\":5}" http://localhost:8000/api/v1/sintese/jobs
   curl -H "X-API-Key: SUA_CHAVE" http://localhost:8000/api/v1/sintese/jobs/ID_DO_JOB
   curl -H "X-API-Key: SUA_CHAVE" http://localhost:8000/api/v1/sintese/jobs/ID_DO_JOB/exemplos
   curl -H "X-API-Key: SUA_CHAVE" http://localhost:8000/api/v1/sintese/jobs/ID_DO_JOB/descartados
   ```
   O status traz contadores (gerados, aceitos, descartados por motivo) e o
   custo. Leia os exemplos: `citacoes_validas=false` marca o que o professor
   citou e não existe no corpus. A validação da etapa é o grupo inspecionar
   um lote de ~200 exemplos antes de gerar em volume.

Para a **Etapa 2** (filtro e dataset):

1. **Filtrar um job de síntese** (custa: k reamostragens do professor + 1
   julgamento por lote; `k_reamostragens: 0` desliga a self-consistency):
   ```powershell
   curl -X POST -H "X-API-Key: SUA_CHAVE" -H "Content-Type: application/json" -d "{\"jobs_de_sintese\":[\"ID_SINTESE\"],\"k_reamostragens\":3}" http://localhost:8000/api/v1/filtro/jobs
   curl -H "X-API-Key: SUA_CHAVE" http://localhost:8000/api/v1/filtro/jobs/ID_FILTRO
   curl -H "X-API-Key: SUA_CHAVE" "http://localhost:8000/api/v1/filtro/jobs/ID_FILTRO/decisoes?aprovado=false"
   ```
   Cada decisão traz `motivos` nomeados (citação inválida, incerteza alta,
   sem lastro, incompleto...), o veredito do juiz e as reamostragens. A
   validação é o grupo ler as reprovações e confirmar que fazem sentido.
2. **Construir o dataset** a partir dos aprovados (e, para a condição de
   controle, um "sem filtro" a partir do job de síntese inteiro):
   ```powershell
   curl -X POST -H "X-API-Key: SUA_CHAVE" -H "Content-Type: application/json" -d "{\"nome\":\"v1\",\"jobs_de_filtro\":[\"ID_FILTRO\"]}" http://localhost:8000/api/v1/datasets
   curl -X POST -H "X-API-Key: SUA_CHAVE" -H "Content-Type: application/json" -d "{\"nome\":\"v1-sem-filtro\",\"jobs_de_sintese_sem_filtro\":[\"ID_SINTESE\"]}" http://localhost:8000/api/v1/datasets
   curl -H "X-API-Key: SUA_CHAVE" "http://localhost:8000/api/v1/datasets/VERSAO/amostra?split=treino&limite=3"
   curl -H "X-API-Key: SUA_CHAVE" -o treino.jsonl http://localhost:8000/api/v1/datasets/VERSAO/arquivo/treino
   ```
   O manifesto traz contagens por split e por tipo. O held-out é decidido
   pelo hash do id do exemplo: o mesmo exemplo cai sempre no mesmo lado, em
   qualquer versão, então o held-out da v1 nunca vira treino na v2.

Para a **Etapa 3** (treino, fora da API; detalhes em `treino/README.md`):

1. **Validar a receita na CPU** com o aluno pequeno e poucos exemplos (15 a
   40 min; prova o pipeline, não treina para valer):
   ```powershell
   python -m venv .venv-treino; .venv-treino\Scripts\Activate.ps1
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -r treino/requirements-treino.txt
   python -m treino.sft_lora --perfil cpu-local --dataset treino.jsonl --held-out held_out.jsonl --dataset-versao VERSAO --saida saida-cpu --dry-run
   python -m treino.sft_lora --perfil cpu-local --dataset treino.jsonl --held-out held_out.jsonl --dataset-versao VERSAO --saida saida-cpu --max-exemplos 40 --max-eval 10
   ```
2. **Treinar no Colab (T4)** com `treino/colab_treino_lgpd.ipynb`: mesmo
   script, `--perfil colab-t4` (Qwen3-1.7B). O notebook gera o GGUF
   quantizado e o `Modelfile`.
3. **Servir no Ollama e registrar na API**:
   ```powershell
   ollama create qwen3-1.7b-lgpd-VERSAO -f Modelfile
   curl -X POST -H "X-API-Key: SUA_CHAVE" -H "Content-Type: application/json" --data-binary "@run.json" http://localhost:8000/api/v1/treinos
   curl -X POST -H "X-API-Key: SUA_CHAVE" http://localhost:8000/api/v1/treinos/ID_DO_RUN/verificar
   ```
   `verificar` confirma que o Ollama serve o `modelo_local_id`; a partir daí
   o aluno treinado é um `local:<id>` como qualquer outro para a avaliação.

Para a **Etapa 4** (avaliação):

1. **Avaliar um modelo no banco inicial** (34 itens; custa o juiz mais, se o
   modelo for de API, as respostas). `modelo` aceita um papel (`teacher`,
   `student`) ou um id qualificado (`local:qwen3-1.7b-lgpd-VERSAO`):
   ```powershell
   curl -X POST -H "X-API-Key: SUA_CHAVE" -H "Content-Type: application/json" -d "{\"modelo\":\"teacher\"}" http://localhost:8000/api/v1/avaliacao/runs
   curl -X POST -H "X-API-Key: SUA_CHAVE" -H "Content-Type: application/json" -d "{\"modelo\":\"local:qwen3-1.7b-lgpd-VERSAO\"}" http://localhost:8000/api/v1/avaliacao/runs
   curl -H "X-API-Key: SUA_CHAVE" http://localhost:8000/api/v1/avaliacao/runs/ID_DO_RUN
   curl -H "X-API-Key: SUA_CHAVE" "http://localhost:8000/api/v1/avaliacao/runs/ID_DO_RUN/itens?so_erros=true"
   ```
   O run traz `metricas`: acurácia com intervalo de 95%, taxa de alucinação
   com intervalo, artigos inexistentes citados, abstenções adequadas,
   categorias de erro, resultado por tipo, latência p50/p95, tokens e custo.
2. **No held-out de um dataset** (`"conjunto":"held_out","dataset_versao":"VERSAO"`)
   e, para a condição de controle sem treino, com `"com_rag":true`.
3. **Comparar dois runs item a item** (mesmo conjunto):
   ```powershell
   curl -H "X-API-Key: SUA_CHAVE" "http://localhost:8000/api/v1/avaliacao/comparar?a=RUN_PROFESSOR&b=RUN_ALUNO"
   ```
   Devolve quais itens melhoraram e pioraram, o delta com intervalo por
   bootstrap, o p-valor de McNemar e uma leitura em uma frase. Runs com
   `versao_da_medida` diferente (prompt do juiz, regras ou conjunto mudaram)
   são marcados como comparação indicativa.

Para a **Etapa 5** (on-policy, calibração e relatórios):

1. **Rodada on-policy** com o aluno já treinado e servido no Ollama. Os
   prompts vêm do split de treino do dataset, nunca do held-out:
   ```powershell
   curl -X POST -H "X-API-Key: SUA_CHAVE" -H "Content-Type: application/json" -d "{\"dataset_versao\":\"VERSAO\",\"aluno\":\"local:qwen3-1.7b-lgpd-VERSAO\",\"amostras_por_prompt\":4}" http://localhost:8000/api/v1/on-policy/jobs
   curl -H "X-API-Key: SUA_CHAVE" http://localhost:8000/api/v1/on-policy/jobs/ID_DO_JOB
   curl -H "X-API-Key: SUA_CHAVE" -o sft.jsonl http://localhost:8000/api/v1/on-policy/jobs/ID_DO_JOB/arquivo/sft
   curl -H "X-API-Key: SUA_CHAVE" -o dpo.jsonl http://localhost:8000/api/v1/on-policy/jobs/ID_DO_JOB/arquivo/dpo
   ```
   `sft.jsonl` treina com `treino/sft_lora.py` (mesmo formato do dataset);
   `dpo.jsonl` com `treino/dpo_lora.py --adapter` continuando o LoRA do SFT.
   O resultado é um novo `local:<id>` a registrar e avaliar como os demais.
2. **Calibração humana do juiz**: cada autor anota itens de um run de
   avaliação; o resumo devolve o kappa juiz × humanos e entre humanos:
   ```powershell
   curl -X POST -H "X-API-Key: SUA_CHAVE" -H "Content-Type: application/json" -d "{\"item_id\":\"Q7.1\",\"anotador\":\"ramon\",\"correto\":true,\"grounded\":true}" http://localhost:8000/api/v1/avaliacao/runs/ID_DO_RUN/anotacoes
   curl -H "X-API-Key: SUA_CHAVE" http://localhost:8000/api/v1/avaliacao/runs/ID_DO_RUN/calibracao
   ```
   Meta da pesquisa: 100 ou mais itens anotados por pelo menos dois autores.
3. **Tabelas do TCC** direto dos runs:
   ```powershell
   curl -H "X-API-Key: SUA_CHAVE" "http://localhost:8000/api/v1/relatorios/condicoes.md?runs=RUN_C0,RUN_C1,RUN_C3&rotulos=RUN_C0:C0%20professor,RUN_C1:C1%20Qwen%20base,RUN_C3:C3%20SFT%20filtrado"
   curl -H "X-API-Key: SUA_CHAVE" "http://localhost:8000/api/v1/relatorios/pareadas?referencia=RUN_C0&candidatos=RUN_C1,RUN_C3"
   curl -H "X-API-Key: SUA_CHAVE" http://localhost:8000/api/v1/relatorios/custos
   ```
   A versão `.md` já vem com intervalos de 95% e nota de rodapé para colar
   no texto. Figuras podem ser geradas a partir do JSON.

## Testes e qualidade

```powershell
pytest                     # suíte inteira, sem LLM real
pytest -m "not lento"      # ciclo rápido
pytest -m llm              # chamadas reais, custam dinheiro
python -m pyflakes app/ tests/ scripts/
pip install radon; python -m radon cc -n D app/   # deve sair vazio
```

Sem rede e sem LLM real: todo serviço de LLM é um `Protocol` com dublê. O
CI (`.github/workflows/ci.yml`) roda instalação limpa, pyflakes, radon,
pytest e auditoria de dependências.

## Estrutura do projeto

```
app/
  main.py              FastAPI, lifespan, health
  api/v1/endpoints/    rotas HTTP: usage, corpus, sintese, filtro, datasets, treinos, avaliacao, on_policy, relatorios
  ui/                  front mínimo (index.html, app.js, app.css) servido em /ui
  core/                config, auth, llm (papéis e provedores), llm_pricing, llm_usage, llm_gate, stats, observability, erros
  services/corpus/     chunking por dispositivo, parser de referências, BM25, ingestão
  services/sintese/    taxonomia, serviço do professor (Protocol + impl), dedup, engine de jobs
  services/filtro/     pré-checagem de citações, reamostragem do professor, juiz com rubrica, decisão
  services/datasets/   prompt do aluno (fixo e versionado), split por hash, construtor e export JSONL
  services/treino/     registro de runs e verificação no servidor local
  services/avaliacao/  conjuntos, modelo medido + juiz (Protocol + impl), métricas, comparação pareada, calibração humana, engine
  services/on_policy/  aluno amostra, juiz seleciona, SFT por rejection sampling e pares DPO
  services/relatorios/ tabela das condições (JSON e Markdown)
  services/lotes.py    laço compartilhado de lotes concorrentes sob orçamento
  services/injecao.py  guardrail anti-injeção e verificação determinística da saída
  schemas/             contratos Pydantic
  repositories/        persistência em arquivo (escrita atômica)
recursos/              insumos versionados: banco inicial de perguntas, texto da LGPD
scripts/               baixar_lgpd (baixa e limpa o texto consolidado do Planalto)
treino/                receitas SFT+LoRA e DPO+LoRA (CPU local e Colab), notebook, GGUF/Modelfile para o Ollama
scripts/               utilitários de linha de comando
tests/                 suíte automatizada
docs/                  arquitetura e fluxos
data/                  saída de runtime (fora do git)
```

## Troubleshooting

| Sintoma | Causa provável | O que fazer |
|---|---|---|
| Tudo responde 503 | `AUTH_MODE` ausente/inválido, ou `disabled` fora de `APP_ENV=test` | Conferir `/health/ready` → `auth_mode` e o `.env` |
| 401 em toda rota | Header `X-API-Key` ausente ou chave fora de `AUTH_API_KEYS` | Conferir a chave; o Swagger tem o botão Authorize |
| `/health/ready` com `credencial_professor: falha` | `OPENAI_API_KEY` ausente | Preencher no `.env` e reiniciar |
| `ping/student` falha com conexão recusada | Ollama parado ou `LOCAL_LLM_BASE_URL` errada | `docker compose up -d ollama`; conferir `/health/ready` → `servidor_local_do_aluno` |
| `ping/student` falha com saída inválida | Servidor local sem suporte a `json_schema` | Atualizar o Ollama; a avaliação do aluno usa texto livre, só o ping exige JSON |
| 409 `orcamento_do_job_excedido` | Job atingiu `LLM_BUDGET_USD_POR_JOB` | Subir o teto ou reduzir o lote; o custo acumulado está em `/usage/llm` → `por_job` |
| 503 `provedor_saturado` | Disjuntor aberto após 429/5xx seguidos | Aguardar `Retry-After`; conferir cota da conta do provedor |
| Aviso "professor e juiz no MESMO provedor" | `JUDGE_PROVIDER` igual a `TEACHER_PROVIDER` | Apontar o juiz para outra família |
| 409 "Corpus LGPD nao ingerido" | Nenhuma ingestão neste `DATA_DIR` | `POST /api/v1/corpus/ingerir` |
| Job de síntese termina `falhou` com todos os lotes em erro | Credencial do professor ausente ou modelo recusado | Conferir `/health/ready` e o log do processo; `ping/teacher` isola o problema |
| Muitos descartes `duplicata_no_job` | Lote pequeno sobre poucos artigos | Aumentar `quantidade` e variar `tipos`; o dedup é por similaridade de tokens |
| Filtro reprova quase tudo por `incerteza_alta` | Temperatura de reamostragem alta demais ou limiar baixo | Ajustar `temperatura_de_reamostragem` (0,8) e `limiar_de_incerteza` (0,34); ler as reamostragens nas decisões |
| 409 ao construir dataset | Nenhum exemplo aprovado nas origens | Conferir `aprovados` no job de filtro; usar `jobs_de_sintese_sem_filtro` para a condição de controle |
| `POST /treinos` responde 404 | `dataset_versao` do `run.json` não existe nesta API | Treinar sobre um dataset exportado por esta instância; a versão está no manifesto |
| `verificar` diz `disponivel: false` | `ollama create` não rodou, ou nome diferente de `modelo_local_id` sem o prefixo `local:` | `ollama list`; criar com o nome exato |
| Treino no Colab falha com erro de parâmetro do TRL | Versão do TRL fora do piso (`trl>=0.19`) | `pip install -U trl`; rodar o `--dry-run` e um treino com `--max-exemplos 8` antes do run completo |
| Run de avaliação com muitos itens `erro` | Ollama parado, modelo não criado, ou timeout | `/health/ready` → `servidor_local_do_aluno`; `ollama list`; subir `LLM_TIMEOUT_S` para alunos lentos em CPU |
| Comparação marcada como indicativa | `versao_da_medida` diferente entre os runs | Reavaliar os dois modelos na mesma versão (mesmo prompt do juiz, mesmo conjunto, mesmo RAG) |
| Job aparece `falhou` com "interrompido por reinicio do processo" | A API reiniciou no meio do job (jobs rodam no processo) | O que foi gravado ficou; criar o job de novo. Não reinicie a API com job em curso |
| On-policy com `pares_dpo: 0` | Todas as amostras de cada prompt caíram do mesmo lado | Subir `amostras_por_prompt` ou `temperatura`; DPO precisa de uma aprovada e uma reprovada no mesmo prompt |
| On-policy com poucas linhas SFT | Aluno ainda fraco: poucos prompts com amostra aprovada | `fallback_para_o_professor: true` mantém a resposta do professor nesses prompts (marcada como `origem: professor`) |
| Calibração sem `kappa_correto_maioria` | Itens com um só anotador ou empate 1×1 | Dois ou mais autores anotando os mesmos itens; três evita empate |

## Documentação

| Documento | Conteúdo |
|---|---|
| [`docs/ARQUITETURA.md`](docs/ARQUITETURA.md) | Como é construído: camadas, módulos, contratos, fluxos explicativos |
| [`docs/GUIA_DE_EXECUCAO.md`](docs/GUIA_DE_EXECUCAO.md) | Passo a passo para rodar a pesquisa inteira, etapa por etapa, com pontos de verificação |
| [`recursos/avaliacao/banco_inicial_lgpd.jsonl`](recursos/avaliacao/banco_inicial_lgpd.jsonl) | 34 perguntas de avaliação com artigos de referência e critérios esperados |
| [`recursos/fontes/lgpd.txt`](recursos/fontes/lgpd.txt) | Texto consolidado da LGPD, limpo e versionado (fonte e data no cabeçalho) |
