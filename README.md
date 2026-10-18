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
| 2 — Filtro + Dataset | self-consistency, juiz de qualidade, held-out, export JSONL | ⏳ próxima |
| 3 — Treino externo | receitas CPU local (0.6B) e Colab T4 (1.7B), registro de runs | — |
| 4 — Avaliação | backtest com juiz cego, verificação determinística, comparação pareada | — |
| 5 — On-policy + relatórios | rodada on-policy, tabelas e figuras do TCC | — |

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
  api/v1/endpoints/    rotas HTTP: usage, corpus, sintese (filtro, datasets, avaliacao nas próximas etapas)
  core/                config, auth, llm (papéis e provedores), llm_pricing, llm_usage, llm_gate, observability, erros
  services/corpus/     chunking por dispositivo, parser de referências, BM25, ingestão
  services/sintese/    taxonomia, serviço do professor (Protocol + impl), dedup, engine de jobs
  services/injecao.py  guardrail anti-injeção e verificação determinística da saída
  schemas/             contratos Pydantic
  repositories/        persistência em arquivo (escrita atômica)
recursos/              insumos versionados: banco inicial de perguntas, texto da LGPD
scripts/               baixar_lgpd (baixa e limpa o texto consolidado do Planalto)
treino/                receitas de treino (CPU local e Colab) — Etapa 3
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

## Documentação

| Documento | Conteúdo |
|---|---|
| [`docs/ARQUITETURA.md`](docs/ARQUITETURA.md) | Como é construído: camadas, módulos, contratos, fluxos explicativos |
| [`recursos/avaliacao/banco_inicial_lgpd.jsonl`](recursos/avaliacao/banco_inicial_lgpd.jsonl) | 34 perguntas de avaliação com artigos de referência e critérios esperados |
| [`recursos/fontes/lgpd.txt`](recursos/fontes/lgpd.txt) | Texto consolidado da LGPD, limpo e versionado (fonte e data no cabeçalho) |
