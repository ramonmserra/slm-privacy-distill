# Guia de execução — do zero às tabelas do TCC

Passo a passo para rodar a pesquisa inteira com esta API. Cada etapa tem um
**ponto de verificação**: só avance quando ele estiver satisfeito. Os
comandos são PowerShell (Windows); em outro shell, só muda a sintaxe das
aspas.

Convenção usada em todos os comandos: a API em `http://localhost:8000` e a
chave no header `X-API-Key`. Defina uma vez por sessão:

```powershell
$B = "http://localhost:8000/api/v1"
$H = @{ "X-API-Key" = "SUA_CHAVE" }          # a chave de AUTH_API_KEYS no .env
function Post($rota, $corpo) { Invoke-RestMethod -Method Post -Uri "$B/$rota" -Headers $H -ContentType "application/json" -Body $corpo }
function Get($rota)          { Invoke-RestMethod -Uri "$B/$rota" -Headers $H }
```

`Post` e `Get` abaixo são essas funções. O Swagger em
`http://localhost:8000/docs` faz o mesmo pela interface, com o botão
**Authorize**.

---

## Visão geral das condições experimentais

| Condição | O que é | Como se obtém |
|---|---|---|
| C0 | Professor (teto) | Avaliar `modelo: teacher` |
| C1 | Qwen3-1.7B base, zero-shot | Avaliar `modelo: student` com `LOCAL_MODEL_STUDENT=qwen3:1.7b` |
| C2 | SFT no dataset **sem filtro** | Dataset de `jobs_de_sintese_sem_filtro` → treino → avaliar `local:<id>` |
| C3 | SFT no dataset **filtrado** | Dataset de `jobs_de_filtro` → treino → avaliar `local:<id>` |
| C4 | C3 + rodada on-policy | On-policy sobre C3 → treino SFT ou DPO → avaliar |
| C5 | Manacá-1B com a receita de C3 | Treino com `--modelo-base menezesbruno/manaca-1b-base` e `--prefixo-do-assistente ""` |
| C6 | Qwen base + RAG, sem treino | Avaliar `modelo: student` com `com_rag: true` |

Toda condição é avaliada **nos mesmos conjuntos** (banco inicial e held-out
do mesmo dataset) e comparada **item a item** contra C0.

---

## Etapa 0 — Preparar o ambiente

1. Instalar e configurar:
   ```powershell
   python -m venv .venv; .venv\Scripts\Activate.ps1
   pip install -r requirements.txt -r requirements-dev.txt
   copy .env.example .env     # se ainda não existir
   ```
   No `.env`, preencher `OPENAI_API_KEY` (professor) e `ANTHROPIC_API_KEY`
   (juiz). Conferir `AUTH_API_KEYS`, `TEACHER_PROVIDER=openai`,
   `OPENAI_MODEL_TEACHER=gpt-5.6-luna`, `JUDGE_PROVIDER=anthropic`.
   Ajustar `LLM_BUDGET_USD_POR_JOB` ao orçamento (default 5 dólares por job).
2. Subir o Ollama com o aluno base:
   ```powershell
   docker compose up -d ollama
   docker compose exec ollama ollama pull qwen3:1.7b
   ```
3. Subir a API e verificar:
   ```powershell
   uvicorn main:app --reload
   Invoke-RestMethod http://localhost:8000/health/ready
   Get usage/providers
   ```
4. Um ping real por papel (centavos):
   ```powershell
   Post usage/ping/teacher ""
   Post usage/ping/judge ""
   Post usage/ping/student ""
   ```

**Verificação:** `/health/ready` com `status: ok`; os três pings devolvem
`resposta.numero = 42`; `usage/providers` sem `avisos`. Se o ping do aluno
falhar por saída inválida, o Ollama precisa de versão com suporte a
`json_schema`; a avaliação do aluno usa texto livre e não depende disso.

**Testes sem rede** (sempre que mudar código): `python -m pytest`.

---

## Etapa 1 — Corpus e síntese

### 1.1 Corpus

```powershell
Post corpus/ingerir '{"fonte":"LGPD"}'
Get "corpus/dispositivos/LGPD:art.7:inc.IX"
Get "corpus/buscar?q=incidente%20de%20seguranca&k=3"
```

**Verificação:** 79 artigos e 452 dispositivos; a busca por incidente
devolve `LGPD:art.48` em primeiro. Para rebaixar o texto da lei (se a
norma mudou): `python -m scripts.baixar_lgpd` e ingerir de novo.

### 1.2 Piloto de síntese (200 exemplos para o grupo ler)

```powershell
$job = Post sintese/jobs '{"quantidade":200,"tamanho_do_lote":5,"concorrencia":3,"seed":42}'
Get "sintese/jobs/$($job.id)"                       # repetir até status != executando
Get "sintese/jobs/$($job.id)/exemplos?limite=50"
Get "sintese/jobs/$($job.id)/descartados"
```

O que inspecionar, por amostragem de 30 a 50 exemplos:

- a pergunta é autocontida e plausível para o domínio?
- o raciocínio cita ids que existem (`citacoes_validas: true`)?
- a resposta não inventa prazo, valor ou regra fora do contexto?
- `sem_resposta` de fato se abstém e diz o que falta?
- há dado pessoal real? (deve haver só marcadores `[PESSOA]`, `[CPF]`)
- os descartes fazem sentido (`duplicata_no_job`, `vazamento_do_banco_inicial`)?

Se o grupo reprovar o lote, ajuste o prompt em
`app/services/sintese/llm_service.py` (a versão muda sozinha), rode a suíte
e repita o piloto. **Não** gere em volume com um prompt que o grupo não leu.

### 1.3 Síntese em volume

```powershell
$s1 = Post sintese/jobs '{"quantidade":2000,"tamanho_do_lote":5,"concorrencia":4,"seed":1,"teto_usd":15}'
$s2 = Post sintese/jobs '{"quantidade":2000,"tamanho_do_lote":5,"concorrencia":4,"seed":2,"teto_usd":15}'
```

Seeds diferentes cobrem artigos em ordens diferentes; a deduplicação entre
jobs acontece no filtro e no dataset. Acompanhe `custo_usd` no job e em
`Get usage/llm`. Alvo da pesquisa: 3.000 a 4.000 exemplos aceitos.

**Verificação:** `status: concluido`, `descartes_por_motivo` sem
`vazamento_do_banco_inicial` acima de poucos casos, custo dentro do
previsto (ordem de 2 a 4 dólares por mil exemplos com o professor atual).

---

## Etapa 2 — Filtro e datasets

### 2.1 Filtro

```powershell
$f = Post filtro/jobs ('{"jobs_de_sintese":["' + $s1.id + '","' + $s2.id + '"],"k_reamostragens":3,"temperatura_de_reamostragem":0.8,"limiar_de_incerteza":0.34,"teto_usd":40}')
Get "filtro/jobs/$($f.id)"
Get "filtro/jobs/$($f.id)/decisoes?aprovado=false&limite=50"
```

Leia as reprovações. `reprovacoes_por_motivo` diz onde o professor erra
mais: `sem_lastro_nas_citacoes` e `incerteza_alta` são os motivos que a
hipótese da pesquisa espera remover do treino. Se quase tudo reprovar por
`incerteza_alta`, releia as reamostragens antes de mexer no limiar.

Custo: k reamostragens do professor por exemplo mais um julgamento por
lote. Para um piloto barato, `"k_reamostragens":0` avalia só a rubrica.

**Verificação:** `status: concluido`; taxa de aprovação entre 50% e 85%
(fora disso, algo está mal calibrado); amostra de 20 reprovações lida e
confirmada pelo grupo.

### 2.2 Datasets

```powershell
$d_com = Post datasets ('{"nome":"v1-filtrado","jobs_de_filtro":["' + $f.id + '"]}')
$d_sem = Post datasets ('{"nome":"v1-sem-filtro","jobs_de_sintese_sem_filtro":["' + $s1.id + '","' + $s2.id + '"]}')
Get "datasets/$($d_com.versao)"
Get "datasets/$($d_com.versao)/amostra?split=held_out&limite=3"
```

Baixar os arquivos para o treino:

```powershell
Invoke-WebRequest -Headers $H -OutFile treino.jsonl   "$B/datasets/$($d_com.versao)/arquivo/treino"
Invoke-WebRequest -Headers $H -OutFile held_out.jsonl "$B/datasets/$($d_com.versao)/arquivo/held_out"
```

Para a condição "só conclusão" (sem raciocínio), repita com
`"incluir_raciocinio":false`. Anote as `versao` dos datasets: elas entram
no `run.json` do treino e identificam o held-out da avaliação.

**Verificação:** `treino.total + held_out.total` igual ao número de
aprovados; `held_out.por_tipo` com todos os tipos; o mesmo id nunca nos dois
arquivos (a suíte garante por hash, mas confira o manifesto).

---

## Etapa 3 — Treino

### 3.1 Validar a receita na CPU (uma vez)

```powershell
python -m venv .venv-treino; .venv-treino\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r treino/requirements-treino.txt
python -m treino.sft_lora --perfil cpu-local --dataset treino.jsonl --held-out held_out.jsonl --dataset-versao VERSAO --saida saida-cpu --dry-run
python -m treino.sft_lora --perfil cpu-local --dataset treino.jsonl --held-out held_out.jsonl --dataset-versao VERSAO --max-exemplos 40 --max-eval 10 --saida saida-cpu
```

**Verificação:** `saida-cpu/run.json` com `status: concluido`,
`metricas.train_loss` e `eval_loss`; pastas `adapter/` e `merged/`. 15 a 40
minutos. Se falhar por parâmetro do TRL, `pip install -U trl` e repita.

### 3.2 Treinar no Colab (C2, C3 e depois C4/C5)

Abrir `treino/colab_treino_lgpd.ipynb` no Colab com GPU T4 e seguir as
células: clone ou upload de `treino/`, upload de `treino.jsonl` e
`held_out.jsonl`, dry-run, treino com `--perfil colab-t4`, GGUF, download
de `modelo-q4_k_m.gguf`, `Modelfile` e `run.json`. Um run por dataset:

| Run | Dataset | Observação |
|---|---|---|
| C3 | `v1-filtrado` | principal |
| C2 | `v1-sem-filtro` | controle do filtro |
| C5 | `v1-filtrado` | `--modelo-base menezesbruno/manaca-1b-base --prefixo-do-assistente ""`; Modelfile com template próprio |

Antes do run completo, rode um treino com `--max-exemplos 8` para pegar
desvio de API das bibliotecas.

### 3.3 Servir e registrar

Na pasta com o GGUF e o Modelfile (o nome é o `modelo_local_id` do
`run.json` sem o prefixo `local:`):

```powershell
docker cp .\modelo-q4_k_m.gguf gena-ollama:/tmp/; docker cp .\Modelfile gena-ollama:/tmp/
docker compose exec -w /tmp ollama ollama create qwen3-1.7b-lgpd-VERSAO -f Modelfile
docker compose exec ollama ollama run qwen3-1.7b-lgpd-VERSAO "O que e dado pessoal sensivel?"
$run = Invoke-RestMethod -Method Post -Uri "$B/treinos" -Headers $H -ContentType "application/json" -InFile run.json
Post "treinos/$($run.id)/verificar" ""
```

**Verificação:** `verificar` devolve `disponivel: true`; a resposta do
`ollama run` cita ids no formato `LGPD:art.N`.

---

## Etapa 4 — Avaliação

### 4.1 Um run por condição, nos dois conjuntos

Banco inicial (34 itens, rubrica do grupo):

```powershell
$c0 = Post avaliacao/runs '{"modelo":"teacher"}'
$c1 = Post avaliacao/runs '{"modelo":"student"}'                       # LOCAL_MODEL_STUDENT=qwen3:1.7b
$c6 = Post avaliacao/runs '{"modelo":"student","com_rag":true,"k_rag":3}'
$c3 = Post avaliacao/runs '{"modelo":"local:qwen3-1.7b-lgpd-VERSAO"}'
$c2 = Post avaliacao/runs '{"modelo":"local:qwen3-1.7b-lgpd-VERSAO_SEM_FILTRO"}'
```

Held-out (volume estatístico):

```powershell
$h0 = Post avaliacao/runs ('{"modelo":"teacher","conjunto":"held_out","dataset_versao":"' + $d_com.versao + '","teto_usd":20}')
$h3 = Post avaliacao/runs ('{"modelo":"local:qwen3-1.7b-lgpd-VERSAO","conjunto":"held_out","dataset_versao":"' + $d_com.versao + '"}')
```

Acompanhe com `Get "avaliacao/runs/$($c3.id)"`. Alunos em CPU respondem
devagar; `concorrencia: 1` e `LLM_TIMEOUT_S` maior se houver timeouts.

Leia os erros do aluno: `Get "avaliacao/runs/$($c3.id)/itens?so_erros=true"`.

### 4.2 Comparar item a item

```powershell
Get "avaliacao/comparar?a=$($c0.id)&b=$($c3.id)"
Get "avaliacao/comparar?a=$($c2.id)&b=$($c3.id)"      # efeito do filtro
Get "avaliacao/comparar?a=$($c1.id)&b=$($c6.id)"      # RAG sem treino vale quanto?
```

**Verificação:** todos os runs com a mesma `versao_da_medida` (a comparação
diz `mesma_versao_da_medida: true`); `metricas.julgados` igual a
`metricas.respondidos`; `itens_com_artigo_inexistente` e `taxa_de_alucinacao`
lidos para cada condição.

**Não** mude o prompt do juiz entre runs que você pretende comparar. Se
mudar, reavalie todas as condições na versão nova.

---

## Etapa 5 — Calibração humana, on-policy e relatórios

### 5.1 Calibração do juiz (os três autores)

Escolha um run (sugestão: C3 no held-out) e uma amostra de 100 ou mais
itens, os mesmos para todos. Cada autor anota `correto` e `grounded` com a
mesma definição da rubrica, sem ver o veredito do juiz:

```powershell
Post "avaliacao/runs/$($h3.id)/anotacoes" '{"item_id":"ID_DO_ITEM","anotador":"ramon","correto":true,"grounded":true,"comentario":""}'
Get "avaliacao/runs/$($h3.id)/calibracao"
```

**Verificação:** `kappa_correto_maioria` e `kappa_entre_humanos` reportados;
a leitura indica a faixa. Se o kappa do juiz ficar abaixo de 0,6 e o dos
humanos acima, o prompt do juiz precisa de revisão antes das conclusões.

### 5.2 Rodada on-policy (C4)

```powershell
$op = Post on-policy/jobs ('{"dataset_versao":"' + $d_com.versao + '","aluno":"local:qwen3-1.7b-lgpd-VERSAO","amostras_por_prompt":4,"temperatura":0.8,"teto_usd":20}')
Get "on-policy/jobs/$($op.id)"
Get "on-policy/jobs/$($op.id)/amostras?so_reprovadas=true&limite=30"
Invoke-WebRequest -Headers $H -OutFile sft.jsonl "$B/on-policy/jobs/$($op.id)/arquivo/sft"
Invoke-WebRequest -Headers $H -OutFile dpo.jsonl "$B/on-policy/jobs/$($op.id)/arquivo/dpo"
```

No Colab, duas opções (ou as duas, como condições separadas):

```powershell
python -m treino.sft_lora --perfil colab-t4 --dataset sft.jsonl --dataset-versao VERSAO --saida saida-onpolicy
python -m treino.dpo_lora --perfil colab-t4 --pares dpo.jsonl --adapter saida/adapter --dataset-versao VERSAO --saida saida-dpo
```

Servir, registrar e avaliar como na Etapa 3 e 4. Comparar contra C3:
`Get "avaliacao/comparar?a=$($c3.id)&b=$($c4.id)"`.

**Verificação:** `prompts_com_aprovada / prompts` acima de 50% (abaixo
disso, o aluno de C3 ainda está fraco demais para se autodestilar);
`pares_dpo` em número útil (centenas).

### 5.3 Tabelas do TCC

```powershell
$runs = "$($c0.id),$($c1.id),$($c6.id),$($c2.id),$($c3.id),$($c4.id)"
$rotulos = "$($c0.id):C0 professor,$($c1.id):C1 Qwen base,$($c6.id):C6 Qwen+RAG,$($c2.id):C2 SFT sem filtro,$($c3.id):C3 SFT filtrado,$($c4.id):C4 on-policy"
Invoke-RestMethod -Uri "$B/relatorios/condicoes.md?runs=$runs&rotulos=$rotulos" -Headers $H | Out-File tabela_banco_inicial.md -Encoding utf8
Get "relatorios/pareadas?referencia=$($c0.id)&candidatos=$($c1.id),$($c3.id),$($c4.id)"
Get relatorios/custos
```

Repita para os runs do held-out. Para figuras, use o JSON de
`relatorios/condicoes` e `avaliacao/runs/{id}` na ferramenta de gráficos do
grupo.

---

## Reprodutibilidade: o que anotar no TCC

| O quê | Onde está |
|---|---|
| Versão do corpus | cabeçalho de `recursos/fontes/lgpd.txt` (fonte e data) |
| Versão do prompt de síntese | `versao_do_prompt` em cada exemplo e job |
| Versão do filtro | `versao_do_filtro` em cada decisão |
| Versão do dataset | `versao` do manifesto (hash do conteúdo + parâmetros) |
| Receita de treino | `run.json` registrado em `/treinos` (hiperparâmetros, hardware, duração, versões de biblioteca) |
| Régua da avaliação | `versao_da_medida` em cada run |
| Modelos | `GET usage/providers` no momento dos runs |
| Custo | `GET relatorios/custos` ao final |

---

## Cuidados

- **Não reinicie a API com job em curso.** Jobs rodam no processo; ao subir
  de novo, o que estava aberto é marcado `falhou` (o gravado permanece).
- **Orçamento**: todo job aceita `teto_usd`; sem ele vale
  `LLM_BUDGET_USD_POR_JOB`. O job para antes da chamada que estouraria.
- **Held-out é sagrado**: nunca entra em treino nem no on-policy. O split é
  por hash do id, mas não copie linhas manualmente entre arquivos.
- **Juiz ≠ professor**: `GET usage/providers` avisa se coincidirem. Não
  destile o juiz.
- **Dados pessoais**: só marcadores. Se algum exemplo trouxer nome real, o
  filtro reprova por `dado_pessoal_real`; confira a amostra do piloto.
