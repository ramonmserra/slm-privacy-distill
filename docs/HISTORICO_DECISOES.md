# Histórico de decisões

Decisões tomadas por medição ou por restrição, com data. No código fica a
intenção e um ponteiro para cá; aqui fica o porquê e o número.

## 2026-10-18 — Execução híbrida: CPU local valida, Colab T4 roda

**Medido:** i7-11390H (4 núcleos), 15,7 GB RAM, MX450 2 GB VRAM. Estimativa
para LoRA, seq. 1.024, 3.000 exemplos, 2 épocas: Qwen3-0.6B 1–2 dias na CPU
(20–40 min numa T4); Qwen3-1.7B 3–5 dias na CPU (1–2 h numa T4). Geração
on-policy do 0.6B na CPU: 17–33 h para 3.000 prompts × 4 amostras.

**Decisão:** a receita de treino é validada na CPU local com Qwen3-0.6B e
algumas centenas de exemplos; os runs das condições experimentais rodam na
T4 do Colab com o mesmo script. A API só exporta datasets e registra runs.

## 2026-10-18 — Professor OpenAI `gpt-5.6-luna`; juiz Anthropic

Escolha do grupo. Consequências carregadas de `FastAPIProject/app/core/llm.py`
(medidas lá contra a API): o modelo só aceita ferramenta e preserva
`usage_metadata` pela Responses API (`use_responses_api=True`); cobra a
escrita de cache de prompt (1,25× o input), então chamadas sem prefixo
reusável pedem `prompt_cache_options={"mode": "explicit"}`; gasta o teto de
saída em raciocínio, então 8.192 tokens são somados ao `max_tokens`.

Juiz em outra família é regra da pesquisa (viés de autopreferência). Default
`anthropic:claude-sonnet-5`; `avisos_de_configuracao` denuncia coincidência.

## 2026-10-18 — KDFlow é white-box; o caminho black-box é SFT + on-policy

README do KDFlow lido em 2026-10-18: professor servido em SGLang local,
hidden states e `lm_head` recomputados no aluno; nenhuma menção a professor
por API. O modo rotulado "Black-box KD" é SFT. Com a pergunta de pesquisa
fixando "API black-box", a destilação medida é por dados (SFT sobre
respostas e raciocínio do professor) mais autodestilação on-policy por
rejection sampling/DPO com o juiz pontuando. KD white-box com professor
aberto local fica como experimento opcional.

## 2026-10-18 — Aluno principal Qwen3-1.7B; Manacá-1B como segundo aluno

Manacá-1B (`menezesbruno/manaca-1b-base`, 1,72B) só existe em versão base e
exige o tokenizer corrigido pelos autores; o SFT teria de ensinar o formato
de chat do zero. Qwen3-1.7B já segue instrução e produz JSON. O provedor
`local` é genérico, então trocar o aluno é trocar `LOCAL_MODEL_STUDENT`.

## 2026-10-18 — Persistência em arquivo na Etapa 0

Metadados e telemetria em JSON com escrita atômica (`_file_store`), uma
instância do processo. SQLite entra só se uma consulta (comparação pareada
entre dezenas de runs) ficar lenta ou ambígua em arquivo. Postgres
obrigatório do projeto de referência não entra: não há tenant.

## 2026-10-18 — Orçamento por job estima só a ENTRADA da chamada

O portão compara custo real acumulado do job + estimativa de entrada
(chars/4 × preço de input) contra o teto. A saída não é estimada: assumir
`max_tokens` de saída por chamada (até 8.192 no professor) recusaria jobs
legítimos cedo demais. O teto pode ser ultrapassado pela saída da última
chamada, na ordem de centavos — erro para o lado barato.

## 2026-10-18 — Fonte da LGPD: Planalto consolidado, com correção de "Art. 5 7."

A página oficial traz "Art. 5 7. (VETADO)." (espaço dentro do número). O
parser lia "artigo 5" e o id `LGPD:art.5` ficava duplicado, sobrescrevendo o
art. 5 verdadeiro no dicionário por id. Duas defesas: `scripts/baixar_lgpd.py`
normaliza o número partido ao limpar, e `Corpus.__init__` recusa ids
duplicados (`CorpusInvalido`) em vez de sobrescrever em silêncio. Resultado
após a correção: 79 artigos, 452 dispositivos, sem duplicatas; todas as
referências do banco inicial resolvem.

## 2026-10-18 — Dedup por Jaccard de tokens (limiar 0,7)

Medida barata e determinística, aplicada contra o próprio job e contra o
banco inicial de avaliação (vazamento). Observado nos testes: perguntas
templadas que diferem só pelo número do artigo são tratadas como
duplicatas, o que é o comportamento desejado. Paráfrase profunda escapa;
se a medição mostrar que escapa demais, entra similaridade por embedding na
Etapa 2.

## 2026-10-18 — Self-consistency: o professor reamostra, o juiz decide a equivalência

A "incerteza do professor" é medida pedindo ao próprio professor k novas
respostas à mesma pergunta, com o MESMO contexto da síntese e temperatura
0,8, e perguntando ao juiz (outra família) se cada alternativa chega à mesma
conclusão com as mesmas bases. `score_incerteza` = 1 − fração equivalente;
reprova acima de 0,34 (com k=3, exige 2 de 3). Alternativa descartada: pedir
ao professor que avalie a própria consistência, que traria o mesmo viés que
o juiz independente existe para evitar. Os prompts e campos da rubrica
entram no hash `VERSAO_DO_FILTRO`, gravado em cada decisão.

## 2026-10-18 — Held-out por hash do id, não por sorteio

`split_de(id, fração)` usa SHA-256 do id do exemplo. O mesmo exemplo cai no
mesmo lado em qualquer versão de dataset, então o held-out da v1 não vira
treino na v2 sem ninguém precisar lembrar de excluir. Custo: a fração é
aproximada por grupo (tipo, artigo); o manifesto reporta as contagens reais.
Medido em teste com 2.000 ids: 15% pedidos, entre 12% e 18% obtidos.

## 2026-10-18 — Prompt do aluno fixo e versionado

`PROMPT_DO_ALUNO` é uma constante curta, com hash gravado no manifesto do
dataset; a avaliação usará a mesma constante. O comportamento (citar ids,
abster-se) vem dos exemplos, não da instrução — aluno pequeno usa mal
instrução longa. `incluir_raciocinio=False` gera a condição de controle
"só conclusão".

## 2026-10-18 — Conjunto de múltipla escolha adiado

O plano previa um conjunto MCQ de questões públicas de concursos sobre a
LGPD. Não há fonte no repositório e não se fabrica gabarito: fica como
insumo a ser reunido pelo grupo, no mesmo formato JSONL do banco inicial.

## 2026-10-18 — Receita de treino: TRL + PEFT, não KDFlow

KDFlow é white-box (professor local em SGLang) e seu modo black-box é SFT.
A receita usa TRL `SFTTrainer` + LoRA, que roda igual na T4 do Colab e na
CPU local e é o que o Colab instala sem compilar nada. Formato
prompt/completion conversacional (perda só na completion) em vez de
`assistant_only_loss`, que exige marcadores `{% generation %}` no template
de chat e o do Qwen3 não os tem. Pisos de versão (`trl>=0.19`), não pins: o
Colab atualiza o torch por conta própria.

## 2026-10-18 — Prefixo `<think>` vazio em treino e inferência

Qwen3 em modo "sem pensar" abre o turno do assistente com um bloco `<think>`
vazio. O treino antepõe esse prefixo à resposta e o `Modelfile` o gera na
inferência: treino e uso veem a mesma sequência de tokens. Para outro
modelo-base, `--prefixo-do-assistente ""` e outro template.

## 2026-10-18 — `run.json` é o corpo de `POST /treinos`

O script grava, com os mesmos nomes de campo do schema `RegistroDeRun`, o que
foi treinado (dataset, base, perfil, hiperparâmetros, hardware, duração,
métricas, versões de biblioteca) e o `modelo_local_id` sugerido
(`local:<base>-lgpd-<8 chars da versão>`). A API exige dataset conhecido e
recusa reusar um id já ocupado por run concluído: cada versão treinada é um
id novo, e rollback é trocar a string.

## 2026-10-18 — Treino em Colab, não local, para os runs reais

Decisão do grupo. O perfil `cpu-local` existe para provar a receita com
poucas dezenas de exemplos antes de gastar a sessão do Colab; não substitui
o run completo (1 a 2 dias na CPU para 3.000 exemplos no 0.6B).

## 2026-10-18 — Avaliação: o modelo medido não entra na versão da medida

`versao_da_medida` = hash(prompt do juiz + prompt do aluno + regras +
conjunto + RAG). O modelo medido fica de fora de propósito: é o que permite
pôr professor e aluno na mesma régua e comparar item a item. Mudou o juiz,
o conjunto ou o RAG, a comparação vira "indicativa" e a API diz isso.

## 2026-10-18 — Acerto e alucinação como eixos independentes

`acerto` = `correto` segundo o juiz. `alucinacao` = `grounded=false` pelo
juiz OU artigo inexistente mencionado na prosa (verificação por código,
que enxerga o que o juiz pode não ver). Uma resposta pode acertar sem
lastro (adivinhou) e esse é o achado que mais importa para a hipótese de
baixa alucinação. Herdado do Sandbox do projeto de referência.

## 2026-10-18 — Comparação pareada: McNemar exato + bootstrap do delta

Dois agregados não se comparam; o item a item sim. McNemar exato (binomial
bilateral) sobre as discordâncias, bootstrap percentil do delta médio
(2.000 reamostras, seed fixa) e uma frase de leitura. Com 34 itens do banco
inicial a potência é baixa; o held-out sintético é o que dá volume.

## 2026-10-18 — Conjunto MCQ segue adiado; calibração humana tem a função, não a tela

`kappa_de_cohen` está em `core/stats.py`. O endpoint para o grupo anotar a
amostra de calibração (≥ 100 itens) fica para a Etapa 5, junto dos
relatórios, para a anotação cair sobre runs reais.

## 2026-10-18 — On-policy com professor black-box = seleção pelo juiz

Sem logits do professor, a autodestilação on-policy é: o aluno amostra N
respostas por prompt de treino com temperatura 0,8; o juiz (o mesmo da
avaliação, cego) compara cada uma à resposta do professor com o rastro dos
artigos citados; aprovada = correta, sem artigo inexistente e com lastro.
SFT usa a aprovada de maior similaridade (rejection sampling); DPO pareia a
melhor aprovada com a pior reprovada do mesmo prompt. Prompts vêm só do
split de treino, por construção. Fallback ao professor é opcional e fica
marcado na linha (`origem`), para a análise separar o que é do aluno.

## 2026-10-18 — Um só juiz para avaliação e on-policy

`JuizLlm` é compartilhado: mesma rubrica, mesmo prompt, mesma versão. Um
juiz diferente na seleção on-policy faria o aluno aprender uma régua e ser
medido por outra.

## 2026-10-18 — Calibração: maioria com ≥2 anotadores, empate descarta

O kappa principal é juiz × voto da maioria dos humanos; com dois anotadores
e empate o item sai do cálculo. Três anotadores (os três autores) evitam o
empate. Kappa entre humanos é reportado como teto do que se pode esperar do
juiz.

## 2026-10-18 — Relatórios em Markdown, figuras fora da API

A tabela das condições sai em JSON e em Markdown com IC de 95%, pronta para o
texto. Gráficos não entram na API: dependem do estilo do TCC e saem do JSON
com a ferramenta que o grupo preferir.

## 2026-10-18 — Front mínimo só para a parte humana do fluxo

HTML + JS puro servido pela própria API em `/ui`, sem framework nem build,
chave em `sessionStorage`. Três telas: anotação de calibração (veredito do
juiz escondido por padrão, para não contaminar o humano), inspeção de
exemplos/decisões/amostras, e painel de runs e relatórios. Disparar síntese,
filtro e treino fica fora: operações caras e raras, já cobertas pelo Swagger
e pelo guia de execução; um botão para elas convidaria ao gasto acidental.
Sintaxe do JS validada por `node --check` na suíte (pula sem Node).
