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
