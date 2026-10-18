# Treino do aluno (fora da API)

A API exporta o dataset e registra o run; o treino acontece aqui, com o
**mesmo script** em dois ambientes:

| Perfil | Onde | Modelo-base | Para que |
|---|---|---|---|
| `cpu-local` | este notebook (i7, 16 GB, sem GPU útil) | Qwen3-0.6B, fp32, seq 512, 1 época | **Validar a receita** ponta a ponta com algumas dezenas de exemplos. Não treina para valer: 3.000 exemplos levariam 1 a 2 dias |
| `colab-t4` | Google Colab, T4 16 GB | Qwen3-1.7B, fp16, LoRA r=16, seq 1024, 2 épocas | **Os runs das condições experimentais** (1 a 2 h cada) |
| `colab-t4-4bit` | Colab T4 | Qwen3-4B em 4-bit | Aluno maior, se houver tempo; sem merge automático |

O treino é SFT com LoRA sobre o JSONL de chat (`system` = prompt do aluno,
`user` = pergunta, `assistant` = raciocínio + resposta do professor), com
perda só nos tokens do assistente. É a destilação black-box da pesquisa.

## Arquivos

| Arquivo | Função |
|---|---|
| `preparacao.py` | Perfis, leitura e validação do JSONL, montagem das mensagens, `run.json`. Sem torch; testado em `tests/test_treino.py` |
| `sft_lora.py` | O treino (TRL `SFTTrainer` + PEFT). `--dry-run` valida sem treinar |
| `exportar_gguf.sh` | merge → GGUF f16 → Q4_K_M (llama.cpp), gera o `Modelfile` |
| `Modelfile.template` | Template ChatML do Qwen3 para o Ollama, sem `SYSTEM` (a API envia o prompt do aluno) |
| `colab_treino_lgpd.ipynb` | Notebook com o fluxo completo no Colab |
| `requirements-treino.txt` | Dependências do treino (pisos, não pins: o Colab atualiza o torch) |

## Validação local na CPU (recomendado antes do Colab)

```powershell
python -m venv .venv-treino
.venv-treino\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r treino/requirements-treino.txt

# baixe treino.jsonl e held_out.jsonl da API (GET /api/v1/datasets/{versao}/arquivo/{split})
python -m treino.sft_lora --perfil cpu-local --dataset treino.jsonl --held-out held_out.jsonl --dataset-versao VERSAO --saida saida-cpu --dry-run
python -m treino.sft_lora --perfil cpu-local --dataset treino.jsonl --held-out held_out.jsonl --dataset-versao VERSAO --saida saida-cpu --max-exemplos 40 --max-eval 10
```

Esperado: `saida-cpu/run.json` com `status: concluido`, `metricas.train_loss`
e `eval_loss`, pastas `adapter/` e `merged/`. Com 40 exemplos e seq 512, de
15 a 40 minutos neste hardware. Se isso roda, a receita está certa; o Colab
só muda o perfil.

## Colab

Abra `treino/colab_treino_lgpd.ipynb` no Colab com GPU T4 e siga as células:
clone do repositório, upload de `treino.jsonl`/`held_out.jsonl`, dry-run,
treino, GGUF, download de `modelo-q4_k_m.gguf`, `Modelfile` e `run.json`.

## Do GGUF ao Ollama e à API

Na máquina do Ollama, na pasta com `modelo-q4_k_m.gguf` e `Modelfile`:

```powershell
ollama create qwen3-1.7b-lgpd-ab12cd34 -f Modelfile      # nome = modelo_local_id sem "local:"
ollama run qwen3-1.7b-lgpd-ab12cd34 "O que e dado pessoal sensivel?"
```

Com Docker Compose, copie os arquivos para dentro do contêiner
(`docker cp`) ou monte a pasta como volume antes do `ollama create`.

Registre o run e confirme que o Ollama o serve:

```powershell
curl -X POST -H "X-API-Key: SUA_CHAVE" -H "Content-Type: application/json" --data-binary "@run.json" http://localhost:8000/api/v1/treinos
curl -X POST -H "X-API-Key: SUA_CHAVE" http://localhost:8000/api/v1/treinos/ID_DO_RUN/verificar
```

`verificar` consulta `LOCAL_LLM_BASE_URL/models` e marca
`disponivel_localmente`. A partir daí o modelo é um `local:<id>` como
qualquer outro para a avaliação (Etapa 4).

## Rodada on-policy (SFT ou DPO)

Depois de servir o aluno no Ollama e rodar `POST /api/v1/on-policy/jobs`,
baixe os arquivos do job:

```powershell
# SFT on-policy: mesmo script, mesmo formato de chat
python -m treino.sft_lora --perfil colab-t4 --dataset sft.jsonl --dataset-versao VERSAO --saida saida-onpolicy

# DPO: continua o LoRA do SFT anterior
python -m treino.dpo_lora --perfil colab-t4 --pares dpo.jsonl --adapter saida/adapter --dataset-versao VERSAO --saida saida-dpo
```

O `dpo_lora.py` aplica o mesmo prefixo `<think>` vazio aos dois lados do
par, usa `beta=0.1` e taxa de aprendizado baixa (5e-6), e grava `run.json`
com `receita: dpo_lora` e `modelo_local_id` com o sufixo `-dpo-`. O `--dry-run`
valida os pares sem treinar.

## Decisões de receita

- **Prefixo `<think>\n\n</think>\n\n` na resposta.** O Qwen3 em modo "sem
  pensar" abre o turno do assistente com um bloco `<think>` vazio. O treino
  antepõe o mesmo prefixo (`--prefixo-do-assistente`) e o `Modelfile` o
  gera na inferência, para treino e uso verem a mesma sequência. Para outro
  modelo-base, passe `--prefixo-do-assistente ""` e ajuste o template.
- **Perda só no assistente** (`assistant_only_loss`): o aluno não deve
  aprender a reproduzir perguntas nem o system prompt.
- **fp16 na T4**: a T4 não tem bf16. `--perfil colab-t4` já cuida disso.
- **Held-out só no eval loss.** O script aborta se um id aparecer nos dois
  arquivos; a avaliação de verdade é a da Etapa 4, com o juiz.
- **`run.json` é o corpo de `POST /treinos`**, sem tradução manual: o que
  foi treinado é o que fica registrado (hiperparâmetros, hardware, duração,
  métricas, versões das bibliotecas).
