#!/usr/bin/env bash
# Converte o modelo FUNDIDO (saida/merged) em GGUF e quantiza para Q4_K_M,
# o formato que o Ollama carrega em CPU. Pensado para o Colab (Linux).
#
#   bash treino/exportar_gguf.sh saida/merged saida/gguf
#
# Resultado: saida/gguf/modelo-f16.gguf (grande) e modelo-q4_k_m.gguf (o que
# se baixa: ~1,1 GB para 1.7B). Alternativa sem compilar o quantizador:
# baixar o f16 e deixar o Ollama quantizar ao criar:
#   ollama create NOME -f Modelfile --quantize q4_K_M   (FROM ./modelo-f16.gguf)
set -euo pipefail

ORIGEM="${1:?uso: exportar_gguf.sh <pasta merged> <pasta de saida>}"
DESTINO="${2:?uso: exportar_gguf.sh <pasta merged> <pasta de saida>}"
mkdir -p "$DESTINO"

if [ ! -d llama.cpp ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp
fi
pip install -q -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

python llama.cpp/convert_hf_to_gguf.py "$ORIGEM" --outfile "$DESTINO/modelo-f16.gguf" --outtype f16

if [ ! -x llama.cpp/build/bin/llama-quantize ]; then
  cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF > /dev/null
  cmake --build llama.cpp/build --target llama-quantize -j"$(nproc)" > /dev/null
fi
llama.cpp/build/bin/llama-quantize "$DESTINO/modelo-f16.gguf" "$DESTINO/modelo-q4_k_m.gguf" Q4_K_M

cp "$(dirname "$0")/Modelfile.template" "$DESTINO/Modelfile"
ls -lh "$DESTINO"
echo "Baixe modelo-q4_k_m.gguf + Modelfile e, na maquina do Ollama: ollama create <nome> -f Modelfile"
