# Resultados da execução (passos 10 a 15)

Atualizado em 2026-11-23. Resume o que foi feito e os números finais.
A última seção diz onde está a evidência de cada item.

## 1. Configuração

| Item | Valor |
|---|---|
| Professor (C0) | `openai:gpt-6-luna` |
| Juiz | `anthropic:claude-sonnet-5-5` |
| Aluno | Qwen3-1.7B (LoRA r=16, 2 épocas, T4 do Colab), servido em GGUF Q4_K_M no Ollama, em CPU |
| Aluno alternativo (C5) | Manacá-1B base, mesma receita |
| Dataset filtrado | `5b1301bbf406`: 2.592 de treino, 454 de held-out |
| Dataset sem filtro | `d6c7d57c179e`: 3.150 de treino, 548 de held-out |
| Conjunto de avaliação | 150 primeiros itens do held-out filtrado |
| Régua do juiz | `versao_da_medida` `92eaa633926b`: o juiz recebe os artigos de referência do item e os artigos citados na resposta |

## 2. Condições avaliadas

| Condição | O que é |
|---|---|
| C0 | professor |
| C1 | Qwen base, sem treino |
| C6 | Qwen base com RAG |
| C2 | Qwen com SFT no dataset sem filtro |
| C3 | Qwen com SFT no dataset filtrado |
| C4 | C3 com autodestilação on-policy (DPO) |
| C5 | Manacá com SFT no dataset filtrado |

## 3. Resultado principal (held-out)

| Condição | Julgados | Acurácia [IC95] | Sem lastro | Artigo inexistente (itens) | Abstenção adequada |
|---|---:|---|---:|---:|---|
| C0 professor | 146 | 86,3% [80–91] | 41,8% | 0 | 10/20 |
| C4 on-policy (DPO) | 142 | 52,1% [44–60] | 81,7% | 0 | 15/18 |
| C3 SFT filtrado | 149 | 49,7% [42–58] | 77,2% | 1 | 15/21 |
| C2 SFT sem filtro | 144 | 48,6% [41–57] | 81,2% | 2 | 13/20 |
| C6 Qwen + RAG | 149 | 36,9% [30–45] | 94,0% | 0 | 6/21 |
| C5 Manacá SFT | 150 | 26,7% [20–34] | 86,7% | 0 | 13/21 |
| C1 Qwen base | 150 | 16,0% [11–23] | 100% | 5 | 1/21 |

- Acurácia: veredito `correto` do juiz. Avalia a conclusão e a substância do fundamento; número de artigo ou inciso errado conta só no lastro.
- Sem lastro: resposta com afirmação não sustentada pelos artigos, ou com artigo inexistente.
- Respostas corretas e com lastro: C0 55,5%; C3 18,8%; C2 17,4%; C4 16,9%; C6 4,7%; C1 0%.

## 4. Comparações pareadas (McNemar e bootstrap)

| Par | Diferença | IC95 | p | Leitura |
|---|---|---|---|---|
| C1 → C3 | +33,6 p.p. | +24,8 a +42,3 | < 0,001 | a destilação funciona |
| C0 → C3 | −36,6 p.p. | −46,9 a −26,9 | < 0,001 | o aluno fica longe do professor |
| C2 → C3 | 0,0 p.p. | −7,0 a +7,0 | 1,00 | o filtro não mudou a acurácia |
| C6 → C3 | +12,8 p.p. | +3,4 a +22,3 | 0,016 | SFT supera RAG (indicativa: medidas diferentes) |
| C1 → C6 | +20,8 p.p. | +13,4 a +28,2 | < 0,001 | RAG ajuda o modelo base (indicativa) |
| C3 → C4 | +5,0 p.p. | −2,1 a +12,8 | 0,248 | autodestilação sem diferença significativa |
| C3 → C5 | −22,8 p.p. | −31,5 a −14,1 | < 0,001 | Manacá abaixo do Qwen |

## 5. Autodestilação on-policy (C4)

| Etapa | Resultado |
|---|---|
| Ensaio (20 prompts) | 25% dos prompts com aprovada com lastro; 65% pela conclusão |
| Job (400 prompts × 4 amostras) | 1.600 amostras, 1.445 julgadas, 539 aprovadas |
| Prompts com aprovada | 233 de 400 (58,3%); com lastro, 83 (20,8%) |
| Pares DPO | 178 |
| Treino | DPO de 1 época sobre o LoRA do C3, 7,2 min na T4 |
| Avaliação | 52,1%, contra 49,7% do C3 (seção 4) |

Decisões de desenho:

- Critério de aprovação: conclusão correta e nenhum artigo inexistente. O critério mais rígido, que exige também lastro em todas as citações, aprovaria respostas em só 20,8% das perguntas, o que não gera pares suficientes para o DPO. Para controlar o risco de reforçar citações erradas, a avaliação do C4 reporta o lastro ao lado do C3 (seção 3).
- O DPO foi aplicado sobre um C3 treinado de novo na mesma sessão, com a mesma receita e semente do C3 avaliado.
- 32 prompts do job ficaram sem veredito do juiz.
- A geração das amostras foi em GPU do Colab; a avaliação, em CPU local.

## 6. Calibração do juiz pelo especialista

| Medida | Valor |
|---|---|
| Itens anotados | 150 do C3 (149 com veredito do juiz) |
| Concordância | 91,95% (137 de 149) |
| Kappa de Cohen (`correto`) | 0,84 |
| Juiz reprovou, especialista aprovou | 7 |
| Juiz aprovou, especialista reprovou | 5 |

Limitação: um anotador só; sem kappa entre humanos.

Achados da comparação com o juiz:

- Quatro itens com gabarito errado ou incompleto (18, 19, 74, 133).
- Dois itens com veredito certo e justificativa do juiz errada (10, 107).

## 7. Custos (estimativa por tabela pública de preços)

| Etapa | Chamadas | US$ |
|---|---:|---:|
| Síntese | 960 | 5,15 |
| Filtro | 2.960 | 42,11 |
| Avaliação | 1.800 | 29,70 |
| On-policy | 2.124 | 16,06 |
| Diagnóstico | 3 | 0,00 |
| **Total** | **7.847** | **93,01** |

A avaliação inclui rejulgamentos e runs substituídos, não só os sete runs da seção 3. O treino usou a T4 gratuita do Colab.

## 8. Mudança feita durante a execução

| Mudança | Motivo | Efeito |
|---|---|---|
| Entrada em minúsculas no Manacá | o tokenizador do Manacá treina em minúsculas e o Ollama não aplica a normalização | C5 foi de 17,6% para 26,7% (+8,4 p.p., p = 0,043) |

Todas as condições foram julgadas com a mesma régua (seção 1).

## 9. Onde consultar as evidências

A pasta `data/` fica fora do git. Há uma cópia em `Resultados\data_evidencias_2026-11-23.zip`.
O front abre em `http://localhost:8000/ui/` com a API no ar (`uvicorn main:app`); ele tem três telas: Anotação, Inspeção e Painel.

| O quê | Arquivo | No front |
|---|---|---|
| Registro completo da execução, com ids | `data/ESTADO_DA_EXECUCAO.md` | — |
| Tabela da seção 3 | `data/relatorios_tcc/tabela_held_out.md` e `.json` | Painel → Runs de avaliação (marcar os runs) → Tabela das condições → Gerar tabela |
| Comparações da seção 4 | `data/relatorios_tcc/pareadas_vs_C0.json`, `pareadas_vs_C3.json`; `data/avaliacao/comparacoes_regua3/` | Painel → Comparação pareada (escolher A e B) → Comparar |
| Custos | `data/relatorios_tcc/custos.json`; bruto em `data/llm_usage.json` | Painel → Custo por etapa |
| Métricas de um run | `data/avaliacao/runs/<id>.json` | Painel → Runs de avaliação |
| Respostas e vereditos de um run | `data/avaliacao/<id>/itens.jsonl` | Inspeção → Fonte "Avaliação — todos os itens" (ou "itens com erro") → escolher o run |
| Exemplos gerados pelo professor | `data/sintese/` | Inspeção → Fonte "Síntese — exemplos" |
| Decisões do filtro | `data/filtro/<job>/decisoes.jsonl` | Inspeção → Fonte "Filtro — aprovados" ou "Filtro — reprovados" |
| Datasets de treino e held-out | `data/datasets/<versao>/` | Painel → Datasets |
| Ensaio on-policy | `data/on_policy/3ac4b07aea41/` | Inspeção → Fonte "On-policy — amostras" → job `3ac4b07aea41` |
| Job on-policy da C4 (amostras, pares, SFT) | `data/on_policy/d0b50b53f8ca/` | Inspeção → Fonte "On-policy — amostras" ou "On-policy — reprovadas" → job `d0b50b53f8ca`; contadores em Painel → Jobs |
| Modelo e receita da C4 | `Resultados\C4_QWEN` (`run.json`, `resumo.txt`, GGUF, `adapter.zip`) | Painel → Jobs (run de treino `6fe7506668fd`) |
| Modelos C3, C2 e C5 | `Resultados\Filtrado_QWEN`, `Sem_Filtrado_QWEN`, `Filtrado_Manaca` | Painel → Jobs (runs de treino) |
| Anotação do especialista | `data/avaliacao/8f4addce7739/especialista_marcelo_itens_vereditos_revisados.jsonl` | Anotação → Baixar anotações (.json) |
| Kappa da calibração | `data/avaliacao/8f4addce7739/anotacoes.jsonl` | Anotação → run `8f4addce7739` → Recalcular |
| Decisões e justificativas | `docs/HISTORICO_DECISOES.md` | — |
| Procedimento de cada passo | `docs/GUIA_DE_EXECUCAO.md` | — |

Ids dos runs de avaliação:

| Condição | Run |
|---|---|
| C0 | `1cdbf8d20ecc` |
| C1 | `4039b8d6d9c5` |
| C6 | `d0160d84d9c8` |
| C2 | `5ea61ff3c4c1` |
| C3 | `8f4addce7739` |
| C4 | `b28c6e04cd12` |
| C5 | `94b3cfb86c0e` |

Ids dos runs de treino: C3 `249deebb142f`, C2 `9b1e21fc9fdb`, C5 `f35c9755678d`, C4 `6fe7506668fd`.
