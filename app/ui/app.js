/* Front minimo da pesquisa: JS puro, sem build. Tres telas sobre os endpoints
   existentes da API. A chave fica em sessionStorage (morre com a aba). */
(() => {
  "use strict";
  const API = "/api/v1";
  const $ = (id) => document.getElementById(id);
  const estado = { anot: { itens: [], pos: 0, run: null, feitos: new Set() }, insp: { offset: 0 }, runs: [], cmp: null, tabela: null };
  const PAGINA_DE_DOWNLOAD = 500; // limite maximo das rotas paginadas da API

  // ------------------------------------------------------------ utilidades
  function chave() { return sessionStorage.getItem("gena_api_key") || ""; }
  function toast(msg, erro = false) {
    const t = $("toast"); t.textContent = msg; t.className = "toast" + (erro ? " erro" : "");
    clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.add("oculto"), erro ? 6000 : 2500);
  }
  async function api(rota, opcoes = {}) {
    const resp = await fetch(API + rota, { ...opcoes, headers: { "X-API-Key": chave(), "Content-Type": "application/json", ...(opcoes.headers || {}) } });
    const texto = await resp.text();
    if (!resp.ok) {
      let detalhe = texto; try { detalhe = JSON.parse(texto).detail || texto; } catch (_) { /* texto cru */ }
      throw new Error(`${resp.status}: ${typeof detalhe === "string" ? detalhe : JSON.stringify(detalhe)}`);
    }
    try { return JSON.parse(texto); } catch (_) { return texto; }
  }
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (v, ic) => v == null ? "-" : `${(v * 100).toFixed(1)}%` + (ic ? ` [${Math.round(ic[0] * 100)}–${Math.round(ic[1] * 100)}]` : "");
  const opcoes = (select, lista, rotulo, valor, vazio = "—") => {
    select.innerHTML = lista.length ? lista.map((x) => `<option value="${esc(valor(x))}">${esc(rotulo(x))}</option>`).join("") : `<option value="">${vazio}</option>`;
  };
  const classeStatus = (s) => s === "concluido" ? "ok" : (s === "falhou" || s === "orcamento_excedido") ? "erro" : "atencao";

  // ------------------------------------------------------------ download
  // Tudo que a API devolve pode ser salvo pelo navegador. Nao ha <a href>
  // direto porque a chave vai no header, nao na URL.
  function salvarArquivo(nome, conteudo, tipo = "application/json") {
    const blob = conteudo instanceof Blob ? conteudo : new Blob([conteudo], { type: tipo });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = nome; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  const salvarJson = (nome, dados) => salvarArquivo(nome, JSON.stringify(dados, null, 2));
  async function baixarRota(rota, nome) {
    const resp = await fetch(API + rota, { headers: { "X-API-Key": chave() } });
    if (!resp.ok) throw new Error(`${resp.status}: ${(await resp.text()).slice(0, 200)}`);
    salvarArquivo(nome, await resp.blob());
  }
  // Percorre uma rota paginada (limite/offset) ate' a ultima pagina.
  async function todasAsPaginas(rota) {
    const tudo = []; let offset = 0;
    for (;;) {
      const pagina = await api(rota(`limite=${PAGINA_DE_DOWNLOAD}&offset=${offset}`));
      tudo.push(...pagina);
      if (pagina.length < PAGINA_DE_DOWNLOAD) return tudo;
      offset += PAGINA_DE_DOWNLOAD;
    }
  }
  async function comBotao(botao, acao) {
    botao.disabled = true;
    try { await acao(); toast("Download pronto."); } catch (e) { toast(e.message, true); } finally { botao.disabled = false; }
  }

  // ------------------------------------------------------------ chave e abas
  $("api-key").value = chave();
  function mostrarChave() { $("estado-chave").textContent = chave() ? "chave guardada" : "sem chave"; }
  $("salvar-chave").onclick = async () => {
    sessionStorage.setItem("gena_api_key", $("api-key").value.trim()); mostrarChave();
    try { await api("/usage/providers"); toast("Autenticado."); carregarTudo(); } catch (e) { toast(e.message, true); }
  };
  document.querySelectorAll(".abas button").forEach((b) => b.onclick = () => {
    document.querySelectorAll(".abas button").forEach((x) => x.classList.toggle("ativa", x === b));
    document.querySelectorAll(".aba").forEach((s) => s.classList.toggle("ativa", s.id === "aba-" + b.dataset.aba));
    if (b.dataset.aba === "painel") painel();
  });

  // ------------------------------------------------------------ dados comuns
  async function carregarRuns() {
    estado.runs = await api("/avaliacao/runs");
    const rot = (r) => `${r.id} · ${r.modelo} · ${r.parametros.conjunto} · ${r.status}`;
    opcoes($("an-run"), estado.runs, rot, (r) => r.id, "nenhum run de avaliação");
    opcoes($("pa-cmp-a"), estado.runs, rot, (r) => r.id);
    opcoes($("pa-cmp-b"), estado.runs, rot, (r) => r.id);
  }
  async function carregarTudo() { try { await carregarRuns(); await listarJobsDaInspecao(); } catch (e) { toast(e.message, true); } }

  // ------------------------------------------------------------ ANOTAÇÃO
  $("an-carregar").onclick = async () => {
    const run = $("an-run").value, anotador = $("an-anotador").value.trim();
    if (!run || !anotador) return toast("Escolha o run e informe seu nome.", true);
    try {
      const itens = await api(`/avaliacao/runs/${run}/itens?limite=500`);
      const feitas = await api(`/avaliacao/runs/${run}/anotacoes?anotador=${encodeURIComponent(anotador)}`);
      estado.anot = { run, itens: itens.filter((r) => r.veredito), pos: 0, feitos: new Set(feitas.map((a) => a.item_id)) };
      if ($("an-pular").checked) estado.anot.itens = estado.anot.itens.filter((r) => !estado.anot.feitos.has(r.item.id));
      mostrarItem();
    } catch (e) { toast(e.message, true); }
  };
  $("an-ver-juiz").onchange = () => $("an-juiz-box").classList.toggle("oculto", !$("an-ver-juiz").checked);

  async function mostrarItem() {
    const { itens, pos } = estado.anot;
    $("an-progresso").textContent = itens.length ? `${pos + 1} / ${itens.length}` : "nada a anotar";
    if (pos >= itens.length) { $("an-item").classList.add("oculto"); $("an-vazio").classList.remove("oculto"); $("an-vazio").textContent = itens.length ? "Todos os itens deste run foram anotados por você. Obrigado." : "Nenhum item julgado neste run."; return; }
    const r = itens[pos];
    $("an-vazio").classList.add("oculto"); $("an-item").classList.remove("oculto");
    $("an-item-id").textContent = r.item.id; $("an-item-tipo").textContent = `tipo: ${r.item.tipo}`;
    $("an-item-abstencao").textContent = r.item.abstencao_esperada ? "abstenção esperada" : "";
    $("an-pergunta").textContent = r.item.pergunta; $("an-resposta").textContent = r.resposta.texto || "(sem resposta)";
    $("an-referencia").textContent = r.item.referencia; $("an-artigos").textContent = r.item.artigos_referencia.join(", ") || "nenhum";
    $("an-juiz").textContent = JSON.stringify(r.veredito, null, 2);
    document.querySelectorAll('input[name="an-correto"]').forEach((i) => i.checked = false);
    document.querySelector('input[name="an-grounded"][value=""]').checked = true;
    $("an-comentario").value = "";
    $("an-rastro").innerHTML = "<span class='mini'>carregando…</span>";
    try {
      const trechos = await Promise.all(r.item.artigos_referencia.map((a) => api(`/corpus/dispositivos/${encodeURIComponent(a)}`).then((d) => d.artigo_completo).catch(() => `(${a} não encontrado no corpus)`)));
      $("an-rastro").innerHTML = trechos.length ? trechos.map((t) => `<pre>${esc(t)}</pre>`).join("") : "<span class='mini'>sem artigos de referência</span>";
    } catch (_) { $("an-rastro").textContent = "corpus indisponível"; }
  }
  $("an-pular-item").onclick = () => { estado.anot.pos++; mostrarItem(); };
  $("an-salvar").onclick = async () => {
    const correto = document.querySelector('input[name="an-correto"]:checked');
    if (!correto) return toast("Marque se a resposta está correta.", true);
    const g = document.querySelector('input[name="an-grounded"]:checked').value;
    const r = estado.anot.itens[estado.anot.pos];
    const corpo = { item_id: r.item.id, anotador: $("an-anotador").value.trim(), correto: correto.value === "true", grounded: g === "" ? null : g === "true", comentario: $("an-comentario").value.trim() };
    try {
      await api(`/avaliacao/runs/${estado.anot.run}/anotacoes`, { method: "POST", body: JSON.stringify(corpo) });
      estado.anot.pos++; toast("Anotação salva."); mostrarItem();
    } catch (e) { toast(e.message, true); }
  };
  $("an-calibrar").onclick = async () => {
    const run = $("an-run").value; if (!run) return;
    try { $("an-calibracao").textContent = JSON.stringify(await api(`/avaliacao/runs/${run}/calibracao`), null, 2); } catch (e) { toast(e.message, true); }
  };
  $("an-baixar").onclick = () => {
    const run = $("an-run").value; if (!run) return toast("Escolha um run.", true);
    comBotao($("an-baixar"), async () => {
      const [calibracao, anotacoes] = await Promise.all([api(`/avaliacao/runs/${run}/calibracao`), api(`/avaliacao/runs/${run}/anotacoes`)]);
      salvarJson(`anotacoes-${run}.json`, { calibracao, anotacoes });
    });
  };

  // ------------------------------------------------------------ INSPEÇÃO
  const FONTES = {
    "sintese":              { jobs: "/sintese/jobs",   rota: (j, q) => `/sintese/jobs/${j}/exemplos?${q}`, render: cartaoExemplo },
    "sintese-descartados":  { jobs: "/sintese/jobs",   rota: (j) => `/sintese/jobs/${j}/descartados`, render: cartaoDescartado, semPaginacao: true },
    "filtro-aprovados":     { jobs: "/filtro/jobs",    rota: (j, q) => `/filtro/jobs/${j}/decisoes?aprovado=true&${q}`, render: cartaoDecisao },
    "filtro-reprovados":    { jobs: "/filtro/jobs",    rota: (j, q) => `/filtro/jobs/${j}/decisoes?aprovado=false&${q}`, render: cartaoDecisao },
    "on-policy":            { jobs: "/on-policy/jobs", rota: (j, q) => `/on-policy/jobs/${j}/amostras?${q}`, render: cartaoAmostra },
    "on-policy-reprovadas": { jobs: "/on-policy/jobs", rota: (j, q) => `/on-policy/jobs/${j}/amostras?so_reprovadas=true&${q}`, render: cartaoAmostra },
    "avaliacao-erros":      { jobs: "/avaliacao/runs", rota: (j, q) => `/avaliacao/runs/${j}/itens?so_erros=true&${q}`, render: cartaoResultado },
    "avaliacao-todos":      { jobs: "/avaliacao/runs", rota: (j, q) => `/avaliacao/runs/${j}/itens?${q}`, render: cartaoResultado },
  };
  async function listarJobsDaInspecao() {
    const f = FONTES[$("in-fonte").value];
    const jobs = await api(f.jobs);
    const rot = (j) => `${j.id} · ${j.status} · ${j.modelo || j.aluno || j.modelo_juiz || ""} ${j.aceitos != null ? `· aceitos ${j.aceitos}` : ""}${j.aprovados != null ? `· aprovados ${j.aprovados}` : ""}`;
    opcoes($("in-job"), jobs, rot, (j) => j.id, "nenhum job");
    estado.insp.offset = 0;
    arquivosDaInspecao();
  }
  // Rotas de arquivo (JSONL inteiro) que existem so' para o on-policy.
  function arquivosDaInspecao() {
    const job = $("in-job").value, onPolicy = $("in-fonte").value.startsWith("on-policy");
    $("in-arquivos").innerHTML = onPolicy && job
      ? ["sft", "dpo", "amostras"].map((n) => `<button class="mini-btn in-arquivo" data-nome="${n}">${n}.jsonl</button>`).join("")
      : "";
    document.querySelectorAll(".in-arquivo").forEach((b) => b.onclick = () => comBotao(b, () => baixarRota(`/on-policy/jobs/${job}/arquivo/${b.dataset.nome}`, `${job}-${b.dataset.nome}.jsonl`)));
  }
  $("in-fonte").onchange = () => listarJobsDaInspecao().catch((e) => toast(e.message, true));
  $("in-job").onchange = arquivosDaInspecao;
  $("in-baixar").onclick = () => {
    const fonte = $("in-fonte").value, f = FONTES[fonte], job = $("in-job").value;
    if (!job) return toast("Nenhum job selecionado.", true);
    comBotao($("in-baixar"), async () => {
      const dados = f.semPaginacao ? await api(f.rota(job)) : await todasAsPaginas((q) => f.rota(job, q));
      salvarJson(`${fonte}-${job}.json`, dados);
    });
  };
  $("in-carregar").onclick = () => { estado.insp.offset = 0; inspecionar(); };
  $("in-proximo").onclick = () => { estado.insp.offset += +$("in-limite").value; inspecionar(); };
  $("in-anterior").onclick = () => { estado.insp.offset = Math.max(0, estado.insp.offset - +$("in-limite").value); inspecionar(); };
  async function inspecionar() {
    const f = FONTES[$("in-fonte").value], job = $("in-job").value;
    if (!job) return toast("Nenhum job selecionado.", true);
    const limite = +$("in-limite").value;
    try {
      let dados = await api(f.rota(job, `limite=${limite}&offset=${estado.insp.offset}`));
      if (f.semPaginacao) dados = dados.slice(estado.insp.offset, estado.insp.offset + limite);
      $("in-pagina").textContent = `itens ${estado.insp.offset + 1}–${estado.insp.offset + dados.length}`;
      $("in-lista").innerHTML = dados.length ? dados.map(f.render).join("") : "<div class='aviso'>Nada nesta página.</div>";
      const detalhe = await api(`${f.jobs}/${job}`);
      $("in-resumo").innerHTML = resumoDoJob(detalhe);
    } catch (e) { toast(e.message, true); }
  }
  function resumoDoJob(j) {
    const partes = [`<b class="${classeStatus(j.status)}">${esc(j.status)}</b>`];
    for (const k of ["gerados", "aceitos", "descartados", "exemplos", "avaliados", "aprovados", "reprovados", "amostras", "amostras_aprovadas", "linhas_sft", "pares_dpo", "itens", "respondidos"]) if (j[k] != null) partes.push(`${k}: <b>${j[k]}</b>`);
    if (j.custo_usd != null) partes.push(`custo: <b>US$ ${(+j.custo_usd).toFixed(4)}</b>`);
    const motivos = j.descartes_por_motivo || j.reprovacoes_por_motivo;
    if (motivos && Object.keys(motivos).length) partes.push(`motivos: ${esc(Object.entries(motivos).map(([k, v]) => `${k}=${v}`).join(", "))}`);
    if (j.erro) partes.push(`<span class="erro">${esc(j.erro)}</span>`);
    return partes.join(" · ");
  }
  const campo = (titulo, valor, cls = "") => valor == null || valor === "" ? "" : `<div class="campo"><b>${titulo}</b><span class="${cls}">${esc(valor)}</span></div>`;
  const metas = (...xs) => `<div class="meta">${xs.filter(Boolean).map((x) => `<span class="rotulo">${esc(x)}</span>`).join("")}</div>`;
  function cartaoExemplo(e) {
    return `<div class="item">${metas(e.id, e.tipo, e.dificuldade, e.cenario_id ? `cenário ${e.cenario_id}` : "", e.citacoes_validas ? "citações válidas" : "CITAÇÃO INVÁLIDA")}
      ${campo("Pergunta", e.pergunta)}${campo("Raciocínio", e.raciocinio)}${campo("Resposta", e.resposta)}
      ${campo("Artigos citados", e.artigos_citados.join(", "))}${campo("Citações inválidas", e.artigos_citados_invalidos.join(", "), "erro")}</div>`;
  }
  function cartaoDescartado(d) { return `<div class="item">${metas(d.motivo, d.tipo)}${campo("Pergunta", d.pergunta)}${campo("Detalhe", d.detalhe)}</div>`; }
  function cartaoDecisao(d) {
    const v = d.veredito || {};
    return `<div class="item">${metas(d.exemplo_id, d.tipo, d.aprovado ? "APROVADO" : "REPROVADO", d.score_incerteza != null ? `incerteza ${d.score_incerteza}` : "", v.nota != null ? `nota ${v.nota}` : "")}
      ${campo("Motivos", d.motivos.join(", "), d.aprovado ? "" : "erro")}${campo("Problemas (juiz)", v.problemas)}
      ${d.reamostragens.length ? `<details><summary>${d.reamostragens.length} reamostragem(ns) do professor</summary>${d.reamostragens.map((r) => `<p class="texto">${esc(r)}</p>`).join("")}</details>` : ""}</div>`;
  }
  function cartaoAmostra(a) {
    const v = a.veredito || {};
    return `<div class="item">${metas(a.prompt_id, `amostra ${a.indice}`, a.aprovada ? "APROVADA" : "reprovada", v.categoria || "", v.similaridade != null ? `sim ${v.similaridade}` : "")}
      ${campo("Texto", a.texto)}${campo("Artigos inexistentes", a.artigos_inexistentes.join(", "), "erro")}${campo("Justificativa", v.justificativa)}${campo("Erro", a.erro, "erro")}</div>`;
  }
  function cartaoResultado(r) {
    const v = r.veredito || {};
    return `<div class="item">${metas(r.item.id, r.item.tipo, r.acerto ? "ACERTO" : "ERRO", r.alucinacao ? "ALUCINAÇÃO" : "", v.categoria || "")}
      ${campo("Pergunta", r.item.pergunta)}${campo("Resposta", r.resposta.texto)}${campo("Referência", r.item.referencia)}
      ${campo("Artigos inexistentes", r.resposta.artigos_inexistentes.join(", "), "erro")}${campo("Justificativa do juiz", v.justificativa)}${campo("Erro", r.resposta.erro, "erro")}</div>`;
  }

  // ------------------------------------------------------------ PAINEL
  $("pa-atualizar").onclick = painel;
  async function painel() {
    try {
      const [saude, prov, custos, s, f, op, tr, datasets] = await Promise.all([
        fetch("/health/ready").then((r) => r.json()), api("/usage/providers"), api("/relatorios/custos"),
        api("/sintese/jobs"), api("/filtro/jobs"), api("/on-policy/jobs"), api("/treinos"), api("/datasets"),
      ]);
      await carregarRuns();
      $("pa-saude").textContent = JSON.stringify({ saude: saude.checks, papeis: prov.papeis.map((p) => `${p.papel}: ${p.modelo}${p.credencial_presente ? "" : " (SEM CREDENCIAL)"}`), avisos: prov.avisos }, null, 2);
      const linhas = Object.entries(custos.por_etapa).map(([e, c]) => `<tr><td>${esc(e)}</td><td class="num">${c.chamadas}</td><td class="num">${c.input_tokens}</td><td class="num">${c.output_tokens}</td><td class="num">${(+c.custo_usd).toFixed(4)}</td></tr>`).join("");
      $("pa-custos").innerHTML = `<tr><th>Etapa</th><th>Chamadas</th><th>Entrada</th><th>Saída</th><th>US$</th></tr>${linhas}<tr><td><b>total</b></td><td class="num">${custos.total.chamadas}</td><td class="num">${custos.total.input_tokens}</td><td class="num">${custos.total.output_tokens}</td><td class="num"><b>${(+custos.total.custo_usd).toFixed(4)}</b></td></tr>`;
      const jobs = [
        ...s.map((j) => ({ tipo: "síntese", ...j, resumo: `aceitos ${j.aceitos}/${j.gerados}` })),
        ...f.map((j) => ({ tipo: "filtro", ...j, resumo: `aprovados ${j.aprovados}/${j.exemplos}` })),
        ...op.map((j) => ({ tipo: "on-policy", ...j, resumo: `sft ${j.linhas_sft} · dpo ${j.pares_dpo}` })),
        ...tr.map((j) => ({ tipo: "treino", ...j, resumo: `${j.modelo_local_id} · ${j.disponivel_localmente == null ? "não verificado" : j.disponivel_localmente ? "servido" : "NÃO servido"}`, custo_usd: null })),
      ].sort((a, b) => (b.criado_em || "").localeCompare(a.criado_em || ""));
      $("pa-jobs").innerHTML = `<tr><th>Tipo</th><th>Id</th><th>Status</th><th>Resumo</th><th>US$</th><th>Criado</th></tr>` +
        jobs.map((j) => `<tr><td>${j.tipo}</td><td><code>${esc(j.id)}</code></td><td class="${classeStatus(j.status)}">${esc(j.status)}</td><td>${esc(j.resumo)}</td><td class="num">${j.custo_usd == null ? "-" : (+j.custo_usd).toFixed(4)}</td><td>${esc((j.criado_em || "").slice(0, 16).replace("T", " "))}</td></tr>`).join("");
      $("pa-datasets").innerHTML = `<tr><th>Versão</th><th>Nome</th><th>Treino</th><th>Held-out</th><th>Criado</th><th>Baixar</th></tr>` +
        (datasets.length ? datasets.map((d) => `<tr><td><code>${esc(d.versao)}</code></td><td>${esc(d.nome)}</td><td class="num">${d.treino.total}</td><td class="num">${d.held_out.total}</td><td>${esc((d.criado_em || "").slice(0, 16).replace("T", " "))}</td><td>` +
          `<button class="mini-btn pa-ds" data-versao="${esc(d.versao)}" data-o="manifesto">manifesto.json</button>` +
          `<button class="mini-btn pa-ds" data-versao="${esc(d.versao)}" data-o="treino">treino.jsonl</button>` +
          `<button class="mini-btn pa-ds" data-versao="${esc(d.versao)}" data-o="held_out">held_out.jsonl</button></td></tr>`).join("")
        : `<tr><td colspan="6" class="mini">nenhum dataset</td></tr>`);
      document.querySelectorAll(".pa-ds").forEach((b) => b.onclick = () => comBotao(b, () => {
        const v = b.dataset.versao, o = b.dataset.o;
        return o === "manifesto" ? api(`/datasets/${v}`).then((m) => salvarJson(`${v}-manifesto.json`, m)) : baixarRota(`/datasets/${v}/arquivo/${o}`, `${v}-${o}.jsonl`);
      }));
      $("pa-runs").innerHTML = `<tr><th></th><th>Id</th><th>Modelo</th><th>Conjunto</th><th>Status</th><th>n</th><th>Acurácia [IC95]</th><th>Alucinação [IC95]</th><th>Abst.</th><th>p50 ms</th><th>US$/item</th><th>Medida</th><th></th></tr>` +
        estado.runs.map((r) => { const m = r.metricas; return `<tr><td><input type="checkbox" class="pa-sel" value="${esc(r.id)}"></td><td><code>${esc(r.id)}</code></td><td><code>${esc(r.modelo)}</code></td><td>${esc(r.parametros.conjunto)}${r.parametros.com_rag ? " +RAG" : ""}</td><td class="${classeStatus(r.status)}">${esc(r.status)}</td><td class="num">${m.julgados}</td><td>${pct(m.acuracia, m.acuracia_ic95)}</td><td>${pct(m.taxa_de_alucinacao, m.taxa_de_alucinacao_ic95)}</td><td>${m.abstencoes_esperadas ? `${m.abstencoes_adequadas}/${m.abstencoes_esperadas}` : "-"}</td><td class="num">${m.latencia_p50_ms ?? "-"}</td><td class="num">${m.custo_usd_por_item ?? "-"}</td><td><code>${esc(r.versao_da_medida)}</code></td><td><button class="mini-btn pa-baixar-run" data-run="${esc(r.id)}">baixar</button></td></tr>`; }).join("");
      document.querySelectorAll(".pa-sel").forEach((c) => c.onchange = camposDeRotulo);
      document.querySelectorAll(".pa-baixar-run").forEach((b) => b.onclick = () => comBotao(b, async () => {
        const id = b.dataset.run;
        const [run, itens] = await Promise.all([api(`/avaliacao/runs/${id}`), todasAsPaginas((q) => `/avaliacao/runs/${id}/itens?${q}`)]);
        salvarJson(`avaliacao-${id}.json`, { run, itens });
      }));
      camposDeRotulo();
    } catch (e) { toast(e.message, true); }
  }
  $("pa-baixar-custos").onclick = () => comBotao($("pa-baixar-custos"), async () => salvarJson("custos.json", await api("/relatorios/custos")));
  function camposDeRotulo() {
    const sel = [...document.querySelectorAll(".pa-sel:checked")].map((c) => c.value);
    $("pa-rotulos").innerHTML = sel.map((id) => `<label class="bloco"><code>${esc(id)}</code> <input data-run="${esc(id)}" placeholder="rótulo da condição, ex.: C3 SFT filtrado"></label>`).join("") || "<span class='mini'>nenhum run marcado</span>";
  }
  $("pa-comparar").onclick = async () => {
    try {
      const c = await api(`/avaliacao/comparar?a=${$("pa-cmp-a").value}&b=${$("pa-cmp-b").value}`);
      $("pa-cmp-leitura").innerHTML = `<b>${esc(c.leitura)}</b>`; $("pa-cmp").textContent = JSON.stringify(c, null, 2);
      estado.cmp = c; $("pa-cmp-baixar").disabled = false;
    } catch (e) { toast(e.message, true); }
  };
  $("pa-cmp-baixar").onclick = () => { if (estado.cmp) salvarJson(`comparacao-${estado.cmp.run_a}-vs-${estado.cmp.run_b}.json`, estado.cmp); };
  $("pa-tabela").onclick = async () => {
    const inputs = [...document.querySelectorAll("#pa-rotulos input")];
    if (!inputs.length) return toast("Marque runs na tabela.", true);
    const runs = inputs.map((i) => i.dataset.run).join(",");
    const rotulos = inputs.filter((i) => i.value.trim()).map((i) => `${i.dataset.run}:${i.value.trim()}`).join(",");
    try {
      const consulta = `runs=${runs}&rotulos=${encodeURIComponent(rotulos)}`;
      const md = await api(`/relatorios/condicoes.md?${consulta}`);
      $("pa-tabela-md").textContent = md;
      estado.tabela = { consulta, md }; $("pa-tabela-baixar-md").disabled = false; $("pa-tabela-baixar-json").disabled = false;
      const linhas = md.split("\n").filter((l) => l.startsWith("|")).map((l) => l.slice(1, -1).split("|").map((c) => c.trim()));
      const [cab, , ...corpo] = linhas;
      $("pa-tabela-html").innerHTML = `<table class="tabela"><tr>${cab.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>${corpo.map((r) => `<tr>${r.map((c) => `<td>${esc(c.replace(/`/g, ""))}</td>`).join("")}</tr>`).join("")}</table>`;
    } catch (e) { toast(e.message, true); }
  };

  $("pa-tabela-baixar-md").onclick = () => { if (estado.tabela) salvarArquivo("tabela-condicoes.md", estado.tabela.md, "text/markdown"); };
  $("pa-tabela-baixar-json").onclick = () => comBotao($("pa-tabela-baixar-json"), async () => {
    if (!estado.tabela) throw new Error("Gere a tabela primeiro.");
    salvarJson("tabela-condicoes.json", await api(`/relatorios/condicoes?${estado.tabela.consulta}`));
  });

  // ------------------------------------------------------------ início
  mostrarChave();
  if (chave()) carregarTudo();
})();
