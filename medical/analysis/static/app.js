/*
Copyright 2025 the LlamaFactory team.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
*/
"use strict";

const state = {
  doctor: null,
  doctors: [],
  overview: null,
  diseases: [],
  selectedPatient: null,
  llmConfigured: null,
  baseFormulaVersion: 0,
  baseFormula: { strata: null, results: {}, loading: new Set(), selected: null },
};
const $ = (selector) => document.querySelector(selector);
const number = new Intl.NumberFormat("zh-CN");
const percent = (value, digits = 1) => value == null ? "—" : `${(value * 100).toFixed(digits)}%`;
const escapeHtml = (value) => String(value ?? "").replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));

async function api(path, includeDoctor = true, options = {}) {
  const url = new URL(path, window.location.origin);
  if (includeDoctor && state.doctor) url.searchParams.set("doctor", state.doctor);
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try { message = (await response.json()).detail || message; } catch (_) { /* response is not JSON */ }
    throw new Error(message);
  }
  return response.json();
}

function showError(error) {
  const banner = $("#error-banner");
  banner.textContent = `载入失败：${error.message}`;
  banner.hidden = false;
}

function metric(label, value, detail = "") {
  return `<div class="metric"><div class="metric-label">${escapeHtml(label)}</div><div class="metric-value">${escapeHtml(value)}</div><div class="metric-detail">${escapeHtml(detail)}</div></div>`;
}

function renderBars(target, rows, labelKey, valueKey, formatter = number.format) {
  const element = $(target);
  if (!rows.length) { element.innerHTML = '<div class="empty-state">暂无可展示数据</div>'; return; }
  const max = Math.max(...rows.map((row) => Number(row[valueKey]) || 0), 1);
  element.innerHTML = rows.map((row) => `
    <div class="bar-row" title="${escapeHtml(row[labelKey])}: ${escapeHtml(formatter(row[valueKey]))}">
      <span class="bar-label">${escapeHtml(row[labelKey])}</span>
      <span class="bar-track"><span class="bar-fill" style="width:${Math.max(1, (Number(row[valueKey]) || 0) / max * 100)}%"></span></span>
      <span class="bar-value">${escapeHtml(formatter(row[valueKey]))}</span>
    </div>`).join("");
}

async function loadOverview() {
  const data = await api("/api/overview");
  state.overview = data;
  $("#source-status").textContent = `${data.doctor_name || "当前医生"} · ${number.format(data.records)} 次问诊`;
  const selectedOption = $("#doctor-select").selectedOptions[0];
  if (selectedOption) selectedOption.textContent = `${data.doctor_name || selectedOption.textContent} · ${number.format(data.records)} 诊`;
  $(".status-dot").classList.add("ready");
  $("#overview-note").textContent = `${number.format(data.patients)} 位患者，${number.format(data.comparable_revisit_pairs)} 组可比较复诊对。`;
  $("#overview-metrics").innerHTML = [
    metric("问诊记录", number.format(data.records), `${data.first_visits} 初诊 · ${data.revisits} 复诊`),
    metric("患者", number.format(data.patients), `${data.multi_visit_patients} 位有多次就诊`),
    metric("内服方就诊", number.format(data.internal_visits), `${data.external_only_visits} 次仅外用处方`),
    metric("diagnosis_illness 病种", number.format(data.disease_count), "按字段原值统计，不做类别合并"),
    metric("前方保留率", percent(data.revisit_summary.median_retention), `相似度中位数 ${percent(data.revisit_summary.median_jaccard)}`),
  ].join("");
  renderBars("#disease-chart", data.top_diseases.slice(0, 9), "disease", "visit_count");
  renderBars("#drug-chart", data.top_drugs.slice(0, 9), "drug_name", "visit_rate", (value) => percent(value, 0));
  const quality = [
    [data.quality.revisit_patients_without_prior_record, "复诊患者缺少此前记录，无法构造变化对"],
    [data.quality.missing_syndrome, "次问诊缺少结构化证候字段"],
    [data.quality.multiple_internal_prescriptions, "次问诊存在多张内服方，分析选药味最多的主方"],
    [data.quality.unclassified_disease, "次问诊未能归入明确疾病"],
  ];
  $("#quality-list").innerHTML = quality.map(([value, label]) => `<div class="quality-item"><strong>${number.format(value)}</strong><span>${escapeHtml(label)}</span></div>`).join("");
}

async function loadDiseases() {
  const [data, metadata] = await Promise.all([
    api("/api/diseases?limit=100&minimum_patients=2"),
    api("/api/metadata"),
  ]);
  state.diseases = data.items;
  const options = data.items.map((item) => `<option value="${escapeHtml(item.disease)}">${escapeHtml(item.disease)}（${item.visit_count}）</option>`).join("");
  $("#disease-select").innerHTML = options;
  $("#revisit-disease").innerHTML = `<option value="">全部疾病</option>${options}`;
  $("#revisit-symptom").innerHTML = `<option value="">全部症状</option>${metadata.symptoms.map((item) => `<option value="${escapeHtml(item.symptom)}">${escapeHtml(item.symptom)}（${item.count}）</option>`).join("")}`;
  if (data.items.length) await loadDiseaseDetail(data.items[0].disease);
}

const formulaKey = (item) => encodeURIComponent(JSON.stringify([item.disease, item.syndrome_value ?? item.syndrome]));

function formulaDrugTags(drugs, limit = 20) {
  if (!drugs?.length) return '<span class="muted">暂无候选药物</span>';
  return `<div class="formula-drugs">${drugs.slice(0, limit).map((drug) => `<span class="formula-drug">${escapeHtml(drug)}</span>`).join("")}${drugs.length > limit ? `<span class="formula-drug more">+${drugs.length - limit}</span>` : ""}</div>`;
}

function supportBar(value) {
  const width = Math.max(0, Math.min(100, Number(value || 0) * 100));
  return `<div class="support-track" aria-label="支持度 ${percent(value)}"><span style="width:${width}%"></span></div>`;
}

function baseFormulaFilters() {
  return {
    disease: $("#base-formula-disease").value,
    syndrome: $("#base-formula-syndrome").value,
    eligibleOnly: $("#base-formula-eligible-only").checked,
  };
}

function filteredBaseFormulaStrata() {
  const items = state.baseFormula.strata?.items || [];
  const filters = baseFormulaFilters();
  return items.filter((item) => (
    (!filters.disease || item.disease === filters.disease)
    && (!filters.syndrome || item.syndrome === filters.syndrome)
    && (!filters.eligibleOnly || item.status === "eligible")
  ));
}

function updateBaseFormulaSyndromes() {
  const selectedDisease = $("#base-formula-disease").value;
  const current = $("#base-formula-syndrome").value;
  const eligibleOnly = $("#base-formula-eligible-only").checked;
  const syndromes = [...new Set((state.baseFormula.strata?.items || [])
    .filter((item) => !eligibleOnly || item.status === "eligible")
    .filter((item) => !selectedDisease || item.disease === selectedDisease)
    .map((item) => item.syndrome))].sort((left, right) => left.localeCompare(right, "zh-CN"));
  $("#base-formula-syndrome").innerHTML = `<option value="">全部证候</option>${syndromes.map((syndrome) => `<option value="${escapeHtml(syndrome)}">${escapeHtml(syndrome)}</option>`).join("")}`;
  if (syndromes.includes(current)) $("#base-formula-syndrome").value = current;
}

function populateBaseFormulaFilters() {
  const currentDisease = $("#base-formula-disease").value;
  const eligibleOnly = $("#base-formula-eligible-only").checked;
  const diseases = [...new Set((state.baseFormula.strata?.items || [])
    .filter((item) => !eligibleOnly || item.status === "eligible")
    .map((item) => item.disease))]
    .sort((left, right) => left.localeCompare(right, "zh-CN"));
  $("#base-formula-disease").innerHTML = `<option value="">全部疾病</option>${diseases.map((disease) => `<option value="${escapeHtml(disease)}">${escapeHtml(disease)}</option>`).join("")}`;
  if (diseases.includes(currentDisease)) $("#base-formula-disease").value = currentDisease;
  updateBaseFormulaSyndromes();
}

function renderBaseFormulaMetrics() {
  const data = state.baseFormula.strata;
  if (!data) return;
  const loaded = Object.values(state.baseFormula.results).filter((result) => result.status === "ready").length;
  $("#base-formula-metrics").innerHTML = [
    metric("疾病—证候组合", number.format(data.stratum_count), "严格使用结构化证候字段"),
    metric("可挖掘组合", number.format(data.eligible_count), `至少 ${data.minimum_patients} 位独立患者`),
    metric("样本不足", number.format(data.stratum_count - data.eligible_count), "保留目录，不推测基础方"),
    metric("已计算基础方", number.format(loaded), "相同参数结果由服务端缓存"),
  ].join("");
}

function baseFormulaCard(item) {
  const key = formulaKey(item);
  const response = state.baseFormula.results[key];
  const loading = state.baseFormula.loading.has(key);
  const selected = state.baseFormula.selected === key;
  const report = response?.base_formula;
  const representative = report?.representative_base_formula;
  const eligible = item.status === "eligible";
  let content;
  if (!eligible) {
    content = `<div class="formula-card-empty">独立患者数低于当前阈值，暂不推测基础方。</div>`;
  } else if (loading) {
    content = `<div class="formula-card-empty loading-line">正在进行患者加权挖掘…</div>`;
  } else if (representative) {
    content = `${formulaDrugTags(representative.drugs, 16)}
      <div class="formula-card-support"><span>患者加权支持度 <strong>${percent(representative.support)}</strong></span><span>95% CI ${percent(representative.support_ci_low)}–${percent(representative.support_ci_high)}</span></div>
      ${supportBar(representative.support)}`;
  } else {
    content = `<div class="formula-card-empty">尚未计算候选基础方。</div>`;
  }
  return `<article class="base-formula-card ${selected ? "is-selected" : ""} ${eligible ? "" : "is-insufficient"}">
    <div class="formula-card-head"><div><p>${escapeHtml(item.syndrome)}</p><h3>${escapeHtml(item.disease)}</h3></div><span class="formula-status ${eligible ? "eligible" : ""}">${eligible ? "可挖掘" : "样本不足"}</span></div>
    <div class="formula-card-counts"><span><strong>${number.format(item.patient_count)}</strong> 位患者</span><span><strong>${number.format(item.visit_count)}</strong> 次问诊</span></div>
    <div class="formula-card-body">${content}</div>
    ${eligible ? `<button type="button" class="formula-card-action" data-formula-key="${key}" ${loading ? "disabled" : ""}>${representative ? "查看完整方型" : loading ? "计算中…" : "计算候选基础方"}</button>` : ""}
  </article>`;
}

function renderBaseFormulaCards() {
  const items = filteredBaseFormulaStrata();
  $("#base-formula-list-note").textContent = `当前显示 ${items.length} 个组合；自动计算患者数最多的前 6 个可挖掘组合。`;
  $("#base-formula-cards").innerHTML = items.length
    ? items.map(baseFormulaCard).join("")
    : '<div class="panel empty-analysis formula-empty">当前筛选条件下没有疾病—证候组合</div>';
  const visibleKeys = new Set(items.map(formulaKey));
  if (state.baseFormula.selected && !visibleKeys.has(state.baseFormula.selected)) {
    state.baseFormula.selected = null;
    $("#base-formula-detail-panel").hidden = true;
  }
  document.querySelectorAll("[data-formula-key]").forEach((button) => button.addEventListener("click", () => {
    const item = (state.baseFormula.strata?.items || []).find((row) => formulaKey(row) === button.dataset.formulaKey);
    if (item) loadBaseFormulaDetail(item, true).catch(showError);
  }));
}

function renderBaseFormulaDetail(item, response) {
  const report = response.base_formula;
  const representative = report.representative_base_formula;
  const components = report.latent_base_formulas.components || [];
  const frequent = (report.frequent_drugs || []).slice(0, 14);
  const alternatives = (report.candidate_base_formulas || [])
    .filter((candidate) => candidate.drugs.join("|") !== representative.drugs.join("|"))
    .slice(0, 8);
  $("#base-formula-detail-panel").hidden = false;
  $("#base-formula-detail-title").textContent = `${item.disease} · ${item.syndrome}`;
  $("#base-formula-detail-meta").textContent = `${item.patient_count} 位患者 · ${item.visit_count} 次问诊`;
  $("#base-formula-detail").innerHTML = `<section class="formula-hero">
      <div><p class="formula-detail-label">代表性经验候选药组 · ${representative.length} 味</p>${formulaDrugTags(representative.drugs)}</div>
      <div class="formula-score"><strong>${percent(representative.support)}</strong><span>患者加权支持度</span><small>95% CI ${percent(representative.support_ci_low)}–${percent(representative.support_ci_high)}</small></div>
    </section>
    ${report.warnings.length ? `<div class="formula-warning">${report.warnings.map(escapeHtml).join("；")}</div>` : ""}
    <div class="two-column formula-detail-grid">
      <section><h4>NMF 潜在方型</h4><div class="latent-formulas">${components.map((component, index) => `<div class="latent-formula"><div><strong>方型 ${index + 1}</strong><span>患者平均权重 ${percent(component.mean_patient_weight)}</span></div>${formulaDrugTags(component.drugs.map((drug) => drug.drug), 12)}</div>`).join("") || '<p class="muted">暂无潜在方型</p>'}</div></section>
      <section><h4>核心药物覆盖率</h4><div class="formula-frequency">${frequent.map((drug) => `<div><span>${escapeHtml(drug.drug)}</span>${supportBar(drug.support)}<strong>${percent(drug.support, 0)}</strong></div>`).join("")}</div></section>
    </div>
    <section class="formula-alternatives"><h4>其他高支持度药组</h4><div class="table-wrap"><table><thead><tr><th>候选药组</th><th>药味数</th><th>支持度</th><th>95% CI</th></tr></thead><tbody>${alternatives.map((candidate) => `<tr><td>${escapeHtml(candidate.drugs.join("、"))}</td><td>${candidate.length}</td><td>${percent(candidate.support)}</td><td>${percent(candidate.support_ci_low)}–${percent(candidate.support_ci_high)}</td></tr>`).join("")}</tbody></table></div></section>`;
}

async function loadBaseFormulaDetail(item, showDetail = false) {
  const key = formulaKey(item);
  const requestVersion = state.baseFormulaVersion;
  if (state.baseFormula.results[key]) {
    if (showDetail && state.baseFormula.results[key].status === "ready") {
      state.baseFormula.selected = key;
      renderBaseFormulaCards();
      renderBaseFormulaDetail(item, state.baseFormula.results[key]);
    }
    return state.baseFormula.results[key];
  }
  if (state.baseFormula.loading.has(key)) return null;
  const requestedDoctor = state.doctor;
  state.baseFormula.loading.add(key);
  renderBaseFormulaCards();
  const params = new URLSearchParams({
    disease: item.disease,
    syndrome: item.syndrome_value,
    minimum_patients: $("#base-formula-min-patients").value,
    minimum_support: "0.3",
    maximum_pattern_length: "20",
    pattern_limit: "30",
    component_count: "3",
    bootstrap_rounds: "500",
  });
  try {
    const response = await api(`/api/base-formulas?${params}`);
    if (state.doctor !== requestedDoctor || state.baseFormulaVersion !== requestVersion) return null;
    state.baseFormula.results[key] = response;
    if (showDetail && response.status === "ready") {
      state.baseFormula.selected = key;
      renderBaseFormulaDetail(item, response);
    }
    return response;
  } finally {
    if (state.doctor === requestedDoctor && state.baseFormulaVersion === requestVersion) {
      state.baseFormula.loading.delete(key);
      renderBaseFormulaMetrics();
      renderBaseFormulaCards();
    }
  }
}

async function loadBaseFormulaStrata(force = false) {
  if (state.baseFormula.strata && !force) {
    renderBaseFormulaMetrics();
    renderBaseFormulaCards();
    return;
  }
  const requestedDoctor = state.doctor;
  const requestVersion = ++state.baseFormulaVersion;
  state.baseFormula = { strata: null, results: {}, loading: new Set(), selected: null };
  $("#base-formula-detail-panel").hidden = true;
  $("#base-formula-metrics").innerHTML = metric("正在读取", "…", "疾病—证候分层目录");
  $("#base-formula-cards").innerHTML = '<div class="panel empty-analysis formula-empty">正在载入疾病—证候组合…</div>';
  const minimumPatients = $("#base-formula-min-patients").value;
  const data = await api(`/api/base-formulas/strata?minimum_patients=${minimumPatients}`);
  if (state.doctor !== requestedDoctor || state.baseFormulaVersion !== requestVersion) return;
  state.baseFormula.strata = data;
  populateBaseFormulaFilters();
  renderBaseFormulaMetrics();
  renderBaseFormulaCards();
  const preload = data.items.filter((item) => item.status === "eligible").slice(0, 6);
  await Promise.allSettled(preload.map((item) => loadBaseFormulaDetail(item, false)));
  if (state.doctor !== requestedDoctor || state.baseFormulaVersion !== requestVersion) return;
  const firstReady = preload.find((item) => state.baseFormula.results[formulaKey(item)]?.status === "ready");
  if (firstReady) {
    state.baseFormula.selected = formulaKey(firstReady);
    renderBaseFormulaCards();
    renderBaseFormulaDetail(firstReady, state.baseFormula.results[state.baseFormula.selected]);
  }
}

async function loadDiseaseDetail(disease) {
  resetLLMAnalysis();
  const [data, comparisons] = await Promise.all([
    api(`/api/diseases/${encodeURIComponent(disease)}`),
    api(`/api/diseases/${encodeURIComponent(disease)}/comparisons?case_limit=8&pair_limit=8&minimum_similarity=0.8`),
  ]);
  const sameDisease = comparisons.same_disease_different_prescriptions;
  const sameFormula = comparisons.same_prescription_different_diseases;
  $("#disease-metrics").innerHTML = [
    metric("问诊次数", number.format(data.visit_count), `${data.patient_count} 位患者`),
    metric("内服方就诊", number.format(data.internal_visit_count), `${data.top_drugs.length} 味高频药已展示`),
    metric("不同处方", number.format(sameDisease.distinct_prescriptions), "按药物、剂量、单位区分"),
    metric("跨病种相似方", number.format(sameFormula.pair_count), `药味相似度 ≥ ${percent(sameFormula.minimum_similarity, 0)}`),
  ].join("");
  $("#disease-drug-table").innerHTML = data.top_drugs.map((drug) => `<tr>
    <td><span class="drug-name">${escapeHtml(drug.drug_name)}</span></td>
    <td>${drug.visit_count}</td><td>${drug.patient_count}</td><td>${percent(drug.disease_rate, 0)}</td><td>${percent(drug.global_rate, 0)}</td>
    <td><span class="lift ${drug.lift < 1 ? "low" : ""}">${drug.lift ?? "—"}</span></td>
    <td>${drug.median_dose ?? "—"} ${escapeHtml(drug.unit)}</td></tr>`).join("");
  renderDiseaseCases(sameDisease);
  renderFormulaPairs(sameFormula);
}

function renderDiseaseCases(data) {
  $("#same-disease-meta").textContent = `${data.visit_count} 次内服方 · ${data.distinct_prescriptions} 种处方`;
  $("#same-disease-cases").innerHTML = data.items.length ? data.items.map((item) => `<section class="case-item">
    <div class="case-head"><div><div class="case-identity">患者 ${escapeHtml(item.patient_id)} · ${escapeHtml(item.date)}</div><div class="case-meta">${escapeHtml(item.sex || "性别未记录")} · ${escapeHtml(item.age || "年龄未记录")}岁 · 该处方出现 ${item.prescription_occurrences} 次</div></div>
    <button type="button" class="case-action" data-llm-visit="${item.visit_id}">分析处方</button></div>
    <div class="case-history">${escapeHtml(item.new_medical_history || "未记录本次病情")}</div>
    <div class="case-prescription"><strong>处方：</strong>${prescriptionDetails(item.drugs)}</div>
  </section>`).join("") : '<div class="empty-analysis">当前疾病没有可展示的内服处方病例</div>';
  document.querySelectorAll("[data-llm-visit]").forEach((button) => button.addEventListener("click", () => analyzePrescription(button.dataset.llmVisit, button)));
}

function renderPairCase(item) {
  return `<div class="pair-case"><strong>${escapeHtml(item.diagnosis_illness)} · 患者 ${escapeHtml(item.patient_id)} · ${escapeHtml(item.date)}</strong>
    <p>${escapeHtml(item.new_medical_history || "未记录本次病情")}</p>
    <p><b>处方：</b>${prescriptionDetails(item.drugs)}</p></div>`;
}

function renderFormulaPairs(data) {
  $("#same-formula-meta").textContent = `药味相似度 ≥ ${percent(data.minimum_similarity, 0)} · ${data.pair_count} 组`;
  $("#same-formula-cases").innerHTML = data.items.length ? data.items.map((item) => `<section class="formula-pair">
    <span class="formula-similarity">处方药味相似度 ${percent(item.similarity, 0)}</span>
    ${renderPairCase(item.left_case)}${renderPairCase(item.right_case)}
    <div class="case-meta">共同药物：${escapeHtml(item.common_drugs.join("、") || "无")}${item.left_only.length || item.right_only.length ? `；差异：${escapeHtml([...item.left_only, ...item.right_only].join("、"))}` : ""}</div>
  </section>`).join("") : '<div class="empty-analysis">当前疾病未找到药味相似度达到阈值的跨病种病例</div>';
}

function renderLLMAnalysis(data) {
  $("#llm-status").textContent = `${data.provider} · ${data.model}${data.cached ? " · 缓存" : ""}`;
  $("#llm-analysis-content").classList.remove("empty-analysis");
  $("#llm-analysis-content").innerHTML = `<section class="llm-section"><h4>开具该处方的原因</h4><p>${escapeHtml(data.prescription_reason)}</p></section>
    <section class="llm-section"><h4>配伍逻辑</h4><p>${escapeHtml(data.compatibility_logic)}</p></section>
    <section class="llm-section"><h4>关键药物角色</h4>${data.key_drug_roles.length ? `<ul>${data.key_drug_roles.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>` : "<p>未返回</p>"}</section>
    <section class="llm-section"><h4>证据边界</h4><p>${escapeHtml(data.evidence_limits)}</p></section>`;
}

function resetLLMAnalysis() {
  $("#llm-status").textContent = state.llmConfigured === false ? "LLM 未配置" : "从病例中选择“分析处方”";
  $("#llm-analysis-content").className = "llm-analysis empty-analysis";
  $("#llm-analysis-content").textContent = "选择一个病例后，系统将使用性别、年龄、new_medical_history、过敏史、家族史、个人史、既往史和本次处方调用 LLM。";
}

async function analyzePrescription(visitId, button) {
  button.disabled = true;
  $("#llm-status").textContent = `正在分析问诊 ${visitId}`;
  $("#llm-analysis-content").className = "llm-analysis empty-analysis";
  $("#llm-analysis-content").textContent = "正在调用 LLM 分析处方原因与配伍逻辑…";
  try {
    const data = await api(`/api/visits/${visitId}/prescription-explanation`, true, { method: "POST" });
    renderLLMAnalysis(data);
  } catch (error) {
    $("#llm-status").textContent = "分析失败";
    $("#llm-analysis-content").textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

function rankList(target, items) {
  $(target).innerHTML = items.length ? items.map((item) => `<li><span>${escapeHtml(item.drug_name)}</span><span>${item.count} 次</span></li>`).join("") : "<li><span>暂无数据</span><span>—</span></li>";
}

function chips(items, type = "") {
  if (!items.length) return '<span class="muted">—</span>';
  return `<div class="chip-group">${items.slice(0, 6).map((item) => `<span class="chip ${type}">${escapeHtml(item)}</span>`).join("")}${items.length > 6 ? `<span>+${items.length - 6}</span>` : ""}</div>`;
}

async function loadRevisits() {
  const params = new URLSearchParams({ limit: "100" });
  const disease = $("#revisit-disease").value;
  const outcome = $("#revisit-outcome").value;
  const symptom = $("#revisit-symptom").value;
  if (disease) params.set("disease", disease);
  if (outcome) params.set("outcome", outcome);
  if (symptom) params.set("symptom", symptom);
  const data = await api(`/api/revisits?${params}`);
  const summary = data.summary;
  $("#revisit-metrics").innerHTML = [
    metric("复诊变化对", number.format(data.total), "前后均有可比较内服方"),
    metric("前方保留率", percent(summary.median_retention), `Jaccard ${percent(summary.median_jaccard)}`),
    metric("新增 / 停用", `+${summary.median_added ?? "—"} / −${summary.median_removed ?? "—"}`, "药味数中位数"),
    metric("完全未调整", percent(summary.unchanged_rate), `剂量变化中位 ${summary.median_dose_changes ?? "—"} 味`),
  ].join("");
  rankList("#top-added", data.top_added);
  rankList("#top-removed", data.top_removed);
  rankList("#top-dose", data.top_dose_changed);
  $("#revisit-table").innerHTML = data.items.map((item) => `<tr>
    <td><button class="patient-link" data-patient="${escapeHtml(item.patient_id)}">${escapeHtml(item.patient_id)}</button></td>
    <td>${escapeHtml(item.previous_date)}<br>→ ${escapeHtml(item.current_date)}<br><small>${item.interval_days ?? "—"} 天</small></td>
    <td>${escapeHtml(item.disease)}<br><span class="tag">${escapeHtml(item.outcome)}</span></td>
    <td>${percent(item.jaccard, 0)}<br><small>保留 ${percent(item.retention, 0)}</small></td>
    <td>${chips(item.added)}</td><td>${chips(item.removed, "removed")}</td>
    <td>${item.dose_changes.length ? item.dose_changes.slice(0, 4).map((change) => `${escapeHtml(change.drug_name)} ${change.before}→${change.after}${escapeHtml(change.unit)}`).join("<br>") : "—"}</td>
  </tr>`).join("");
  document.querySelectorAll("[data-patient]").forEach((button) => button.addEventListener("click", () => openPatient(button.dataset.patient)));
}

async function loadPatientList(query = "") {
  const data = await api(`/api/patients?limit=50&query=${encodeURIComponent(query)}`);
  const repeated = data.items.filter((item) => item.visit_count > 1);
  $("#patient-list").innerHTML = repeated.length ? repeated.map((item) => `<button class="patient-button ${state.selectedPatient === item.patient_id ? "is-active" : ""}" data-key="${escapeHtml(item.patient_id)}">
    <strong>${escapeHtml(item.patient_id)}</strong><span>${item.visit_count} 诊</span><small>${escapeHtml(item.diseases.join(" · "))} · ${item.first_date}—${item.last_date}</small></button>`).join("") : '<div class="empty-state">未找到匹配的多次就诊患者</div>';
  document.querySelectorAll(".patient-button").forEach((button) => button.addEventListener("click", () => openPatient(button.dataset.key)));
}

function changeLine(label, items) {
  return items && items.length ? `<p><strong>${label}：</strong>${escapeHtml(items.join("、"))}</p>` : "";
}

function prescriptionDetails(drugs) {
  if (!drugs?.length) return "未记录内服处方";
  return drugs.map((drug) => {
    const dose = drug.dose == null ? "" : drug.dose;
    return `${escapeHtml(drug.drug_name)}${escapeHtml(dose)}${escapeHtml(drug.unit)}`;
  }).join("、");
}

async function openPatient(patientKey) {
  switchView("patient");
  state.selectedPatient = patientKey;
  $("#patient-query").value = patientKey;
  const data = await api(`/api/patients/${encodeURIComponent(patientKey)}`);
  $("#timeline-title").textContent = `患者 ID：${data.patient_id}`;
  $("#timeline-meta").textContent = `${data.visit_count} 次就诊`;
  $("#timeline").classList.remove("empty-state");
  $("#timeline").innerHTML = data.timeline.map((visit) => {
    const dose = visit.change?.dose_changes?.map((item) => `${item.drug_name} ${item.before}→${item.after}${item.unit}`) || [];
    const change = visit.change ? `<div class="change-block">${changeLine("新增", visit.change.added)}${changeLine("停用", visit.change.removed)}${changeLine("剂量", dose)}<p>处方相似度 ${percent(visit.change.jaccard, 0)}，保留率 ${percent(visit.change.retention, 0)}</p></div>` : "";
    return `<section class="timeline-item"><div class="timeline-date">${escapeHtml(visit.date)} · ${escapeHtml(visit.is_first)}</div>
      <div class="timeline-title">${escapeHtml(visit.raw_diagnosis || visit.disease)}</div>
      <div class="timeline-meta-row"><span>反馈：${escapeHtml(visit.outcome)}</span><span>内服 ${visit.drug_count} 味</span><span>${escapeHtml(visit.process_types.join("、") || "剂型未知")}</span></div>
      <div class="timeline-history"><strong>本次症状与病情：</strong>${escapeHtml(visit.new_medical_history || "未记录")}</div>
      <div class="timeline-prescription"><strong>本次处方明细：</strong>${prescriptionDetails(visit.drugs)}</div>${change}</section>`;
  }).join("");
  await loadPatientList(patientKey);
}

async function loadDoctors() {
  const [data, llmStatus] = await Promise.all([api("/api/doctors", false), api("/api/llm/status", false)]);
  state.llmConfigured = llmStatus.configured;
  state.doctors = data.items;
  state.doctor = data.default_doctor;
  $("#doctor-select").innerHTML = data.items.map((doctor) => {
    const count = doctor.record_count == null ? "" : ` · ${number.format(doctor.record_count)} 诊`;
    return `<option value="${escapeHtml(doctor.key)}">${escapeHtml(doctor.doctor_name)}${count}</option>`;
  }).join("");
  $("#doctor-select").value = state.doctor;
}

async function loadDoctor(doctorKey) {
  const select = $("#doctor-select");
  state.doctor = doctorKey;
  state.selectedPatient = null;
  state.baseFormulaVersion += 1;
  state.baseFormula = { strata: null, results: {}, loading: new Set(), selected: null };
  select.disabled = true;
  $(".status-dot").classList.remove("ready");
  $("#source-status").textContent = "正在切换医生数据";
  $("#error-banner").hidden = true;
  $("#patient-query").value = "";
  $("#patient-list").innerHTML = "";
  $("#timeline-title").textContent = "选择一位患者";
  $("#timeline-meta").textContent = "";
  $("#timeline").className = "timeline empty-state";
  $("#timeline").textContent = "从左侧选择患者 ID，查看病情与调方过程。";
  resetLLMAnalysis();
  try {
    await Promise.all([loadOverview(), loadDiseases()]);
    const activeView = $(".tab.is-active").dataset.view;
    if (activeView === "base-formula") await loadBaseFormulaStrata();
    if (activeView === "revisit") await loadRevisits();
    if (activeView === "patient") await loadPatientList();
  } finally {
    select.disabled = false;
  }
}

function switchView(view) {
  document.querySelectorAll(".tab").forEach((tab) => {
    const active = tab.dataset.view === view;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  document.querySelectorAll(".view").forEach((section) => {
    const active = section.id === `view-${view}`;
    section.classList.toggle("is-active", active);
    section.hidden = !active;
  });
  if (view === "base-formula") loadBaseFormulaStrata().catch(showError);
  if (view === "revisit") loadRevisits().catch(showError);
  if (view === "patient" && !$("#patient-list").children.length) loadPatientList().catch(showError);
}

function bindEvents() {
  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => switchView(tab.dataset.view)));
  $("#disease-select").addEventListener("change", (event) => loadDiseaseDetail(event.target.value).catch(showError));
  $("#base-formula-min-patients").addEventListener("change", () => loadBaseFormulaStrata(true).catch(showError));
  $("#base-formula-eligible-only").addEventListener("change", () => {
    populateBaseFormulaFilters();
    renderBaseFormulaCards();
  });
  $("#base-formula-disease").addEventListener("change", () => {
    updateBaseFormulaSyndromes();
    renderBaseFormulaCards();
  });
  $("#base-formula-syndrome").addEventListener("change", renderBaseFormulaCards);
  $("#revisit-disease").addEventListener("change", () => loadRevisits().catch(showError));
  $("#revisit-symptom").addEventListener("change", () => loadRevisits().catch(showError));
  $("#revisit-outcome").addEventListener("change", () => loadRevisits().catch(showError));
  $("#doctor-select").addEventListener("change", (event) => loadDoctor(event.target.value).catch(showError));
  $("#patient-search-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const query = $("#patient-query").value.trim();
    if (/^\d+$/.test(query)) openPatient(query).catch(showError);
    else loadPatientList(query).catch(showError);
  });
}

async function init() {
  bindEvents();
  await loadDoctors();
  await loadDoctor(state.doctor);
}

init().catch(showError);
