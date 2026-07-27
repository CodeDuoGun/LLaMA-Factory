# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path


STATIC = Path(__file__).resolve().parents[1] / "static"


def test_overview_has_western_tcm_disease_and_syndrome_distributions() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    styles = (STATIC / "styles.css").read_text(encoding="utf-8")

    assert 'id="western-diagnosis-chart"' in html
    assert 'id="tcm-diagnosis-chart"' in html
    assert 'id="syndrome-chart"' in html
    assert 'id="overview-diagnosis-limit"' in html
    assert '<option value="9" selected>前 9 项</option>' in html
    assert '<option value="18">前 18 项</option>' in html
    assert '<option value="all">全部</option>' in html
    assert 'rows("diagnosis_illness")' in script
    assert 'rows("diagnosis_sickness")' in script
    assert 'rows("diagnosis_disease")' in script
    assert "renderDiagnosisDistributions" in script
    assert ".overview-diagnosis-grid .bar-label" in styles
    assert "white-space: pre-line" in styles
    assert "text-overflow: clip" in styles


def test_patient_view_filters_grouped_patients_by_visit_type() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    assert 'id="patient-visit-type"' in html
    assert '<option value="初诊">初诊患者（1 次问诊）</option>' in html
    assert '<option value="复诊">复诊患者（多次问诊）</option>' in html
    assert "按 patient_id 聚合" in html
    assert "visit_type=${encodeURIComponent(visitType)}" in script
    assert "openPatient(button.dataset.key, true)" in script
    assert "async function openPatient(patientKey, preservePatientList = false)" in script
    assert "renderPatientTimelines" not in script


def test_base_formula_view_has_dimensions_comparison_and_detail_regions() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")

    assert 'data-view="base-formula"' in html
    assert 'id="view-base-formula"' in html
    assert 'id="base-formula-min-patients"' in html
    assert 'id="base-formula-eligible-only"' in html
    assert 'value="diagnosis_illness"' in html
    assert 'value="diagnosis_sickness"' in html
    assert 'value="diagnosis_disease"' in html
    assert 'value="is_first"' in html
    assert 'value="disease_course" disabled' in html
    assert 'value="etiology_pathogenesis" disabled' in html
    assert 'id="base-formula-compare-button"' in html
    assert 'id="base-formula-experiment-schemes"' in html
    assert 'id="base-formula-add-scheme"' in html
    assert 'id="base-formula-comparison-panel"' in html
    assert 'id="base-formula-cards"' in html
    assert 'id="base-formula-detail-panel"' in html


def test_base_formula_frontend_uses_lazy_directory_and_detail_queries() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    assert "/api/base-formulas/dimensions" in script
    assert "/api/base-formulas/multidimensional/strata?${params}" in script
    assert "/api/base-formulas/multidimensional?${params}" in script
    assert "slice(0, 6)" in script
    assert "renderBaseFormulaComparison" in script
    assert '[["diagnosis_illness"], ["diagnosis_illness", "diagnosis_sickness"]]' in script
    assert "compareBaseFormulaSchemes" in script
    assert "formula-experiment-results" in script
    assert 'if (view === "base-formula")' in script
    assert "baseFormulaVersion" in script


def test_base_formula_cards_have_responsive_styles() -> None:
    styles = (STATIC / "styles.css").read_text(encoding="utf-8")

    assert ".base-formula-grid" in styles
    assert ".base-formula-card" in styles
    assert ".formula-hero" in styles
    assert ".formula-dimension-grid" in styles
    assert ".base-formula-comparison-panel" in styles
    assert ".formula-experiment-results" in styles
    assert "grid-template-columns: 1fr" in styles
