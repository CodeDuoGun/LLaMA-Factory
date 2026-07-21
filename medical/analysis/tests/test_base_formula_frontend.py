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


def test_base_formula_view_has_navigation_filters_and_detail_region() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")

    assert 'data-view="base-formula"' in html
    assert 'id="view-base-formula"' in html
    assert 'id="base-formula-min-patients"' in html
    assert 'id="base-formula-eligible-only"' in html
    assert 'id="base-formula-disease"' in html
    assert 'id="base-formula-syndrome"' in html
    assert 'id="base-formula-cards"' in html
    assert 'id="base-formula-detail-panel"' in html


def test_base_formula_frontend_uses_lazy_directory_and_detail_queries() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    assert "/api/base-formulas/strata?minimum_patients=" in script
    assert "/api/base-formulas?${params}" in script
    assert "slice(0, 6)" in script
    assert 'if (view === "base-formula")' in script
    assert "baseFormulaVersion" in script


def test_base_formula_cards_have_responsive_styles() -> None:
    styles = (STATIC / "styles.css").read_text(encoding="utf-8")

    assert ".base-formula-grid" in styles
    assert ".base-formula-card" in styles
    assert ".formula-hero" in styles
    assert "grid-template-columns: 1fr" in styles
