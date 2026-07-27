# Copyright 2025 the LlamaFactory team.
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

"""FastAPI application for the multi-doctor medical analysis dashboard."""

from __future__ import annotations

import argparse
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from medical.analysis.base_formula import (
    discover_base_formulas,
    discover_base_formulas_by_dimensions,
    formula_dimension_metadata,
    formula_strata,
    multidimensional_formula_strata,
)
from medical.analysis.catalog import DoctorCatalog
from medical.analysis.engine import MedicalAnalysis
from medical.analysis.llm import PrescriptionLLMService


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = MODULE_DIR.parent / "data" / "202301_online"
DEFAULT_DOCTOR = "1314"


@asynccontextmanager
async def lifespan(app: FastAPI):
    explicit_file = os.getenv("MEDICAL_ANALYSIS_DATA")
    if explicit_file and not Path(explicit_file).expanduser().is_file():
        raise RuntimeError(f"Medical analysis data file not found: {explicit_file}")
    app.state.catalog = DoctorCatalog.discover(
        os.getenv("MEDICAL_ANALYSIS_DATA_ROOT", str(DEFAULT_DATA_ROOT)),
        default_doctor=os.getenv("MEDICAL_ANALYSIS_DEFAULT_DOCTOR", DEFAULT_DOCTOR),
        explicit_file=explicit_file,
    )
    app.state.catalog.get()
    app.state.llm = PrescriptionLLMService()
    app.state.base_formula_cache = {}
    yield


app = FastAPI(
    title="多医生问诊处方分析",
    description="脱敏的疾病—病情—处方关系与复诊调方分析 API",
    version="1.2.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=MODULE_DIR / "static"), name="static")


def analysis(request: Request, doctor: str | None = None) -> MedicalAnalysis:
    try:
        return request.app.state.catalog.get(doctor)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=f"Doctor dataset not found: {doctor}") from error


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(MODULE_DIR / "static" / "index.html")


@app.get("/api/doctors")
def doctors(request: Request) -> dict[str, object]:
    return request.app.state.catalog.list_doctors()


@app.get("/api/health")
def health(request: Request, doctor: str | None = None) -> dict[str, object]:
    service = analysis(request, doctor)
    return {
        "status": "ok",
        "source": service.source_name,
        "doctor_name": service.doctor_name,
        "records": service.raw_record_count,
    }


@app.get("/api/overview")
def overview(request: Request, doctor: str | None = None) -> dict[str, object]:
    return analysis(request, doctor).overview()


@app.get("/api/metadata")
def metadata(request: Request, doctor: str | None = None) -> dict[str, object]:
    return analysis(request, doctor).metadata()


@app.get("/api/diseases")
def diseases(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    minimum_patients: int = Query(1, ge=1, le=100),
    doctor: str | None = None,
) -> dict[str, object]:
    return analysis(request, doctor).diseases(limit=limit, minimum_patients=minimum_patients)


@app.get("/api/base-formulas/strata")
def base_formula_strata(
    request: Request,
    minimum_patients: int = Query(30, ge=1, le=1000),
    include_unspecified_syndrome: bool = False,
    visit_type: str | None = Query(None, pattern="^(初诊|复诊)$"),
    process_type: str | None = None,
    doctor: str | None = None,
) -> dict[str, object]:
    return formula_strata(
        analysis(request, doctor),
        minimum_patients=minimum_patients,
        include_unspecified_syndrome=include_unspecified_syndrome,
        visit_type=visit_type,
        process_type=process_type,
    )


@app.get("/api/base-formulas/dimensions")
def base_formula_dimensions(request: Request, doctor: str | None = None) -> dict[str, object]:
    return formula_dimension_metadata(analysis(request, doctor))


@app.get("/api/base-formulas/multidimensional/strata")
def multidimensional_base_formula_strata(
    request: Request,
    dimensions: str = Query("diagnosis_illness,diagnosis_disease"),
    minimum_patients: int = Query(30, ge=1, le=1000),
    include_unspecified: bool = False,
    process_type: str | None = None,
    doctor: str | None = None,
) -> dict[str, object]:
    try:
        return multidimensional_formula_strata(
            analysis(request, doctor),
            [dimension.strip() for dimension in dimensions.split(",") if dimension.strip()],
            minimum_patients=minimum_patients,
            include_unspecified=include_unspecified,
            process_type=process_type,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/base-formulas/multidimensional")
def multidimensional_base_formula_query(
    request: Request,
    diagnosis_illness: str | None = None,
    diagnosis_sickness: str | None = None,
    diagnosis_disease: str | None = None,
    is_first: str | None = Query(None, pattern="^(初诊|复诊)$"),
    minimum_patients: int = Query(30, ge=1, le=1000),
    minimum_support: float = Query(0.3, gt=0, le=1),
    maximum_pattern_length: int = Query(15, ge=1, le=25),
    pattern_limit: int = Query(30, ge=1, le=200),
    component_count: int = Query(3, ge=1, le=12),
    bootstrap_rounds: int = Query(500, ge=0, le=2000),
    process_type: str | None = None,
    doctor: str | None = None,
) -> dict[str, object]:
    service = analysis(request, doctor)
    dimension_values = {
        name: value
        for name, value in {
            "diagnosis_illness": diagnosis_illness,
            "diagnosis_sickness": diagnosis_sickness,
            "diagnosis_disease": diagnosis_disease,
            "is_first": is_first,
        }.items()
        if value is not None
    }
    if not dimension_values:
        raise HTTPException(status_code=422, detail="At least one dimension value is required.")
    try:
        strata = multidimensional_formula_strata(
            service,
            dimension_values,
            minimum_patients=minimum_patients,
            include_unspecified=True,
            process_type=process_type,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    stratum = next((item for item in strata["items"] if item["dimension_values"] == dimension_values), None)
    if stratum is None:
        raise HTTPException(status_code=404, detail="Multidimensional stratum not found.")
    if stratum["status"] != "eligible":
        return {
            "status": "insufficient_sample",
            "stratum": stratum,
            "minimum_patients": minimum_patients,
            "base_formula": None,
        }

    cache = getattr(request.app.state, "base_formula_cache", None)
    if cache is None:
        cache = request.app.state.base_formula_cache = {}
    cache_key = (
        id(service),
        "multidimensional",
        tuple(dimension_values.items()),
        process_type,
        minimum_support,
        maximum_pattern_length,
        pattern_limit,
        component_count,
        bootstrap_rounds,
    )
    if cache_key not in cache:
        try:
            cache[cache_key] = discover_base_formulas_by_dimensions(
                service,
                dimension_values=dimension_values,
                process_type=process_type,
                minimum_support=minimum_support,
                maximum_pattern_length=maximum_pattern_length,
                pattern_limit=pattern_limit,
                component_count=component_count,
                bootstrap_rounds=bootstrap_rounds,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
    return {"status": "ready", "stratum": stratum, "base_formula": cache[cache_key]}


@app.get("/api/base-formulas")
def base_formula_query(
    request: Request,
    disease: str,
    syndrome: str,
    minimum_patients: int = Query(30, ge=1, le=1000),
    minimum_support: float = Query(0.3, gt=0, le=1),
    maximum_pattern_length: int = Query(15, ge=1, le=25),
    pattern_limit: int = Query(30, ge=1, le=200),
    component_count: int = Query(3, ge=1, le=12),
    bootstrap_rounds: int = Query(500, ge=0, le=2000),
    visit_type: str | None = Query(None, pattern="^(初诊|复诊)$"),
    process_type: str | None = None,
    doctor: str | None = None,
) -> dict[str, object]:
    service = analysis(request, doctor)
    strata = formula_strata(
        service,
        minimum_patients=minimum_patients,
        include_unspecified_syndrome=True,
        visit_type=visit_type,
        process_type=process_type,
    )
    stratum = next(
        (item for item in strata["items"] if item["disease"] == disease and item["syndrome_value"] == syndrome),
        None,
    )
    if stratum is None:
        raise HTTPException(status_code=404, detail="Disease-syndrome stratum not found.")
    public_stratum = {key: value for key, value in stratum.items() if key != "syndrome_value"}
    if stratum["status"] != "eligible":
        return {
            "status": "insufficient_sample",
            "stratum": public_stratum,
            "minimum_patients": minimum_patients,
            "base_formula": None,
        }

    cache = getattr(request.app.state, "base_formula_cache", None)
    if cache is None:
        cache = request.app.state.base_formula_cache = {}
    cache_key = (
        id(service),
        disease,
        syndrome,
        visit_type,
        process_type,
        minimum_support,
        maximum_pattern_length,
        pattern_limit,
        component_count,
        bootstrap_rounds,
    )
    if cache_key not in cache:
        try:
            cache[cache_key] = discover_base_formulas(
                service,
                disease=disease,
                syndrome=syndrome,
                visit_type=visit_type,
                process_type=process_type,
                minimum_support=minimum_support,
                maximum_pattern_length=maximum_pattern_length,
                pattern_limit=pattern_limit,
                component_count=component_count,
                bootstrap_rounds=bootstrap_rounds,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
    return {"status": "ready", "stratum": public_stratum, "base_formula": cache[cache_key]}


@app.get("/api/diseases/{disease}")
def disease_detail(
    request: Request,
    disease: str,
    limit: int = Query(30, ge=1, le=100),
    doctor: str | None = None,
) -> dict[str, object]:
    try:
        return analysis(request, doctor).disease_detail(disease, limit=limit)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=f"Disease not found: {disease}") from error


@app.get("/api/diseases/{disease}/comparisons")
def disease_comparisons(
    request: Request,
    disease: str,
    case_limit: int = Query(8, ge=2, le=30),
    pair_limit: int = Query(8, ge=1, le=30),
    minimum_similarity: float = Query(0.8, ge=0.5, le=1.0),
    doctor: str | None = None,
) -> dict[str, object]:
    service = analysis(request, doctor)
    if not any(visit.disease == disease for visit in service.visits):
        raise HTTPException(status_code=404, detail=f"Disease not found: {disease}")
    return service.prescription_comparisons(
        disease=disease,
        case_limit=case_limit,
        pair_limit=pair_limit,
        minimum_similarity=minimum_similarity,
    )


@app.get("/api/revisits")
def revisits(
    request: Request,
    disease: str | None = None,
    outcome: str | None = None,
    symptom: str | None = None,
    limit: int = Query(100, ge=1, le=500),
    doctor: str | None = None,
) -> dict[str, object]:
    return analysis(request, doctor).revisits(disease=disease, outcome=outcome, symptom=symptom, limit=limit)


@app.get("/api/associations")
def associations(
    request: Request,
    disease: str | None = None,
    symptom: str | None = None,
    minimum_support: int = Query(3, ge=1, le=100),
    limit: int = Query(100, ge=1, le=500),
    doctor: str | None = None,
) -> dict[str, object]:
    return analysis(request, doctor).symptom_drug_associations(
        disease=disease, symptom=symptom, minimum_support=minimum_support, limit=limit
    )


@app.get("/api/patients")
def patients(
    request: Request,
    query: str = "",
    visit_type: str = Query("", pattern="^(|初诊|复诊)$"),
    limit: int | None = Query(None, ge=1, le=10000),
    doctor: str | None = None,
) -> dict[str, object]:
    return analysis(request, doctor).patients(query=query, visit_type=visit_type, limit=limit)


@app.get("/api/patients/{patient_id}")
def patient_timeline(request: Request, patient_id: str, doctor: str | None = None) -> dict[str, object]:
    try:
        return analysis(request, doctor).patient_timeline(patient_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=f"Patient ID not found: {patient_id}") from error


@app.get("/api/llm/status")
def llm_status(request: Request) -> dict[str, object]:
    return request.app.state.llm.status()


@app.post("/api/visits/{visit_id}/prescription-explanation")
def prescription_explanation(
    request: Request,
    visit_id: int,
    doctor: str | None = None,
    refresh: bool = False,
) -> dict[str, object]:
    service = analysis(request, doctor)
    try:
        case = service.visit_case(visit_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=f"Visit not found: {visit_id}") from error
    try:
        return request.app.state.llm.explain(service.source_name, case, refresh=refresh)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"LLM处方分析失败：{error}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-doctor medical analysis dashboard.")
    parser.add_argument("--data", type=Path, help="Only expose one consultation JSON file.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root containing doctor JSON files or online_* directories.",
    )
    parser.add_argument("--doctor", default=DEFAULT_DOCTOR, help="Default doctor ID or directory key.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    if args.data:
        os.environ["MEDICAL_ANALYSIS_DATA"] = str(args.data.resolve())
    else:
        os.environ.pop("MEDICAL_ANALYSIS_DATA", None)
        os.environ["MEDICAL_ANALYSIS_DATA_ROOT"] = str(args.data_root.resolve())
        os.environ["MEDICAL_ANALYSIS_DEFAULT_DOCTOR"] = args.doctor
    uvicorn.run("medical.analysis.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
