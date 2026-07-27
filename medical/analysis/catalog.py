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

"""Discover and lazily load one or more consultation datasets per doctor."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .engine import MedicalAnalysis


DATASET_DIRECTORY_PREFIXES = ("online_", "zongyuan_")
DOCTOR_NAME_ALIASES = {"wuweiping": "吴卫平"}


@dataclass
class DoctorSource:
    key: str
    name: str
    doctor_id: str
    files: list[Path]
    analysis: MedicalAnalysis | None = field(default=None, repr=False)
    duplicate_records: int = 0


class DoctorCatalog:
    """Dataset registry grouped by supported ``medical/data/<source>_<doctor>`` directories."""

    def __init__(self, sources: list[DoctorSource], default_doctor: str | None = None) -> None:
        if not sources:
            raise ValueError("No online doctor datasets were found.")
        self.sources = {source.key: source for source in sources}
        self.default_doctor = default_doctor if default_doctor in self.sources else sources[0].key

    @classmethod
    def discover(
        cls,
        data_root: str | Path,
        default_doctor: str | None = None,
        explicit_file: str | Path | None = None,
    ) -> DoctorCatalog:
        if explicit_file:
            path = Path(explicit_file).expanduser().resolve()
            source = cls._source_from_files(path.parent, [path])
            return cls([source], default_doctor=source.key)

        root = Path(data_root).expanduser().resolve()
        sources = []
        for directory in sorted(root.iterdir()):
            if not directory.is_dir() or not directory.name.startswith(DATASET_DIRECTORY_PREFIXES):
                continue
            pattern = "*_record_????????.json" if directory.name.startswith("zongyuan_") else "*.json"
            files = sorted(file for file in directory.glob(pattern) if file.is_file())
            if files:
                sources.append(cls._source_from_files(directory, files))
        flat_sources: dict[str, DoctorSource] = {}
        for file in sorted(root.glob("*.json")):
            if not file.is_file() or file.name.startswith(("._", ".")):
                continue
            source = cls._source_from_files(root, [file])
            source.key = source.doctor_id or source.name or file.stem
            if source.key in flat_sources:
                flat_sources[source.key].files.append(file)
            else:
                flat_sources[source.key] = source
        sources.extend(flat_sources.values())
        return cls(sources, default_doctor=default_doctor)

    @staticmethod
    def _source_from_files(directory: Path, files: list[Path]) -> DoctorSource:
        key = directory.name
        for prefix in DATASET_DIRECTORY_PREFIXES:
            if key.startswith(prefix):
                key = key.removeprefix(prefix)
                break
        metadata_file = next((file for file in files if "AI医生" in file.name), files[0])
        filename = metadata_file.stem
        name = DOCTOR_NAME_ALIASES.get(key, filename.split("_", 1)[0].strip() or key)
        doctor_id_match = re.search(r"_(\d+)_\d{14}$", filename)
        doctor_id = doctor_id_match.group(1) if doctor_id_match else ""
        return DoctorSource(key=key, name=name, doctor_id=doctor_id, files=files)

    def list_doctors(self) -> dict[str, object]:
        items = []
        for source in self.sources.values():
            items.append(
                {
                    "key": source.key,
                    "doctor_name": source.name,
                    "doctor_id": source.doctor_id,
                    "file_count": len(source.files),
                    "record_count": source.analysis.raw_record_count if source.analysis else None,
                    "loaded": source.analysis is not None,
                    "duplicate_records": source.duplicate_records,
                }
            )
        return {"default_doctor": self.default_doctor, "items": items}

    def get(self, doctor: str | None = None) -> MedicalAnalysis:
        key = doctor or self.default_doctor
        source = self.sources.get(key)
        if source is None:
            raise KeyError(key)
        if source.analysis is None:
            records_by_id = {}
            records_without_id = []
            doctor_names = []
            doctor_ids = []
            total_records = 0
            for path in source.files:
                with path.open("r", encoding="utf-8") as file:
                    records = json.load(file)
                if not isinstance(records, list):
                    raise ValueError(f"The input JSON must contain a list: {path}")
                total_records += len(records)
                for record in records:
                    record_id = record.get("id")
                    if record_id in (None, ""):
                        records_without_id.append(record)
                    else:
                        records_by_id[str(record_id)] = record
                    if record.get("doctor_name"):
                        doctor_names.append(str(record["doctor_name"]).strip())
                    if record.get("doctor_id") not in (None, ""):
                        doctor_ids.append(str(record["doctor_id"]).strip())
            records = list(records_by_id.values()) + records_without_id
            source.duplicate_records = total_records - len(records)
            if doctor_names:
                source.name = max(set(doctor_names), key=doctor_names.count)
            if doctor_ids:
                source.doctor_id = max(set(doctor_ids), key=doctor_ids.count)
            source.analysis = MedicalAnalysis(
                records,
                source_name=source.key,
                doctor_name=source.name,
                doctor_id=source.doctor_id,
            )
        return source.analysis
