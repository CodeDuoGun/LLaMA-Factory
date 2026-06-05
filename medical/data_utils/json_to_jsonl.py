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

"""Convert a JSON array file into JSONL format."""

import json
from pathlib import Path
from typing import Any


INPUT_PATH = Path("medical/processed_data/postasr_speaker_content.json")
OUTPUT_PATH = Path("medical/processed_data/postasr_speaker_content.jsonl")


def main() -> None:
    """Read a JSON array and write one JSON object/array per line."""
    with INPUT_PATH.open("r", encoding="utf-8") as input_file:
        records: list[Any] = json.load(input_file)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Converted {len(records)} records -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
