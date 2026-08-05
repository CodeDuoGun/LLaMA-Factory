# 医疗数据工具说明

`medical/data_utils` 包含原始数据清洗、AI 病历结构化、诊断标签归一化、数据库导入、音频处理和
SFT 数据构造工具。除非脚本另有说明，以下命令均在项目根目录执行。

## 推荐处理流程

```text
原始问诊/处方数据
  → raw_data_clear 数据清洗
  → medical_record_agents.py AI 结构化和诊断归一化
  → medical_records_ai.jsonl
  → import_ai_medical_records.py 导入 MySQL
  → analysis 病历标注页人工审核
```

数据库配置由 `config.py` 使用 Dynaconf 读取，优先使用环境变量或 `.env`，也可读取
`medical/data_utils/config.yaml`。建议通过环境变量设置连接，不要把真实密码写进提交的文档或脚本：

```bash
export DATABASE_URL='mysql+pymysql://user:password@host:port/database?charset=utf8mb4'
```

## AI 病历结构化

### `medical_record_agents.py`

主处理脚本。它按医生读取 JSON/JSONL，调用文本、视觉、诊断和可选评审 Agent，生成带 `ai_` 前缀的
结构化病历，结果按医生增量写入：

```text
medical/processed_data/doctor_<doctor_id>_<doctor_name>/medical_records_ai.jsonl
```

它支持图片分类、舌面图解析、检查报告解析、主诉和现病史整理、既往史抽取、诊断补全、处方相关字段
处理、失败重试和断点跳过。同一患者按就诊时间串行处理，不同患者可由 `--batch-size` 并发处理。

单医生示例：

```bash
python medical/data_utils/medical_record_agents.py \
  --input medical/data/202301_online/朱子奇_AI医生分身混合问诊数据_43_20260729175534.json \
  --output-dir medical/processed_data \
  --doctor-id 43 \
  --batch-size 8 \
  --progress-every 10
```

常用参数：

| 参数 | 作用 |
|---|---|
| `--doctor-id` | 只处理指定医生；可一次传入多个 ID，也可重复传入 |
| `--limit` | 限制处理条数，`0` 表示不限制 |
| `--batch-size` | 最大并发病历数 |
| `--timeout` / `--max-retries` | 单次模型请求超时和重试次数 |
| `--record-timeout` | 单条病历全部阶段总超时，`0` 表示不额外限制 |
| `--stage-name` | 只运行指定阶段；可在一个参数后传多个阶段名，也可重复传参 |
| `--enable-review` | 启用评审 Agent |
| `--reprocess` | 忽略已有成功结果并重新处理 |
| `--reprocess-failures` | 只重跑失败记录 |
| `--reprocess-missing-histories` | 重跑主诉或现病史原始值与 AI 值同时为空的记录 |

只清洗病史和诊断的示例：

```bash
python medical/data_utils/medical_record_agents.py \
  --input <input.jsonl> \
  --output-dir medical/processed_data \
  --stage-name clinical_cleaning diagnosis_completion
```

可选阶段为 `diagnosis_normalization`、`image_classification`、`inspection_vlm`、`tongue_face_vlm`、
`current_visit_history`、`clinical_cleaning`、`diagnosis_completion`、`clinical_extraction` 和
`treatment_principle`。不传 `--stage-name` 时保持原有行为并运行全部阶段；传入阶段后，输出文件名会记录
阶段组合，例如 `medical_records_ai__stage_clinical_cleaning__diagnosis_completion.jsonl`，记录内同时写入
`ai_processing_stages`。选择 `inspection_vlm` 或 `tongue_face_vlm` 时会自动加入其必要依赖
`image_classification`，文件名和记录中的阶段列表均以实际执行阶段为准。

`clinical_cleaning` 会将 `admin_face_describe`（舌象及面相）、`birth_detail`（生育史）、
`is_marriage_history`（婚恋史）及三个原始诊断字段一并提供给模型，输出主诉、现病史一致性、五史和三个
AI 诊断及其修正理由。

诊断标签归一化已经集成在此主流程中。模型结果先经过代码内别名表，再经过 Excel 标准词表：

| AI 字段 | 第一层归一化 | 第二层标准词表 |
|---|---|---|
| `ai_diagnosis_illness` | `DIAGNOSIS_TERM_ALIASES` | `medical/data/ICD10.xlsx` |
| `ai_diagnosis_disease` | `SYNDROME_TERM_ALIASES` | `medical/data/辩证.xlsx` |
| `ai_diagnosis_sickness` | `SICKNESS_TERM_ALIASES` | `medical/data/辨病.xlsx` |

Excel 映射在单个进程内缓存，归一化只替换名称，暂不向 AI 字段写入编码。

字段级最终标准名规则优先于 Excel 中的旧名称：西医诊断 `酒渣鼻` / `玫瑰痤疮` 统一为
`玫瑰痤疮`；中医疾病 `酒槽鼻` / `酒齄鼻` 统一为 `酒齄鼻`。

病史清洗会把同一患者的上一诊时间、主诉和现病史传入下一次复诊。主诉只允许参考本次患者/医生主诉、
上一诊现病史和上一诊主诉，不补造病程时间；月经史统一从个人史、特殊史移动到现病史；检查报告按
“上一次就诊—本次就诊”的时间区间区分本次检查与既往检查。

### `run_six_doctors.sh`

为吴卫平、巢国俊、朱子奇、李同新、杨钦河、郝建军各启动一个后台处理进程，并保存 PID 和日志：

```bash
bash medical/data_utils/run_six_doctors.sh
```

可通过环境变量覆盖运行参数：

```bash
CONDA_ENV=llama_factory BATCH_SIZE=2 LIMIT_PER_DOCTOR=4000 \
  bash medical/data_utils/run_six_doctors.sh
```

日志默认位于 `medical/processed_data/logs/six_doctors/`，PID 文件为该目录下的 `workers.pid`。

### `run_failed_six_doctors.sh`

使用同一批六位医生配置，重跑失败记录以及主诉/现病史缺失记录：

```bash
bash medical/data_utils/run_failed_six_doctors.sh
```

## 诊断标签离线归一化

### `normalize_ai_diagnosis_labels.py`

用于归一化已经生成的 JSONL，适合历史数据补处理或检查词表覆盖率。字段和词表关系与主流程相同。
`辨病.xlsx`、`辩证.xlsx` 中同一编码的第一条名称视为标准名，后续名称视为别名；`ICD10.xlsx` 中的诊断名称
按标准名称匹配。存在歧义或没有匹配的名称会保留原值，并写入审计报告。

```bash
python medical/data_utils/normalize_ai_diagnosis_labels.py \
  --input medical/processed_data/doctor_43_朱子奇/medical_records_ai.jsonl
```

默认产生两个文件：

- `medical_records_ai_normalized.jsonl`：归一化后的完整病历数据，可继续导入或分析。
- `medical_records_ai_normalized.jsonl.audit.json`：本次映射统计、变更数量、未匹配标签和歧义项，只用于审计。

常用模式：

```bash
# 只统计，不写归一化 JSONL
python medical/data_utils/normalize_ai_diagnosis_labels.py --input <input.jsonl> --check-only

# 原地更新；执行前会创建不可覆盖的备份
python medical/data_utils/normalize_ai_diagnosis_labels.py --input <input.jsonl> --in-place

# 允许覆盖指定输出文件
python medical/data_utils/normalize_ai_diagnosis_labels.py --input <input.jsonl> --overwrite
```

内部函数 `normalize_label_value(old_value, mapping)` 返回：归一化后的值、成功匹配的标签出现次数、未匹配
标签列表。这里的匹配数是标签数，不是病历数。

### `normalize_ai_diagnosis_six_doctors.sh`

并行处理上述六位医生的 `medical_records_ai.jsonl`：

```bash
bash medical/data_utils/normalize_ai_diagnosis_six_doctors.sh
CHECK_ONLY=1 bash medical/data_utils/normalize_ai_diagnosis_six_doctors.sh
OVERWRITE=1 bash medical/data_utils/normalize_ai_diagnosis_six_doctors.sh
```

可使用 `PYTHON_BIN`、`PROCESSED_DATA_DIR` 和 `LOG_DIR` 覆盖默认环境。该脚本不会修改源 JSONL，除非单独
使用单文件脚本的 `--in-place`。

## MySQL 数据读写

从 MySQL 导出时可用 `--doctor-id` 指定一个或多个医生。查询会使用 `doctor_id IN (...)` 过滤，结果按医生
分别写入 `doctor_<doctor_id>_<doctor_name>/records.jsonl`：

```bash
python medical/data_utils/medical_record_agents.py \
  --data-source mysql \
  --export-mode \
  --output-dir medical/processed_data \
  --doctor-id 43 52 1314
```

也可以重复传参，例如 `--doctor-id 43 --doctor-id 52`。不传 `--doctor-id` 时导出表内全部医生。

| 文件 | 作用 |
|---|---|
| `config.py` | 加载 `.env`、环境变量和 `config.yaml` |
| `db_manager.py` | SQLAlchemy 连接池、事务和通用数据库操作 |
| `ai_medical_records.py` | 标注表结构、JSONL 字段拆分和病历仓储操作 |
| `import_ai_medical_records.py` | 批量建表/导入，以及按患者、问诊单、病历查询 |
| `import_one_ai_medical_record.py` | 预览或写入单条数据，适合先验证字段转换和数据库连接 |

批量导入和查询示例：

```bash
python medical/data_utils/import_ai_medical_records.py import --input <medical_records_ai.jsonl> --batch-size 100
python medical/data_utils/import_ai_medical_records.py patient 12345
python medical/data_utils/import_ai_medical_records.py inquiry <order_sn>
python medical/data_utils/import_ai_medical_records.py record 1001
```

已有相同 `order_sn` 的记录会跳过。清空整表属于破坏性操作，必须显式确认：

```bash
python medical/data_utils/import_ai_medical_records.py clear --yes
```

写入前可先预览一条：

```bash
python medical/data_utils/import_one_ai_medical_record.py --input <medical_records_ai.jsonl> --dry-run
python medical/data_utils/import_one_ai_medical_record.py --input <medical_records_ai.jsonl> --line-number 1
```

## 格式化与基础组件

| 文件 | 作用 |
|---|---|
| `medical_record_formatters.py` | 把舌面图、检查报告等结构化结果转换为前端和提示词使用的可读文本 |
| `structured_agent.py` | 约束模型输出结构、解析结果并处理调用重试的通用 Agent 基础组件 |
| `knowledge.py` | 病历处理中的中医知识检索与上下文辅助 |
| `convert_record_ps2json.py` | 旧数据中处方字段的 JSON 化转换，运行前应核对脚本内输入输出配置 |

## 原始数据和处方数据处理

| 文件 | 作用 |
|---|---|
| `raw_data_clear/clear_xiqu_chaoguojun_record.py` | 清洗西区巢国俊病历，并合并处方 Excel |
| `raw_data_clear/clear_zongyuan_records.py` | 清洗总院病历并合并收费/处方数据，可选 LLM 辅助 |
| `raw_data_clear/rebuild_internal_prescriptions.py` | 仅根据收费表重建内服处方，保留其他病历字段并排除退费项 |
| `raw_data_clear/clear_postasr_to_traindata.py` | 将后处理 ASR 对话清洗为训练数据 |
| `raw_data_clear/invalid_clear_multichat_asr.py` | 早期多轮 ASR 数据专用脚本，使用前需检查脚本内数据路径和常量 |
| `export_missdata_records.py` | 导出关键字段缺失的病历到 Excel，便于人工排查 |
| `prepare_postasr_prompt_context.py` | 为 ASR 后处理准备病历提示上下文 |
| `process_prescription_template.py` | 将处方模板 Excel 整理为 JSON |
| `convert_template_to_txt.py` | 将处方模板 JSON 转为便于检查的文本 |
| `pack_es_prescription.py` | 打包 Elasticsearch 使用的处方 JSONL，可组合症状和处方模板信息 |

这些脚本的数据源格式差异较大。首次运行前先执行 `python <script> --help`，并检查默认输入、输出路径；
对没有命令行参数的历史脚本，应先修改其顶部配置为当前数据路径。

## 音频与对话数据

| 文件 | 作用 |
|---|---|
| `audio/extract_audio.py` | 从问诊视频提取音频，衔接对象存储、ASR 和数据库记录 |
| `audio/asr_to_dialog.py` | 将 ASR 句子按角色和上下文合并为对话轮次 |
| `audio/clean_medical_dialogues_with_llm.py` | 使用 LLM 清洗医疗对话文本 |
| `audio/convert_medical_dialogue_to_sft.py` | 将清洗后的对话拆分为训练/验证/测试 SFT 数据 |
| `audio/convert_postclear_to_sharegpt.py` | 将后清洗数据转换为 ShareGPT 格式 |
| `audio/image_integrity.py` | 图片完整性检查的公共函数 |
| `audio/quarantine_incomplete_images.py` | 隔离图片不完整的数据，避免进入后续训练流程 |
| `audio/audio_db_schema.py`、`audio/db.py` | 音频处理相关表结构和数据库访问 |
| `audio/generate_emergency_sft_with_llm.py` | 生成急症场景 SFT 样本 |
| `audio/del_intent.py` | 清理对话中的意图标记 |

## SFT 数据与实验分析

| 文件 | 作用 |
|---|---|
| `sft_gen_prescription/extract_valid_record.py` | 筛选可用于处方任务的有效病历 |
| `sft_gen_prescription/generate_prescription_sft_dataset.py` | 结合病历和知识图谱生成处方 SFT 数据，运行前需准备 Neo4j 配置 |
| `analysis_v1/record_prescription_analyaze.py` | 病历和处方的早期统计分析 |
| `analysis_v1/analyze_prescription_strategy.py` | 分析处方策略和用药模式 |
| `analysis_v1/analyze_template_prescription.py` | 分析结构化处方模板匹配结果 |
| `analysis_v1/analyze_template_prescription_txt.py` | 输出处方模板分析的文本版本 |

`analysis_v1` 属于数据研究脚本，部分路径和参数与具体批次绑定；正式运行前需先检查文件顶部配置。

## 开方 RAG Agent 与评测

| 文件 | 作用 |
|---|---|
| `prescription_rag_agents.py` | 定义结构化辨病 Agent 和开方 Agent；开方结果要求逐药提供召回病例证据并支持证据不足拒答 |
| `evaluate_prescription_rag.py` | 按医生运行患者级隔离的评测数据，连接 ES 检索并输出完整指标和逐病例明细 |

```bash
python medical/data_utils/evaluate_prescription_rag.py \
  --doctor-id 43 --doctor-name 朱子奇 \
  --index alpha_medical_prescription_rag_v1 \
  --limit 20 --top-k 10
```

评测数据默认位于 `medical/eval_data/doctor_<id>_<姓名>/prescription_rag_eval.jsonl`，完整格式见
`medical/eval_data/README.md`。`--limit 0` 表示评测全部 `split=test` 样本。

## 测试

```bash
PYTHONPATH=. WANDB_DISABLED=true pytest -q --import-mode=importlib \
  medical/data_utils/test_ai_medical_records.py \
  medical/data_utils/test_medical_record_agents.py \
  medical/data_utils/test_normalize_ai_diagnosis_labels.py \
  medical/data_utils/test_structured_agent.py
```

数据库连通性测试会访问实际数据库，应在确认当前环境的 `DATABASE_URL` 指向测试库后再单独运行
`medical/data_utils/test_db.py`。
