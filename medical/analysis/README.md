# 问诊处方纵向分析

本目录提供一套可直接运行的多医生前后端实现，用于分析在线医生问诊数据中的：

- 疾病、病情和内服处方的关系；
- 按 `diagnosis_illness` 原始值统计疾病核心用药、病种内使用率及相对全体处方的 Lift；
- 同病不同处方、相似处方不同疾病的病例对照；
- 基于病例病情和处方的 LLM 开方原因、配伍逻辑解释；
- 相邻复诊的保留药、新增药、停用药和剂量增减；
- 当前症状与新增药之间的探索性关联；
- 完全脱敏的患者纵向时间轴。

分析系统向浏览器返回原始 `patient_id`，用于患者搜索和时间轴展示；不会返回姓名、身份证号或手机号。

## 启动

在项目根目录执行：

```bash
python -m medical.analysis.app
```

然后访问 <http://127.0.0.1:8008>。系统自动发现以下目录中的所有 JSON 文件：

```text
medical/data/online_*/
```

每个 `online_<doctor_key>` 目录代表一位医生。前端医生选择器会自动显示已发现的医生；同一目录下的后续补充
JSON 会自动合并，并按问诊记录 `id` 去重。例如李同新的后续文件只需放入：

```text
medical/data/online_litongxin/
```

也可以指定其他同结构数据：

```bash
python -m medical.analysis.app --data /absolute/path/to/records.json --port 8008
```

指定数据根目录或默认医生：

```bash
python -m medical.analysis.app --data-root medical/data --doctor zhuziqi
```

或在部署时设置环境变量：

```bash
MEDICAL_ANALYSIS_DATA=/absolute/path/to/records.json uvicorn medical.analysis.app:app --host 0.0.0.0 --port 8008
```

## API

| 地址 | 用途 |
|---|---|
| `GET /api/doctors` | 已发现医生及数据文件信息 |
| `GET /api/overview` | 数据规模、质量、复诊总览 |
| `GET /api/metadata` | 症状和疗效反馈枚举 |
| `GET /api/diseases` | 标准疾病分组 |
| `GET /api/diseases/{disease}` | 病种核心药、症状和调方关联 |
| `GET /api/diseases/{disease}/comparisons` | 同病异方、相似方异病病例 |
| `GET /api/base-formulas/strata` | 疾病—证候组合、独立患者数和可挖掘状态 |
| `GET /api/base-formulas` | 按疾病和证候查询患者加权的候选基础方 |
| `GET /api/revisits` | 可按疾病、症状、反馈筛选的复诊变化对 |
| `GET /api/associations` | 症状—新增药探索性 Lift |
| `GET /api/patients` | 带 `patient_id` 的患者列表 |
| `GET /api/patients/{patient_id}` | 按 `patient_id` 查询患者时间轴 |
| `GET /api/llm/status` | LLM 配置状态 |
| `POST /api/visits/{visit_id}/prescription-explanation` | 生成处方原因与配伍逻辑 |
| `GET /docs` | FastAPI 自动接口文档 |

`/api/revisits` 示例：

```text
/api/revisits?doctor=zhangxiaoyu&disease=胰腺癌&symptom=便秘&outcome=改善
```

除 `/api/doctors` 外，所有分析接口都接受 `doctor=<doctor_key>` 参数。

## 统计口径

1. 疾病严格使用 `diagnosis_illness` 原值分组，不进行器官、病理类型或同义词类别合并。
2. 症状从主诉、医生摘要和现病史中抽取，并处理近距离否定词。
3. 疗效反馈分为改善、稳定、加重和不明确；这是文本标签，不代表临床疗效判定。
4. 患者时间轴的“本次症状与病情”直接使用同一问诊记录的 `new_medical_history`，并与该记录的就诊日期绑定。
5. 患者时间轴的“本次处方明细”展示当前节点主方的药物名称、剂量和单位。
6. 单次问诊有多张内服处方时，选择药味数最多的完整主方进行前后比较，避免直接使用 `ps[0]`。
7. 处方通过稳定的 `drug_id` 对齐；只有剂量单位相同时才比较剂量变化。
8. 症状—新增药关联使用当前症状组和无该症状组的新增率之比，默认支持度至少为 3 个复诊对。
9. 所有关系均为单医生观察数据中的模式和相关性，不能解释为药物疗效或因果关系。

## LLM 配置

处方解释仅在用户点击病例的“分析处方”后调用，不会在页面加载时批量调用。优先使用以下环境变量：

```bash
export MEDICAL_ANALYSIS_LLM_PROVIDER=doubao
export MEDICAL_ANALYSIS_LLM_API_KEY=your-key
export MEDICAL_ANALYSIS_LLM_BASE_URL=https://example.com/v1
export MEDICAL_ANALYSIS_LLM_MODEL=your-model
```

未设置环境变量时，会按 `medical/config.yaml` 中的 `LLM_PROVIDER` 尝试读取对应配置。发送给 LLM 的内容包括：
性别、年龄、`new_medical_history`、过敏史、家族史、个人史、既往史、`diagnosis_illness` 和本次处方；不会发送
患者姓名、手机号、身份证号或 `patient_id`。同一医生同一问诊的分析结果会在当前服务进程内缓存。

## 测试

```bash
WANDB_DISABLED=true pytest -q --import-mode=importlib medical/analysis/tests
```

## 患者加权的基方挖掘

`base_formula` 对指定疾病和证型执行患者加权 FP-Growth、患者级 NMF、药物共现网络和患者级
Bootstrap。结果只包含聚合统计，不输出姓名、证件号、手机号、患者 ID 或问诊 ID。例如：

```bash
python -m medical.analysis.base_formula \
  medical/data/online_zhuziqi/朱子奇_AI医生分身混合问诊数据_43_20260721114115.json \
  --disease 慢性萎缩性胃炎 \
  --syndrome 肝胃不和证 \
  --min-support 0.3 \
  --max-pattern-length 20 \
  --components 3 \
  --bootstrap 500 \
  --output /tmp/zhuziqi_base_formula.json
```

自动遍历所有具有结构化 `diagnosis_disease` 证候的组合并生成目录：

```bash
python -m medical.analysis.base_formula \
  medical/data/online_zhuziqi/朱子奇_AI医生分身混合问诊数据_43_20260721114115.json \
  --all-strata \
  --min-patients 30 \
  --min-support 0.3 \
  --max-pattern-length 20 \
  --components 3 \
  --bootstrap 500 \
  --output medical/analysis/output/朱子奇_疾病证候基础方目录.json
```

目录会列出全部组合；只有达到 `--min-patients` 的组合才生成 `base_formula`，其余组合标记为
`insufficient_sample`。默认不把空证候纳入目录，可通过 `--include-unspecified-syndrome` 单独列出。

API 查询示例：

```text
GET /api/base-formulas/strata?doctor=zhuziqi&minimum_patients=30
GET /api/base-formulas?doctor=zhuziqi&disease=慢性萎缩性胃炎&syndrome=肝胃不和证
```

相同医生、疾病、证候和算法参数的 API 结果会在服务进程内缓存。低于最低患者数时接口返回
`status=insufficient_sample`，不会生成候选基础方；不存在的组合返回 HTTP 404。

同一患者在筛选范围内有 `n` 次问诊时，每次问诊权重为 `1/n`，所以每位患者总权重均为 1。
饮片和颗粒剂可通过 `--process-type` 分开分析，初诊和复诊可通过 `--visit-type` 分开分析。
若 `mining_summary.search_was_capped` 为 `true`，说明频繁项集碰到了搜索药味数上限，应在计算资源
允许时提高 `--max-pattern-length` 后复核；上限过大可能造成频繁项集组合爆炸。
输出属于单医生观察数据中的经验模式，不代表临床疗效或因果关系，也不能直接用于自动开方。
