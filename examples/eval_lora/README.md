# 医疗处方微调评估配置

## 评估流程

### 1. 训练完成后推理预测

使用 `examples/eval_lora/qwen3.5_lora_predict_prescription.yaml` 对验证集进行批量推理：

```bash
llamafactory-cli train examples/eval_lora/qwen3.5_lora_predict_prescription.yaml
```

推理结果会保存在 `saves/qwen3.5-9b/lora/predict/prescription/` 目录，包含模型生成的标准 JSON 输出。

### 2. 评估指标说明

该配置设置 `predict_with_generate: true`，会对 eval_dataset 中每条数据：
- 输入患者病史信息
- 让模型生成 `<think>` 思考链 + JSON 处方
- 将结果写入输出目录

### 3. 后续评估

当前配置仅输出生成结果。建议后续扩展：
- **自动评测**：对比生成 JSON 与标准答案的字段匹配率、处方相似度等
- **人工评测**：抽取样本由中医专家评估辨证准确性和处方合理性
- **BLEU/ROUGE**：与标准处方文本做 n-gram 重叠度对比

### 4. 调整说明

- `max_samples`: 默认评测全部数据，可减少以快速验证
- `per_device_eval_batch_size`: 评估 batch_size
- `output_dir`: 包含 `predict/prescription` 的完整路径，需与训练配置一致
