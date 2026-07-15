asr 识别错误数据id：587、255、455、825（日语）、858无asr结果、 1040无asr、1043无asr、1057无asr、
1095无asr、1124无asr、1158无asr、1179无asr、1180无asr、1239无asr、1256、1286、1294、1300、


无效对话数据id: 
(330，331), done
358、done 删除
（392、393，394）done
410（检查报告名称）、done
（413，414，415三个重复数据）、done
(435、437) done
（440、441一次问诊两次通话，需要合并） done
540、done
（553、554、555、556、557一次问诊多次通话）、 done
（559、560一次问诊）、done
（563、564一次问诊）、done
（570、571、572一次问诊）、729127， done, 删除重复
（586、587、一次问诊）、done
（590、591一次问诊）、done
（608、609一次问诊）、done
（623、624一次问诊）、done
（628、629、630一次问诊）、done
（632 无数据, 633）、done
（637、638一次问诊）、done
（641、642、643一次问诊）、done
（645、646一次）、 done
（647、648、649一次）done
（673、674）一次问诊、done
（684、685一次）、done
（686、687一次）、done
（689、690一次）、done
（712、713一次）、done
（730、731一次）、done
（743、744一次）、done
（754、755、756一次）、done
（772、773一次）、done
（850、851一次）、done
（864、865一次）、done
（907、908、909一次）done
（920\921、922一次）、done
（925、926一次）、done
（952、953、954、955一次）、done
（970、971一次）、done
（ 981、982、983一次）、done
（984、985一次）、done
（987、988一次）、done
（1003、1004一次）、done
（1014、1015一次）done
（1020、1021一次）、done 
（1029、1030、1031一次）done
（1068、1069一次）、done
（1075、1076一次）、done
（1125、1126一次）、done
（1139、1140一次）、done
（1142,1143,1144、1145、1146一次）、done
（1189、1190、1191一次）、done
（1198、1199一次）、done
（1211、1212一次）、done
（1250、1251）、done
（1254、1255）、done
（1278、1279、1280）、done
（1287、1288）、done
（1293、1294、1295、1296一次）、
（1297、1298）、done


复杂对话id:
528(医生会问问诊外的闲聊话题)、548（可能有换患者情况存在）、（950前面废话太多）、（1023开头废话）、1055开头废话
1257开头废话、



清洗：
1、对话内容为空的 714951\
2、intent输出
3、speaker name

二次清洗：
1、医患对话一问一答、偶数、对话合并 进度到（ 743524 ）
4、

# sft train
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml 

test tcm branch

# 中医问诊开方模型 微调数据集格式以及比例

# 辩证开方 sft
raw: 
# 多轮问诊 sft
raw:
processed:
sft:

# 通用数据 sft

# 模板方加减任务 sft

# 知识图谱检索
python -m medical.data_utils.knowledge herb-knowledge 黄芩 连翘
python -m medical.data_utils.knowledge symptom-pathogenesis 口苦 苔黄腻
python -m medical.data_utils.knowledge disease-prescription-knowledge 痤疮
python -m medical.data_utils.knowledge similar-prescriptions --disease 痤疮 --syndrome 肺胃湿热 --symptom 口苦
python -m medical.data_utils.knowledge sft-context --disease 痤疮 --syndrome 肺胃湿热 --herb 黄芩 --herb 连翘


# train
```bash
    # 先 运行命令 “” 生成 train_dataset: 
    1.traindataset: dataset_info.json wuweiping_prescription
    2.run command: 
```

# eval
```bash
    # 先 运行命令 “” 生成 val_dataset: 
    1.evaldataset: dataset_info.json wuweiping_prescription_test_20260714
    2.run command: 
```


# 多模态模型微调

## 问诊对话数据集 【文本/图片】
1.文本：多轮对话
2.图文：看脸/看舌/看患处

## 训练
1. 

## 验证

## 还需额外 一批数据 
不能确诊
不能开处方
遇到危险信号转线下/急诊
儿童孕妇肝肾异常先问禁忌

# SP
你是一个线上预问诊助手，只做病史采集、风险提示和就医建议。
你不能做最终诊断，不能替代线下医生，不能开具处方。
遇到胸痛、呼吸困难、意识障碍、高热不退、严重过敏等情况，应建议立即就医。
回答要简洁、温和、先追问关键病史，再给风险提示。