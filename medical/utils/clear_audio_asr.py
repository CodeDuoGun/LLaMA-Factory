"""
对医患对话进行清洗
1、如果对话人连续讲话，则需要合并本次所有asr结果
2、去掉所有说话人的头禅
3、去掉所有ASR错误
4、去掉所有重复词，只保留有效医患对话结果
5、去掉无意义语气词，嗯嗯嗯 啊啊啊啊 哦哦哦 好好好等
6、出现的报告名称、指标名称、药物名称、检查名称、治疗名称等，如果患者病历中有提及，严格和病历内容保持一致，纠正ASR错误
7、对于医生的每个问题，注意是问题而不是回答，要给出意图标签。可选的意图包括【症状采集、疗效评估、副作用评估、危险指标评估、中医辨证、生活方式、情绪状态、睡眠、二便、月经、舌象、既往治疗、检查报告】
"""
import json
import os
from openai import OpenAI
from tqdm import tqdm


def read_record_data(record_data_file:str="medical/data/AI医生分身问诊数据-吴卫平_20260525142743.json"):
    with open(record_data_file, "r", encoding="utf-8") as f:
        record_data = json.load(f)
    return record_data

record_data = read_record_data()


def get_record_info_by_id(record_id:str):
    record_info = {}
    for item in record_data:
        if str(item["id"]) == record_id:
            record_info = item
            break
    print(f"record_info: {record_info}")
        
    record_text = ",".join([f"{k}:{v}" for k, v in record_info.items() if k in ("is_first", "patient_sex", "patient_age", "patient_height", "patient_weight", "doc_ass_stu_appeal","new_medical_history")])
    return record_text, record_info.get("is_first")

def call_llm_api(data, record_info):
    """
    调用LLM API，对医患对话进行清洗,并返回清洗后的医患对话
    """
    intent_labels = """
1. 症状询问
说明: 询问患者当前的皮肤症状，如发红、发烫、瘙痒、疼痛、肿胀等主观感受
"发红发烫发痒的情况怎么样？疼痛情况怎么样？"
"夜烫几点到几点？"

2. 治疗方案指导
说明: 给出具体的用药、治疗、生活调理建议
"我就给你开个开个八副吧，吃四副，休息两天再吃四副，好吧？"
"一般我建议你米诺环素和中药隔开两小时以上。"

3. 诊断结果说明
说明: 给出或解释诊断结论、病情定性
"你这是玫瑰痤疮，这个病比较复杂。"
"你的月经量比较平稳，那就考虑血热血瘀症。"

4. 用药反馈询问
说明: 询问患者服用或使用药物后的反应、效果和异常情况
"内服外用药用了以后有什么异常反应？有没有？"
"那两副中药吃了有没有不舒服？"

5. 面部症状询问
说明: 指导患者调整摄像头或做特定面部动作以便观察
"给我一个正面的，让我看一下。正面拉进来，拉近点。"
"左右脸转转，让我看看。"

6. 中医四诊询问
说明: 询问舌头、面色、患者的饮食、睡眠、大小便等日常生活习惯
"吃饭大小便正常吗？"
"睡眠怎么样？大便干还是稀？"
"舌头我看看。"
"面色怎么样？"

7. 医嘱叮嘱
说明: 叮嘱患者注意事项、生活禁忌等
"尽可能情况下不要戴口罩。"
"不要吃生冷油腻的，少吃啊。"

8. 病史追溯
说明: 询问患者过去的病情、治疗史、过敏史等
"你当时是记得吃什么药引起来的？"
"过去一直是这么红吗？"

9. 病因病机分析
说明: 解释可能病因、病理机制或病情发展趋势
"因为你那个肝功能指标不正常，可能要保肝治疗。"
"这个病呢，它不是一种常见病，古代没这种病。"

10. 检查检验询问 
说明: 询问患者是否做过相关检查、化验或影像学检查
"你那个做过血尿常规没有？"
"第二次又做过一次肝功吗？"

11. 病情观察指导
说明: 指导患者如何观察和记录病情变化
"注意观察有什么问题，保持联系。"
"观察一下，看他情况怎么样。"

12. 复诊随访安排
说明: 安排下次复诊时间或告知联系方式
"那行，那我们保持联系，有什么问题随时联系。"
"那我这几天我还没去呢，我打算这两天再去。"

13. 情感关怀安慰 
说明: 对患者进行心理安慰和情绪疏导
"不要紧张，放松，没事的。"

14. 其他交流 
说明: 打招呼、结束语、过渡语等日常交流内容

"""
    prompt = f"""你是一位资深中医知识结构化专家，擅长从医患对话中提取结构化信息，用于构建中医智能诊断大模型。

## 任务
请对以下医患对话进行清洗和结构化处理，返回清洗后的医患对话内容。

## 清洗规则（必须严格遵守）

1 **去除头禅**：去掉所有说话人发言中的口头禅（如"那个"、"的话"、"就是"、"嗯"等）。
2. **纠正ASR错误**：根据上下文和病历信息，修正所有ASR（自动语音识别）导致的文字错误。
3. **去除重复词**：去掉所有重复的词句，仅保留有效医患对话内容。
4. **去除无意义语气词**：去掉"嗯嗯嗯"、"啊啊啊啊"、"哦哦哦"、"好好好"、以及各类无意义的语气词和停顿。
5. **统一专业术语**：出现的报告名称、指标名称、药物名称、检查名称、治疗名称等，如果患者病历中有提及，必须严格和病历内容保持一致，纠正ASR错误。
6. **标注医生问题意图**：对于医生的每个问题（非回答），在问题后附加且仅加一个意图标签。意图标签可选范围为：{intent_labels}。格式示例：医生：您最近睡眠怎么样？[意图: 睡眠]

## 患者病历信息
{record_info}

## 待处理医患对话
{data}

## 输出要求
- 仅输出清洗后的医患对话内容json格式，无需额外解释。
- 医生问题后必须标注意图标签。
- 保持对话的逻辑连贯性，不要遗漏关键诊疗信息。
"""
    client = OpenAI(api_key="", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",)
    content = ""
    response = client.chat.completions.create(
        model="qwen-plus", 
        messages=[{"role": "user", "content": prompt}], 
        stream=True, 
        temperature=0.1,
        # max_tokens=4096
    )
    for chunk in response:
        if chunk:
            content += chunk.choices[0].delta.content
    # 提取有效json结果
    try:
        res = json.loads(content)
    except Exception as e:
        print(f"提取有效json结果失败: {e} {content}")
        res = {}
    return res


def save_result(result, output_file):
    """每次保存前，先读取之前的保存结果，如果record_id 已经存在，则更新结果，否则新增 保存结果到文件 """
    try:
        with open(output_file, "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        results = []
    results.append(result)
    with open(output_file, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def merge_audio_json(audio_json):
    """合并连续的speakerid对话内容为一个"""
    if not audio_json:
        return []

    merged_audio_json = []
    current = None

    for item in audio_json:
        if current is None:
            current = {
                "content": item["content"],
                "speaker": item["speaker"],
                "begin_time": item["begin_time"],
                "end_time": item["end_time"]
            }
        elif item["speaker"] == current["speaker"]:
            current["content"] += item["content"]
            current["end_time"] = item["end_time"]
        else:
            merged_audio_json.append(current)
            current = {
                "content": item["content"],
                "speaker": item["speaker"],
                "begin_time": item["begin_time"],
                "end_time": item["end_time"]
            }

    if current is not None:
        merged_audio_json.append(current)

    return merged_audio_json

def clear_audio_asr(audio_asr_file, output_file):
    """
    对医患对话进行清洗
    """
    # 1. read jsonl file
    with open(audio_asr_file) as f:
        lines = f.readlines()
    total = len(lines)
    for line in tqdm(lines, desc="处理医患对话", total=total):
            data = json.loads(line)
            if not data["audio_file"]:
                continue
            try:
                record_id = data["audio_file"].split("/")[-1].split(".")[0].split("_")[0]
                print(f"record_id: {record_id}, type: {type(record_id)}")
                record_info, is_first = get_record_info_by_id(record_id)
                # 合并连续的speakerid对话内容为一个
                merged_audio_json = merge_audio_json(data["audio_json"])
                need_clear_data = {
                    "audio_file": data["audio_file"],
                    "is_valid": data["is_valid"], 
                    "history": merged_audio_json
                }

                cleared_data = call_llm_api(need_clear_data, record_info)
                result = {
                    "record_id": record_id,
                    "is_first": is_first,
                    "record_text": record_info,
                    "cleared_data": cleared_data
                }
                save_result(result, output_file)
            except Exception as e:
                print(f"医患对话清洗失败: {e} {record_id}")
                result = {
                    "record_id": record_id,
                    "is_first": is_first,
                    "record_text": record_info,
                    "cleared_data": ""
                }
                save_result(result, output_file)



if __name__ == "__main__":
    audio_asr_file = "medical/data/wuweiping问诊对话audio_labels.jsonl"
    os.makedirs("medical/processed_data", exist_ok=True)
    output_file = "medical/processed_data/cleared_audio_asr.jsonl"
    clear_audio_asr(audio_asr_file, output_file)
    print("医患对话清洗完成")