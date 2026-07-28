from typing import Any

from pydantic import BaseModel, Field, field_validator


class DiagnosisResult(BaseModel):
    """三个诊断字段重新分类及补全后的结果."""

    diagnosis_illness: str = Field(..., min_length=1, description="西医疾病诊断；不能为空或占位词")
    diagnosis_illness_reason: str = Field(default="", description="西医诊断的来源、重新分类或补全依据")
    diagnosis_disease: str = Field(..., min_length=1, description="中医证型/证候诊断；不能为空或占位词")
    diagnosis_disease_reason: str = Field(default="", description="中医证候的来源、重新分类或补全依据")
    diagnosis_sickness: str = Field(..., min_length=1, description="中医疾病诊断；不能为空或占位词")
    diagnosis_sickness_reason: str = Field(default="", description="中医疾病的来源、重新分类或补全依据")

    @field_validator("diagnosis_illness", "diagnosis_disease", "diagnosis_sickness", mode="before")
    @classmethod
    def require_diagnosis_value(cls, value: Any) -> str:
        text = "" if value is None else str(value).strip()
        if text.lower() in {"", "null", "none", "无", "未知", "未明确", "不详", "待查", "暂无", "正常"}:
            raise ValueError("诊断结果不能为空或占位词")
        return text


class HistoryCleaningResult(BaseModel):
    """主诉、现病史一致性校验及五史清洗结果."""

    patient_appeal: str = Field(default="", description="清洗后的主诉")
    new_medical_history: str = Field(default="", description="经一致性校验和清洗后的本次现病史")
    complaint_history_consistent: bool | None = Field(default=None, description="主诉与本次现病史是否一致")
    consistency_issues: list[str] = Field(default_factory=list, description="不一致点；无不一致时为空")
    old_medical_history: str = Field(default="", description="有效既往史")
    allergic_history: str = Field(default="", description="有效过敏史")
    personal_history: str = Field(default="", description="有效个人史")
    special_history: str = Field(default="", description="有效专科史、婚育史或月经史")
    family_history: str = Field(default="", description="有效家族史")

    @field_validator("complaint_history_consistent", mode="before")
    @classmethod
    def normalize_consistency_value(cls, value: Any) -> bool | None | Any:
        """兼容模型对可空布尔字段的常见文本输出."""
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"", "null", "none", "未知", "不确定", "无法判断", "未涉及", "资料不足"}:
                return None
            if normalized in {"true", "是", "一致", "相符", "符合"}:
                return True
            if normalized in {"false", "否", "不一致", "不相符", "不符合"}:
                return False
        return value


class CurrentVisitHistoryResult(BaseModel):
    """本次复诊现病史提取结果."""

    new_medical_history: str = Field(
        default="",
        description="仅包含本次就诊相关内容、符合现病史书写规范的现病史",
    )


class InspectionFinding(BaseModel):
    """单张检查报告图片的分类和解析结果."""

    image_type: str = Field(default="", serialization_alias="图片类别", description="预定义的图片类别")
    is_valid_report: bool = Field(default=False, serialization_alias="是否有效检查报告")
    report_name: str = Field(default="", description="报告或检查项目名称")
    report_date: str = Field(default="", serialization_alias="报告日期")
    content: str = Field(default="", serialization_alias="解析结果")
    abnormal_indicators: list[str] = Field(default_factory=list, serialization_alias="异常指标")
    conclusion: str = Field(default="", serialization_alias="报告结论")


class InspectionResult(BaseModel):
    """一组检查报告图片的分类和有效信息汇总."""

    reports: list[InspectionFinding] = Field(default_factory=list)
    history_evidence_summary: str = Field(default="", serialization_alias="现病史检查证据摘要")


class TongueBodyFeatures(BaseModel):
    """舌质特征."""

    color: str = Field(default="", description="淡红、淡白、淡黄、黄、红、绛、青紫、蓝、黑之一")
    type: str = Field(default="", description="正常、胖大、瘦薄、裂纹、齿痕、芒刺、老嫩，可用顿号组合")


class TongueCoatFeatures(BaseModel):
    """舌苔特征."""

    color: str = Field(default="", description="白、淡黄、黄、灰、黑、霉酱、绿、染之一")
    type: str = Field(default="", description="匀、厚、薄、润、燥、滑、糙、腻、腐、剥落、有根或无根")


class TongueVeinFeatures(BaseModel):
    """舌下络脉特征."""

    color: str = Field(default="", description="淡紫、略青、青紫、紫黑或暗红")
    type: str = Field(default="", description="粗细、长度、曲张、瘀点、迂曲、怒张、囊泡、分支或对称性")


class TongueFeatures(BaseModel):
    """舌象结构化特征."""

    legal: str = Field(default="", description="图片合规性：是或否")
    back: str = Field(default="", description="正面或反面")
    tongue_name: TongueBodyFeatures = Field(default_factory=TongueBodyFeatures)
    tongue_coat: TongueCoatFeatures = Field(default_factory=TongueCoatFeatures)
    surface: str = Field(default="", description="舌面可见异常；无异常时为正常")
    vein: TongueVeinFeatures = Field(default_factory=TongueVeinFeatures)


class TongueFaceResult(BaseModel):
    """舌、面和局部患处图片分析结果."""

    tongue: TongueFeatures = Field(default_factory=TongueFeatures)
    face: dict[str, Any] = Field(default_factory=dict, description="面色、光泽、形态等可见特征")
    lesions: dict[str, Any] = Field(default_factory=dict, description="患处位置、形态、颜色、范围等可见特征")


class ClinicalExtractionResult(BaseModel):
    """供后续知识库存储的病机结构化字段."""

    etiology: list[str] = Field(default_factory=list, description="病因")
    pathogenesis: list[str] = Field(default_factory=list, description="病机")
    disease_location: list[str] = Field(default_factory=list, description="病位")
    disease_stage: str = Field(default="", description="病期")
    disease_course: str = Field(default="", description="病程及演变")
    onset_triggers: list[str] = Field(default_factory=list, description="诱因")
    key_symptoms: list[str] = Field(default_factory=list, description="关键症状和体征")
