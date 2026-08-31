from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class AgentReviewResult(BaseModel):
    """评审 Agent 对业务 Agent 结构化结果的审核意见."""

    followed_prompt: bool = Field(..., description="业务 Agent 的可观察行为是否遵守完整系统提示词和用户任务")
    response_meets_requirements: bool = Field(..., description="最终响应的内容、字段语义和格式是否符合要求")
    passed: bool = Field(..., description="结果是否完全符合原提示词约束和响应模型要求")
    issues: list[str] = Field(default_factory=list, description="未通过的具体问题；通过时为空")
    revision_instructions: str = Field(
        default="",
        description="给原 Agent 的可执行修改要求；通过时为空",
    )


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
        if any(token in text for token in ("待定", "待确定", "原因待定")):
            raise ValueError("诊断结果不能包含待定类占位文本")
        return text

class HistoryCleaningResult(BaseModel):
    """主诉、现病史、五史及诊断的一体化清洗结果."""

    patient_appeal: str = Field(default="", description="清洗后的主诉；所有允许来源均无有效信息时可为空")
    new_medical_history: str = Field(default="", description="经一致性校验、清洗或补全后的本次现病史，最终必须有内容")
    complaint_history_consistent: bool | None = Field(default=None, description="主诉与本次现病史是否一致")
    consistency_issues: list[str] = Field(default_factory=list, description="不一致点；无不一致时为空")
    old_medical_history: str = Field(default="", description="有效既往史")
    allergic_history: str = Field(default="", description="有效过敏史")
    personal_history: str = Field(default="", description="有效个人史")
    special_history: str = Field(default="", description="有效专科特殊史")
    family_history: str = Field(default="", description="有效家族史")
    diagnosis_illness: str = Field(default="", description="清洗修正后的西医诊断")
    diagnosis_illness_reason: str = Field(default="", description="西医诊断修正理由")
    diagnosis_disease: str = Field(default="", description="清洗修正后的中医证型")
    diagnosis_disease_reason: str = Field(default="", description="中医证型修正理由")
    diagnosis_sickness: str = Field(default="", description="清洗修正后的中医诊断")
    diagnosis_sickness_reason: str = Field(default="", description="中医诊断修正理由")

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


ImageCategory = Literal["舌", "面", "患处", "检验检查报告类", "其他类"]


class ImageClassification(BaseModel):
    """单张病历图片的分类结果."""

    image_index: int = Field(..., ge=1, description="图片在输入列表中的序号，从 1 开始")
    image_type: ImageCategory = Field(..., description="舌、面、患处、检验检查报告类或其他类")


class ImageClassificationResult(BaseModel):
    """一组病历图片的逐张分类结果."""

    images: list[ImageClassification] = Field(default_factory=list)


class InspectionFinding(BaseModel):
    """单张检查报告图片的分类和解析结果."""

    image_type: str = Field(default="", serialization_alias="图片类别", description="预定义的图片类别")
    is_valid_report: bool = Field(default=False, serialization_alias="是否有效检查报告")
    report_name: str = Field(default="", description="报告或检查项目名称")
    report_date: str = Field(default="", serialization_alias="报告日期")
    relative_to_visit_time: str = Field(
        default="",
        serialization_alias="相对就诊时间",
        description="报告日期相对本次 see_doc_time 的时间，如2年前、3个月前、5天前或当天",
    )
    content: str = Field(default="", serialization_alias="解析结果")
    abnormal_indicators: list[str] = Field(default_factory=list, serialization_alias="异常指标")
    abnormal_results: list[str] = Field(
        default_factory=list,
        serialization_alias="异常结果",
        description="检查、影像或门诊病历中原文明确记载的异常检查结果",
    )
    conclusion: str = Field(default="", serialization_alias="报告结论")
    discharge_diagnosis: str = Field(default="", serialization_alias="出院诊断")
    discharge_condition: str = Field(default="", serialization_alias="出院情况")


class InspectionResult(BaseModel):
    """一组检查报告图片的分类和有效信息汇总."""

    reports: list[InspectionFinding] = Field(default_factory=list)
    history_evidence_summary: str = Field(default="", serialization_alias="现病史检查证据摘要")


class TongueBodyFeatures(BaseModel):
    """舌质特征."""

    color: str = Field(default="", description="舌质可见颜色；淡白、淡红、红、绛、青紫等仅供参考")
    shape: list[str] = Field(
        default_factory=list,
        description="舌体可见形态；老、嫩、胖、大、瘦、点刺、裂纹、齿痕等仅供参考",
    )
    posture: list[str] = Field(
        default_factory=list,
        description="可见舌态；痿软、强硬、歪斜、颤动、吐弄、短缩等仅供参考",
    )
    fluid: str = Field(default="", description="舌体表面可见津液状态；润、干、少津、无津等仅供参考")
    surface: list[str] = Field(default_factory=list, description="舌体表面其他可见特征，如瘀点、红点或芒刺")


class TongueCoatFeatures(BaseModel):
    """舌苔特征."""

    color: str = Field(default="", description="舌苔可见颜色；白苔、黄苔、灰苔、黑苔等仅供参考")
    thickness: str = Field(default="", description="舌苔可见厚薄；薄苔、厚苔等仅供参考")
    moisture: str = Field(default="", description="舌苔可见润燥；润苔、滑苔、燥苔、糙苔等仅供参考")
    texture: str = Field(default="", description="舌苔可见质地；腻苔、腐苔、霉酱苔等仅供参考")
    integrity: str = Field(default="", description="舌苔完整程度；全苔、偏苔、剥苔、花剥苔等仅供参考")
    root: str = Field(default="", description="舌苔根性；有根苔、无根苔等仅供参考")
    distribution: list[str] = Field(
        default_factory=list,
        description="舌苔可见分布；舌尖、舌中、舌边、舌根、整体等仅供参考",
    )


class TongueVeinFeatures(BaseModel):
    """舌下络脉特征."""

    color: str = Field(default="", description="舌下络脉可见颜色；正常、淡紫、青紫、紫暗等仅供参考")
    morphology: list[str] = Field(
        default_factory=list,
        description="舌下络脉可见形态；粗张、迂曲、曲张、分支、对称等仅供参考",
    )


class TongueFeatures(BaseModel):
    """舌象结构化特征."""

    legal: str = Field(default="", description="图片合规性：是或否")
    side: str = Field(default="", description="舌面方向：正面或反面")
    tongue_body: TongueBodyFeatures = Field(default_factory=TongueBodyFeatures)
    tongue_coat: TongueCoatFeatures = Field(default_factory=TongueCoatFeatures)
    sublingual_veins: TongueVeinFeatures = Field(default_factory=TongueVeinFeatures)


class FaceSpiritFeatures(BaseModel):
    """面部神态可见特征."""

    level: str = Field(default="", description="神态：得神、少神、失神、假神或神志异常")
    expression: str = Field(default="", description="表情：自然、痛苦、疲惫或其他可见表现")
    gaze: str = Field(default="", description="目光：有神、少神、呆滞或无法判断")


class FaceComplexionFeatures(BaseModel):
    """面色可见特征."""

    category: str = Field(default="", description="面部可见主色；青、赤、黄、白、黑、正常等仅供参考")
    detail: str = Field(
        default="",
        description="面部可见颜色细节；青白、青紫、潮红、萎黄、淡白、苍白等仅供参考",
    )
    distribution: str = Field(default="", description="颜色分布，如全面、两颧、局部或眼眶")
    brightness: str = Field(default="", description="面部可见明暗；明亮、鲜明、晦暗等仅供参考")


class FacialLocalFeatures(BaseModel):
    """五官及面部局部可见特征."""

    eyes: list[str] = Field(
        default_factory=list,
        description="眼部可见特征，如目赤、目黄、眼睑浮肿、分泌物或目光状态",
    )
    nose: list[str] = Field(
        default_factory=list,
        description="鼻部可见特征，如鼻翼煽动、鼻色变化或鼻腔分泌物",
    )
    lips: list[str] = Field(
        default_factory=list,
        description="口唇可见颜色或形态；淡白、红、绛、青紫、干裂等仅供参考",
    )
    gums: list[str] = Field(
        default_factory=list,
        description="牙龈可见特征，如红肿、出血或颜色变化",
    )
    skin: list[str] = Field(
        default_factory=list,
        description="面部皮肤可见特征，如丘疹、红斑、色斑、油脂、干燥、脱屑、毛孔或结节",
    )


class FaceFeatures(BaseModel):
    """面象结构化特征."""

    legal: str = Field(default="", description="是否存在清晰、可分析的真实人脸：是或否")
    spirit: FaceSpiritFeatures = Field(default_factory=FaceSpiritFeatures)
    complexion: FaceComplexionFeatures = Field(default_factory=FaceComplexionFeatures)
    luster: str = Field(default="", description="面部可见光泽；明润、少华、晦暗、油光等仅供参考")
    morphology: list[str] = Field(
        default_factory=list,
        description="面部可见形态；浮肿、消瘦、肌肉松弛、不对称等仅供参考",
    )
    local_features: FacialLocalFeatures = Field(default_factory=FacialLocalFeatures)


class LesionFeatures(BaseModel):
    """局部患处可见特征."""

    location: list[str] = Field(default_factory=list, description="患处所在部位")
    morphology: list[str] = Field(default_factory=list, description="丘疹、红斑、斑疹、脓疱、结节等形态")
    color: list[str] = Field(default_factory=list, description="患处颜色")
    boundary: str = Field(default="", description="边界是否清楚及边缘形态")
    extent: str = Field(default="", description="数量、范围、散在、密集、融合或对称性")
    exudation: str = Field(default="", description="有无渗出及可见程度")
    scaling: str = Field(default="", description="有无鳞屑及可见程度")
    ulceration: str = Field(default="", description="有无糜烂、溃疡或结痂")


class TongueFaceResult(BaseModel):
    """舌、面和局部患处图片分析结果."""

    tongue: TongueFeatures = Field(default_factory=TongueFeatures)
    face: FaceFeatures = Field(default_factory=FaceFeatures)
    lesions: LesionFeatures = Field(default_factory=LesionFeatures)


TONGUE_FACE_RESULT_FIELD_CN_MAPPING: dict[str, str] = {
    "tongue": "舌象",
    "tongue.legal": "舌象图片",
    "tongue.side": "舌面方向",
    "tongue.tongue_body": "舌体",
    "tongue.tongue_body.color": "舌质颜色",
    "tongue.tongue_body.shape": "舌体形态",
    "tongue.tongue_body.posture": "舌态",
    "tongue.tongue_body.fluid": "津液",
    "tongue.tongue_body.surface": "舌面特征",
    "tongue.tongue_coat": "舌苔",
    "tongue.tongue_coat.color": "苔色",
    "tongue.tongue_coat.thickness": "苔厚薄",
    "tongue.tongue_coat.moisture": "苔润燥",
    "tongue.tongue_coat.texture": "苔质",
    "tongue.tongue_coat.integrity": "舌苔完整度",
    "tongue.tongue_coat.root": "苔根",
    "tongue.tongue_coat.distribution": "舌苔分布",
    "tongue.sublingual_veins": "舌下络脉",
    "tongue.sublingual_veins.color": "舌下络脉颜色",
    "tongue.sublingual_veins.morphology": "舌下络脉形态",
    "face": "面象",
    "face.legal": "面象图片",
    "face.spirit": "面神",
    "face.spirit.level": "神态",
    "face.spirit.expression": "表情",
    "face.spirit.gaze": "目光",
    "face.complexion": "面色",
    "face.complexion.category": "面色",
    "face.complexion.detail": "面色表现",
    "face.complexion.distribution": "面色分布",
    "face.complexion.brightness": "面色明暗",
    "face.luster": "面部光泽",
    "face.morphology": "面部形态",
    "face.local_features": "面部局部",
    "face.local_features.eyes": "眼部",
    "face.local_features.nose": "鼻部",
    "face.local_features.lips": "口唇",
    "face.local_features.gums": "牙龈",
    "face.local_features.skin": "面部皮肤",
    "lesions": "患处",
    "lesions.location": "患处部位",
    "lesions.morphology": "患处形态",
    "lesions.color": "患处颜色",
    "lesions.boundary": "患处边界",
    "lesions.extent": "患处范围",
    "lesions.exudation": "渗出",
    "lesions.scaling": "鳞屑",
    "lesions.ulceration": "糜烂溃疡结痂",
}


class ClinicalExtractionResult(BaseModel):
    """供后续知识库存储的病机结构化字段及治则治法."""

    etiology: list[str] = Field(default_factory=list, description="病因")
    pathogenesis: list[str] = Field(default_factory=list, description="病机")
    disease_location: list[str] = Field(default_factory=list, description="病位")
    disease_stage: str = Field(default="", description="病期")
    disease_course: str = Field(default="", description="病程及演变")
    key_symptoms: list[str] = Field(default_factory=list, description="关键症状和体征")
    treatment_principle: str = Field(default="", description="治则治法")


class TreatmentPrincipleResult(BaseModel):
    """基于病史和处方反推的治则治法."""

    treatment_principle: str = Field(default="", description="治则治法")
