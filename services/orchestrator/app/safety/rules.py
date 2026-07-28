from app.safety.models import RiskAssessment, RiskLevel

EMERGENCY_PATTERNS = (
    "正在伤害自己",
    "已经伤害自己",
    "现在就要自杀",
    "正在实施自杀",
    "已经服下",
)

RED_PATTERNS = (
    "自杀的想法",
    "想自杀",
    "伤害自己的想法",
    "不想活",
    "结束自己的生命",
)

AMBER_PATTERNS = (
    "持续低落",
    "每天都很焦虑",
    "无法正常工作",
    "睡不着",
)


def assess_risk(transcript: str, visual_summary: str) -> RiskAssessment:
    normalized = transcript.strip()
    if any(pattern in normalized for pattern in EMERGENCY_PATTERNS):
        return RiskAssessment(
            level=RiskLevel.EMERGENCY,
            reasons=["immediate_self_harm_language"],
            confidence=1.0,
        )
    if any(pattern in normalized for pattern in RED_PATTERNS):
        return RiskAssessment(
            level=RiskLevel.RED,
            reasons=["explicit_self_harm_ideation"],
            confidence=0.99,
        )
    if any(pattern in normalized for pattern in AMBER_PATTERNS):
        return RiskAssessment(
            level=RiskLevel.AMBER,
            reasons=["distress_or_impairment_language"],
            confidence=0.9,
        )
    return RiskAssessment(
        level=RiskLevel.GREEN,
        reasons=[],
        confidence=0.7,
    )
