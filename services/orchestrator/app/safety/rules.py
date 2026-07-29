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

NON_ACTIONABLE_CONTEXT_MARKERS = (
    "没有",
    "并没有",
    "并不",
    "从没",
    "不曾",
    "否认",
    "新闻",
    "报道",
    "引用",
    "电影",
    "书里",
    "朋友说",
    "开玩笑",
    "假设",
)

CLAUSE_BOUNDARIES = ("。", "！", "？", "；", "，", "\n", "但是", "但", "不过", "可是")


def _has_actionable_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    for pattern in patterns:
        start = 0
        while (match_index := text.find(pattern, start)) >= 0:
            prefix = text[:match_index]
            boundary_index = max(
                (prefix.rfind(boundary) + len(boundary) for boundary in CLAUSE_BOUNDARIES),
                default=0,
            )
            local_context = prefix[boundary_index:][-24:]
            if not any(
                marker in local_context
                for marker in NON_ACTIONABLE_CONTEXT_MARKERS
            ):
                return True
            start = match_index + len(pattern)
    return False


def assess_risk(transcript: str, visual_summary: str) -> RiskAssessment:
    _ = visual_summary
    normalized = transcript.strip()
    if _has_actionable_pattern(normalized, EMERGENCY_PATTERNS):
        return RiskAssessment(
            level=RiskLevel.EMERGENCY,
            reasons=["immediate_self_harm_language"],
            confidence=1.0,
        )
    if _has_actionable_pattern(normalized, RED_PATTERNS):
        return RiskAssessment(
            level=RiskLevel.RED,
            reasons=["explicit_self_harm_ideation"],
            confidence=0.99,
        )
    if _has_actionable_pattern(normalized, AMBER_PATTERNS):
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
