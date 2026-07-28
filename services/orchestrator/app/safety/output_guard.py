from app.safety.models import AgentResponse


class UnsafeResponseError(ValueError):
    pass


class OutputGuard:
    def validate(self, response: AgentResponse) -> AgentResponse:
        text = f"{response.spoken_text} {response.display_text}"
        diagnosis_patterns = (
            "你患有抑郁症",
            "你患有焦虑症",
            "你得了抑郁症",
            "你得了焦虑症",
            "可以诊断为",
            "说明你有精神疾病",
        )
        if any(pattern in text for pattern in diagnosis_patterns):
            raise UnsafeResponseError("diagnosis is forbidden in Agent output")
        return response
