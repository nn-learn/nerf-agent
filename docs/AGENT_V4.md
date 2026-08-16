# Agent V4：用户自治的纵向心理支持闭环

Agent V3 解决“这一轮能做什么、能用什么证据、能否调用能力以及数字人如何表达”。Agent V4 在不改变本地模型栈的前提下，增加跨轮但不过度自主的支持闭环：系统可以记住当前处于倾听、澄清、选择、练习、反思、结束还是人工接管阶段，但不能替用户设定治疗目标、诊断用户或自动执行干预。

模型栈保持为本地 Ollama Qwen 3.6、BGE-M3、faster-whisper、Edge TTS 和 RAD-NeRF。V4 新增的是位于模型之上的确定性策略、状态、审计与评测，不更换语音或检索模型。

## V4 安全边界

- 产品定位仍是心理支持与工程演示，不是诊断、治疗或医疗器械认证；
- 目标只能来自用户当前会话中的明确表达，Memory 和 RAG 不能替用户确认目标；
- RED/EMERGENCY 可以覆盖普通流程，但必须标记为 `SAFETY_OVERRIDE`，不能伪装成用户同意；
- 练习、外部通信、记忆删除等动作必须经过各自的同意或审批门；
- 会话状态与策略 trace 只保留枚举、布尔值、计数和 reason code，不保留目标原文、transcript、RAG 文本或 Memory 文本；
- 面部或声音信号不能被用于推断临床改善、治疗效果或用户同意。

## V4.0：Care Loop 状态机

V4.0 已将会话级 `CareLoopState` 接入 LangGraph 与实时 Session：

```text
final_risk -> intent_policy -> care_loop
                               |-- risk override -> HANDOFF
                               |-- explicit user goal -> confirmed goal + phase
                               `-- no explicit goal -> LISTEN / CLARIFY
```

阶段枚举为 `LISTEN`、`CLARIFY`、`UNDERSTAND`、`CHOOSE_STEP`、`PRACTICE`、`REFLECT`、`CLOSE` 和 `HANDOFF`。目标只保存为 `BE_HEARD`、`CALM_BODY`、`UNDERSTAND_EXPERIENCE`、`CONNECT_HUMAN` 等有限类别，并通过 `UNSET`、`USER_CONFIRMED`、`SAFETY_OVERRIDE` 区分所有权。

状态在单个 Session 内跨轮延续，只有未被 barge-in 或取消的当前 turn 才能提交；Session 结束时立即清空。模型可读取这一内容无关状态以保持对话连续，但无权修改状态。事件 `care.loop.transitioned` 提供内容无关的阶段审计。

## 后续阶段

- V4.1：低风险微干预目录、适应条件、禁忌条件、冷却与确定性选择账本；
- V4.2：提议、明确接受、进行、完成、拒绝、停止和打断的同意状态机；
- V4.3：仅依赖用户主动反馈的效果信号，以及无原始心理内容的纵向漂移监测；
- V4.4：多轮情景评测、隐私泄漏门、危机覆盖门和 fail-closed 发布报告。

## 设计依据

V4 将 WHO 对健康 AI 的人类自治、知情同意、安全、透明和持续审计原则映射为可测试的代码契约；将 NIST AI RMF 的治理、测量和持续管理思路映射为发布门；可观测字段采用 OpenTelemetry GenAI 语义约定的结构化思路。MCP 的 elicitation 和 task 状态机只作为未来外部能力的人机确认参考，实验性 task 不作为当前核心会话状态的依赖。

- [WHO Ethics and governance of AI for health](https://www.who.int/publications/i/item/9789240037403)
- [WHO guidance for large multi-modal models](https://www.who.int/news/item/18-01-2024-who-releases-ai-ethics-and-governance-guidance-for-large-multi-modal-models)
- [NIST AI RMF Generative AI Profile](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence)
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/)
- [MCP elicitation](https://modelcontextprotocol.io/specification/2025-11-25/client/elicitation)
- [MCP tasks](https://modelcontextprotocol.io/specification/2025-11-25/basic/utilities/tasks)
