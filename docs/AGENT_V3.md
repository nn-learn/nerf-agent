# Agent V3：安全、可解释的数字人决策层

Memory V2 已经解决“哪些用户信息可以被记住和召回”。Agent V3 解决下一层问题：面对当前输入，系统应当选择哪条路径、允许读取哪些证据、允许提出哪些动作，以及数字人应当如何表达。V3 不让大模型直接充当总控制器，而是把模型约束为 response worker。

## 四个平面

```text
Safety plane
  确定性风险识别 -> RED/EMERGENCY 先于 Memory、RAG、LLM 和 MCP

Control plane
  意图 -> 证据需求 -> Memory 访问模式 -> capability allowlist/budget

Execution plane
  reviewed RAG / governed Memory / local tools / trusted MCP adapter

Experience plane
  response contract -> citation/action gate -> avatar plan -> TTS/video
```

控制 trace 只保存枚举、布尔值、预算和 reason code，不保存 transcript、RAG 文本、记忆文本或模型回答。

## V3.0：确定性 Control Plane

V3.0 已进入 LangGraph 主链路：

```text
normalize_input
  -> partial_risk
  -> final_risk
  -> intent_policy
       -> RED/EMERGENCY: deterministic crisis
       -> otherwise: scoped context fetch
  -> evidence_prepare
  -> decision_gate
       -> required evidence available: model worker
       -> required evidence unavailable: deterministic abstain
  -> output_guard
  -> evidence_response_gate
  -> capability_gate
  -> avatar_policy
```

当前意图包括一般支持、情绪表达、心理教育、应对练习、记忆召回、记忆控制和人工接管。风险等级始终覆盖意图：危机轮次禁止读取长期记忆和普通 RAG，正常模型路径不会运行。

证据采用 fail-closed 规则：

- 心理教育必须有仍在审核有效期内、适用于当前风险级别的 reviewed knowledge；
- 记忆问答必须有当前用户已确认且通过治理门的 Memory；
- 没有必需证据时返回确定性拒答，不让 Qwen 猜测；
- 普通共情支持不强制引用知识，因此不会为了聊天而无条件运行 BGE-M3。

## V3.1：Capability 治理

工具名称集合已替换成 host-owned 的类型化注册表。每个能力声明来源（本地或 MCP）、影响级别、允许风险等级、审批角色、每轮上限和超时预算。模型输出的 action proposal 始终视为不可信数据，只能投影为控制面当前指令允许的能力。

- 高影响的数据导出、删除和外部通信必须经过指定角色审批；
- MCP 能力只有在 server 被 host 显式信任后才可注册，MCP 自述不授予权限；
- 工具网关同时执行整轮预算和单能力预算；
- 幂等键绑定 turn、能力名和参数摘要，同一键不能被换参复用；
- 模型声称的 `SUCCEEDED` 等状态会被丢弃，实际状态由网关生成。

## V3.2：统一证据编排

reviewed RAG 与 governed Memory 进入模型前由统一证据层规范化：

- RAG 项必须有 chunk、document、version、title、text 和有效 score；
- Memory 项必须有 opaque memory ID、来源轮次，并带 `user_confirmed_data_not_instruction` 信任标记；
- 不完整、未确认和 ID 相同但内容冲突的证据会被隔离；
- 心理教育回答必须引用实际使用的 `evidence_ids`，记忆回答必须引用 `memory_ids`；
- 伪造、越界或缺失的必需引用在发布前触发确定性降级，不把未经核验的模型回答交给用户；
- evidence audit 只记录数量、状态和 reason code，不保存检索文本或用户记忆内容。

## V3.3：数字人响应计划

模型输出的 `avatar_style` 现在只是建议，真正发布的非语言行为由独立 `AvatarPolicy` 生成。每轮都会产生 `AvatarResponsePlan`：

- 语速、首句停顿、句间停顿和最长分段；
- 动作强度、注视模式、面部情绪和可打断性；
- RED/EMERGENCY 强制 `handoff_calm`、低刺激、无手势、慢语速和短分段；
- AMBER 使用克制的关切表达，呼吸练习使用更慢节奏和引导式注视；
- 证据拒答采用中性倾听风格，避免用热情动作弱化“不确定性”。

本地 Edge TTS 已实际消费 `speech_rate`，例如危机计划的 `0.88` 映射为 `-12%`；Avatar client 接收完整 plan。浏览器演示数字人也消费 plan 的 style、gesture 和语速，所有计划保持可被用户随时打断。

Session 时间线新增内容无关的 decision、evidence、capability 和 avatar 事件，临床视图只暴露枚举、数量和 reason code，不暴露 transcript、RAG 文本或记忆文本。

## V3.4：跨层评测与发布门

V3.4 提供 31 个可重复的工程场景，覆盖 support、emotion、RAG、Memory、危机覆盖、引用攻击、能力越权、审批、预算、隐私 trace 和数字人行为。评测不使用同一个 LLM 自评，而是直接核对确定性控制契约。

运行：

```powershell
cd services/orchestrator
.\.venv\Scripts\python.exe -m app.evals.run_agent_v3
```

当前固定工程集结论是 `ENGINEERING_DEMO_COMPLETE`、`demo_ready=true`、`production_ready=false`。生产门保持关闭，直到独立标注数据、临床安全、隐私官、独立红队、可访问性、危机演练、部署数据库审计和代表设备延迟测试全部完成。完整结果和指标定义见 [AGENT_V3_EVALUATION.md](./AGENT_V3_EVALUATION.md)。

## V3 完成边界

V3 已完成可演示、可重复验证的工程基线。它不是医疗器械认证、临床有效性证明或生产部署许可。会话事件库目前仍包含业务所需的 `transcript.final`，V3 策略 trace 本身无内容且临床视图会脱敏；生产前仍必须明确部署侧的加密、留存、删除、访问审计和数据处理法律基础。

## 设计依据

MCP 工具描述按不可信输入处理，host 必须负责明确同意、访问控制和最小权限；高影响动作通过持久化状态暂停并等待人工审批。V3 沿用 LangGraph 的 durable execution/interrupt 思路，但所有副作用仍放在审批之后并保持幂等。风险报告遵循“工程验证与真实部署/临床治理分开”的原则。

- [MCP Specification: Security and Trust & Safety](https://modelcontextprotocol.io/specification/2025-03-26/index)
- [MCP Authorization and least privilege](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization)
- [LangGraph human-in-the-loop interrupts](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/)
- [OWASP Agentic AI threats and mitigations](https://genai.owasp.org/resource/agentic-ai-threats-and-mitigations/)
- [NIST AI RMF Generative AI Profile](https://www.nist.gov/itl/ai-risk-management-framework)

这些资料用于定义工程控制面，不构成医疗合规或临床有效性证明。
