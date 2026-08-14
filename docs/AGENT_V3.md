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
  -> decision_gate
       -> required evidence available: model worker
       -> required evidence unavailable: deterministic abstain
  -> output_guard
```

当前意图包括一般支持、情绪表达、心理教育、应对练习、记忆召回、记忆控制和人工接管。风险等级始终覆盖意图：危机轮次禁止读取长期记忆和普通 RAG，正常模型路径不会运行。

证据采用 fail-closed 规则：

- 心理教育必须有仍在审核有效期内、适用于当前风险级别的 reviewed knowledge；
- 记忆问答必须有当前用户已确认且通过治理门的 Memory；
- 没有必需证据时返回确定性拒答，不让 Qwen 猜测；
- 普通共情支持不强制引用知识，因此不会为了聊天而无条件运行 BGE-M3。

## 后续阶段

- **V3.1**：用类型化 capability registry 代替工具名称集合；加入来源、影响级别、审批角色、每轮预算、参数摘要、幂等和 MCP 信任边界。
- **V3.2**：统一 reviewed RAG 与 governed Memory 的证据 envelope；校验引用、来源冲突、降级和“不足以支持医学事实”的边界。
- **V3.3**：生成独立 AvatarResponsePlan，约束语速、停顿、动作强度、注视和可打断性；危机表达不采用娱乐化动作。
- **V3.4**：构建跨层场景集，分别评估意图、风险覆盖、证据拒答、禁止能力、trace 完整性、数字人安全与端到端延迟。

## 设计依据

MCP 工具描述按不可信输入处理，host 必须负责明确同意、访问控制和最小权限；高影响动作通过持久化状态暂停并等待人工审批。V3 沿用 LangGraph 的 durable execution/interrupt 思路，但所有副作用仍放在审批之后并保持幂等。风险报告遵循“工程验证与真实部署/临床治理分开”的原则。

- [MCP Specification: Security and Trust & Safety](https://modelcontextprotocol.io/specification/2025-03-26/index)
- [MCP Authorization and least privilege](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization)
- [LangGraph human-in-the-loop interrupts](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/)
- [OWASP Agentic AI threats and mitigations](https://genai.owasp.org/resource/agentic-ai-threats-and-mitigations/)
- [NIST AI RMF Generative AI Profile](https://www.nist.gov/itl/ai-risk-management-framework)

这些资料用于定义工程控制面，不构成医疗合规或临床有效性证明。
