# Memory V2.2：证据驱动、可治理的时态长期画像

V2.0 不把更多对话原文塞进 prompt，而是在 V1 的用户确认事实之上增加一层可治理的长期理解。它面向心理支持场景，默认保守：单次表达不塑造人格，模型建议不等于事实，矛盾未解决时停止召回。

## 已跑通的闭环

```text
V1 已确认记忆
    -> 按真实来源会话记录 observation
    -> 同一 subject_key 内规范化与聚类
    -> 至少两个独立会话支持
    -> 生成 AWAITING_CONFIRMATION 长期画像
    -> 用户查看逐条证据并确认/否定
    -> ACTIVE 画像进入分层召回
```

同一主题出现不同语义签名且横跨至少两个会话时，系统生成冲突组，将该主题全部画像置为 `STALE`。用户必须选择一个选项，或“暂不判断”；冲突期间没有画像进入模型上下文。

## 三层记忆

1. **Evidence layer**：V1 事实/偏好/目标/有效方法/边界，保留来源 turn、有效期、确认状态和更新链。
2. **Profile layer**：多个会话支持的语义画像，包含支持/冲突证据、会话数、置信度和用户编辑标志。
3. **Governance layer**：画像确认、否定抑制、冲突仲裁、撤回/编辑失效、物理删除级联和审计事件。

实时回答使用 `LayeredMemoryRetriever`：只有 `ACTIVE` 画像可召回；当画像已覆盖底层证据时，不重复注入同一事实。画像和原子记忆都继续作为“用户确认的数据而非指令”。

## V2.2 双时间与偏好变化

V2.2 将“内容何时对用户成立”和“系统何时看到它”拆开：

- `valid_at_ms`：用户事实或偏好的有效时间；
- `observed_at_ms`：系统写入 observation 的时间；
- `valid_from_ms` / `valid_to_ms`：画像的半开有效区间 `[from, to)`。

仅仅后出现相反说法不会自动覆盖旧画像，而是继续按冲突处理。只有明确包含“以前……现在……”或“现在改成……”等变化锚点时，才生成独立的 `OPEN` 变更提案；系统先保持原画像可用，用户确认后才把旧画像改为 `HISTORICAL`、将新画像改为 `ACTIVE`。拒绝变更会保留旧画像并抑制同一证据反复提案。

纯历史陈述（例如“以前我喜欢详细回答”）保留在可审计证据层，但不会污染当前画像或制造当前冲突。变化句只解析当前分句，因此“以前喜欢详细，现在改成简短”不会被前半句误判为新偏好。已应用变更获得独立版本签名，后续同义证据只补强当前版本，不会覆盖历史边界。

这一基线刻意不从消息先后顺序猜测用户已经改变，也不自动衰减或遗忘用户确认的内容。未来若加入 freshness/decay，只能降低召回优先级，不能绕过用户治理状态。

## 隐私与安全边界

- API 不返回内部 `user_id`、observation ID、来源 session ID、签名哈希或证据摘要。
- 未确认、过期、撤回、冲突中和完整性异常的内容不能进入画像召回。
- 编辑或撤回一个证据会立即使派生画像失效；永久删除会级联清除 observation、画像、冲突选项和画像召回记录。
- 用户否定的同一画像签名不会因后台重建反复出现。
- V2 只做个性化理解，不提取诊断、药物建议、危机判断或视觉人格推断；确定性风险链仍独立于记忆。
- 所有聚合发生在异步记忆链路，不占用实时音频/回复线程。

## API 与界面

```text
GET  /api/sessions/{id}/memory-profiles
POST /api/sessions/{id}/memory-profiles/{profile_id}/decision
GET  /api/sessions/{id}/memory-conflicts
POST /api/sessions/{id}/memory-conflicts/{conflict_id}/decision
GET  /api/sessions/{id}/memory-changes
POST /api/sessions/{id}/memory-changes/{change_id}/decision
```

记忆中心展示“Agent 对你的长期理解”、证据折叠区、画像确认/否定、冲突选项和时间变化卡片。变化候选不会出现在普通画像确认入口，避免绕过历史切换状态机。用户无需理解向量库或知识图谱，也能看见系统为何形成这个理解并控制它是否生效。

## V2 专项评测

运行：

```powershell
Set-Location services\orchestrator
.\.venv\Scripts\python.exe -m app.memory.run_consolidation_evaluation `
  ..\..\evals\memory_v2_cases.jsonl
```

当前 5 个确定性工程用例结果：

| 指标 | 当前结果 | V2.0 合并门槛 |
| --- | ---: | ---: |
| profile proposal accuracy | 1.000 | 1.000 |
| conflict detection accuracy | 1.000 | 1.000 |
| pre-confirmation leakage rate | 0.000 | 0.000 |
| confirmed profile recall rate | 1.000 | 1.000 |
| deletion cascade rate | 1.000 | 1.000 |

覆盖的失败模式包括：单会话过度概括、未确认事实聚合、跨会话稳定偏好、跨会话矛盾、确认前泄漏和来源删除后画像残留。这只是小型合成治理回归，不代表真实用户画像准确率或临床有效性。

下一阶段需要独立标注数据：至少覆盖时间限定、观点变化、否定、反讽、第三方事实、敏感信息、跨语言表达和长时间跨度；分别报告画像 precision/recall、冲突 precision/recall、过度个性化率、用户否定率、画像采纳后回答一致性，以及不同人群切片。

### V2.2 时间治理评测

运行：

```powershell
Set-Location services\orchestrator
.\.venv\Scripts\python.exe -m app.memory.run_temporal_evaluation `
  ..\..\evals\memory_v22_cases.jsonl
```

当前 6 个确定性工程场景结果：

| 指标 | 当前结果 | 门槛 |
| --- | ---: | ---: |
| temporal classification accuracy | 1.000 | 1.000 |
| false change rate | 0.000 | 0.000 |
| pre-decision current preservation rate | 1.000 | 1.000 |
| valid/observed time accuracy | 1.000 | 1.000 |
| bitemporal boundary accuracy | 1.000 | 1.000 |
| revoked change apply rate | 0.000 | 0.000 |

覆盖明确变化、无锚点相反说法、纯历史说法、同义稳定说法、旧值先出现的变化句、撤回变更证据，以及 `[valid_from, valid_to)` 边界。该小型合成集只证明状态机回归，不证明开放域时态理解准确率；上线前仍需真实中文口语金标集，并优先报告 false change rate。

## V2.3：受治理的事件记忆索引

V2.3 在 evidence/profile 两层之上增加了一个**可重建的离线检索索引**，而不是增加一个可以绕过用户控制的新记忆层：

```text
ACTIVE 且完整性检查通过的源记忆
    -> 按 session、有效时间间隔、成员数和字符数进行有界分组
    -> 生成确定性的 episode summary（后台链路）
    -> summary 只用于定位候选 source memory_id
    -> 再次读取当前仍为 ACTIVE 的源记忆
    -> 只有源记忆文本能够进入模型上下文
```

单个事件最多包含 6 条源记忆、800 个字符，默认按 30 分钟时间间隔切断。编辑、撤回或永久删除任一源记忆时，派生摘要会立即失效或物理删除；后台重建使用 `secure_delete`，旧摘要不会作为可召回文本残留。摘要不在记忆中心展示，因为它不是用户需要单独确认的新事实，用户仍通过源记忆控制其内容与生命周期。

freshness/decay 仅对已经通过治理门的候选做轻量重排：边界和偏好不衰减，事实、目标和应对方式采用不同半衰期，最低保留 0.90 的排序系数。时间不会自动撤回、隐藏或删除一条用户确认的记忆，也不会让失效、跨用户或未确认内容重新进入候选集。

运行专项评测：

```powershell
Set-Location services\orchestrator
.\.venv\Scripts\python.exe -m app.memory.run_episode_evaluation `
  ..\..\evals\memory_v23_cases.jsonl
```

当前 4 个确定性工程用例的结果为：flat multi-hop Recall@K `0.500`，episode Recall@K `0.875`，episode Precision@K `1.000`，摘要进入模型上下文的泄漏率 `0.000`。跨 episode 案例没有收益，这正是当前边界；该小型合成集只验证索引链路和治理不变量，不代表开放域效果。

关系图仍保持关闭。只有独立多跳集合达到至少 50 个案例，并在检索与最终回答两个阶段证明稳定收益后，才讨论 GraphRAG。当前实现参考了 [LongMemEval](https://arxiv.org/abs/2410.10813) 对跨会话、时间和更新能力的任务拆分，以及事件中心长期记忆的简化路线；[Zep 的时态知识图方法](https://arxiv.org/abs/2501.13956) 作为后续对照臂，而不是预设架构。后台整理而非阻塞实时回复的方向，也与 [Letta 的后台 memory processing](https://docs.letta.com/guides/agents/architectures/sleeptime) 一致。

## 后续 2.x 路线

- **V2.1（已实现，可选）**：规则锁定的本地 Qwen 结构化 claim 分组。模型只接收已确认 claim 和不透明 `memory_id`，只返回 ID 聚类，不能输出画像文本、主题、证据或用户属性。完整 ID 覆盖、唯一分配、subject、正反极性和时间边界由代码验证；任何超时、网络错误、格式错误或语义越界都会整批退回精确匹配。常见回答风格、语气、呼吸/冥想等已知语义仍完全由规则处理。该模式默认关闭，只运行在后台线程：

  ```powershell
  $env:PSYAVATAR_MEMORY_CLAIM_NORMALIZER_MODE="ollama"
  ```

  V2.1 使用现有 `PSYAVATAR_TEXT_MODEL`（默认 `qwen3.6:latest`）和 Ollama 地址，不更换用户原有模型。模型成功结果只缓存在进程内，失败结果不缓存，下一次重建可以重试。切换规范化器导致签名变化时，旧 active 画像先变为 `STALE`，新画像必须重新由用户确认。
- **V2.2（已实现）**：双时间 observation、画像有效区间、明确变化提案、历史画像、用户确认/拒绝、证据撤回失效、独立 API/UI 和专项评测。
- **V2.3（已实现）**：离线事件摘要索引、源证据展开、派生数据级联删除，以及只影响召回排序的 freshness/decay；GraphRAG 继续由独立评测门控。
- **V2.4**：扩大独立金标集，比较 flat evidence、profile-first、BGE hybrid 和 reranker，评估检索与最终回答两个阶段。
- **V2.5**：在隐私评审、人因评审和持出集门槛通过后，再考虑更长时间的真实用户试验。

### V2.1 当前工程验证

测试覆盖模型只聚类 ID、规则锁定、伪造/漏掉/重复 ID、跨 subject 合并、跨正反极性合并、HTTP 失败回退、失败重试、成功缓存、两会话门槛、用户确认门槛，以及规范化版本迁移。它们验证的是治理行为，并不证明 Qwen 对开放域语义等价判断的真实准确率。开启 Ollama 模式前仍应新增独立标注的等价/矛盾 pair 集，报告 pairwise precision、recall 和 false-merge rate；心理产品应优先约束 false merge。
