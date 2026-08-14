# Memory V2.4 分层评测协议

V2.4 的目标不是用一个总分宣布“记忆已经做好”，而是把失败定位到不同阶段：治理、检索、证据准备和最终回答。仓库自带的 6 个案例是可重复的工程回归，不是独立金标集，也不能证明真实心理支持效果。

## 四个对照臂

| 对照臂 | 作用 | 是否默认在线启用 |
| --- | --- | --- |
| `flat-evidence` | 受治理的词法源记忆基线 | 是，作为基础能力 |
| `profile-first-confirmed` | 已确认画像优先，再展开到金标源证据 | 画像层已在线；评测中的批量确认仅为反事实对照 |
| `episode-freshness-v2.3` | 画像 + 事件源证据展开 + 仅排序 freshness | 是，受配置开关控制 |
| `bge-m3-hybrid` | 原有 BGE-M3 语义分支与词法分支融合 | 仍为离线/影子对照，不因小样本领先而自动上线 |

后台画像和事件索引在计时开始前构建，P50/P95 只统计在线 `retrieve`。画像和事件摘要本身不按未知 ID 扣分：评测沿受治理的 `evidence_memory_ids` 回溯到金标源记忆；生产模型上下文仍遵守原有规则。

## 指标分层

检索层报告：

- `complete_evidence_recall_at_5`：每个可回答问题的相关证据覆盖比例后再取均值；不同于“命中任意一条即为 1”的旧二值 Recall。
- `precision_at_5`、`MRR`、`nDCG@5`：分别观察噪声、首条相关证据位置和分级排序质量。
- `correct_abstention_rate`：没有相关记忆时是否返回空上下文。
- `forbidden_retrieval_rate`：未确认、过期、被替代、跨用户、第三方隐私和完整性异常内容是否泄漏。

回答准备层报告：

- `evidence_sufficiency_rate`：可回答问题是否取全必需证据，未知问题是否保持空上下文。
- `irrelevant_context_rate`：进入候选上下文但不在金标相关集合中的比例。

最终回答层只在显式指定 `--with-ollama-answers` 时运行，继续使用本地 `qwen3.6:latest`。系统不保存回答原文，只保存是否遵循答案契约、是否采纳错误记忆，以及 `MISSING_REQUIRED_TERM_GROUP_n`、`FORBIDDEN_TERM_PRESENT`、`ANSWER_TOO_SHORT`、`ANSWER_TOO_LONG` 等诊断码。

## 当前工程基线

2026-08-14 在当前 CPU-only 开发机、6 案例/6 查询上运行真实 BGE-M3：

| 对照臂 | 完整证据 Recall@5 | Precision@5 | nDCG@5 | 证据充分率 | 禁用记忆泄漏 | 在线 P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flat evidence | 0.500 | 0.750 | 0.600 | 0.500 | 0.000 | 约 7.5 ms |
| profile first | 0.625 | 0.750 | 0.612 | 0.667 | 0.000 | 约 9.3 ms |
| episode + freshness | 0.750 | 0.750 | 0.708 | 0.833 | 0.000 | 约 15.6 ms |
| BGE-M3 hybrid | 0.875 | 1.000 | 0.947 | 0.833 | 0.000 | 约 553.8 ms |

BGE-M3 是当前最佳**检索证据对照臂**，不是自动上线结论。最初对它追加 6 个本地 Qwen 最终回答后，答案契约遵循率为 `0.833`，错误记忆采纳率为 `0.000`；唯一失败来自空记忆时的随机拒答措辞。V2.5 将空上下文分支改为模型调用前的确定性拒答，2026-08-14 重新运行真实 `qwen3.6:latest` 后，答案契约遵循率为 `1.000`，错误记忆采纳率仍为 `0.000`。这项改动同时减少了一次无证据模型调用，但并不改变数据集仍为小型工程集的性质。

延迟是单机观测值，不是稳定 SLA。小型合成集会产生宽且不可靠的置信区间，当前报告因此明确设置 `production_promotion_ready=false`。

## 运行方式

快速三臂回归：

```powershell
Set-Location services\orchestrator
.\.venv\Scripts\python.exe -m app.memory.run_strategy_evaluation
```

加入真实 BGE-M3：

```powershell
.\.venv\Scripts\python.exe -m app.memory.run_strategy_evaluation --with-bge
```

只对最佳 BGE 臂调用本地 Qwen，避免对四臂重复生成：

```powershell
.\.venv\Scripts\python.exe -m app.memory.run_strategy_evaluation `
  --with-bge `
  --with-ollama-answers `
  --answer-arm bge-m3-hybrid
```

### V2.4.1 多用户压力门

V2.4.1 另外提供 10 个隔离用户、每人 5 类查询的 50 查询确定性压力评测，覆盖当前事实、应对方式、时间更新、未知问题拒答、过期记忆、未确认记忆、完整性异常、跨用户隔离和 episode 来源约束：

```powershell
.\.venv\Scripts\python.exe -m app.memory.run_stress_evaluation
```

当前工程结果为：answerable recall `1.000`、correct abstention `1.000`、禁用记忆泄漏 `0.000`、跨用户泄漏 `0.000`、summary 上下文泄漏 `0.000`，CPU 检索 P95 约 `13.9 ms`。报告将 `independently_annotated_query_count` 固定为 `0`，所以即使工程查询数达到 50，`independent_graph_gate_ready` 仍为 `false`。合成压力规模与独立金标规模不能相互替代。

## 从工程集升级为独立金标

1. 先冻结查询、候选记忆和切片，再进行阈值实验，禁止用 TEST 反向调参。
2. 每个查询至少由两名互不知晓系统排序结果的标注者给出 0–3 级相关性，并单独标注禁止召回原因。
3. 保存原始 annotations，报告 exact agreement 和 quadratic weighted kappa；冲突由第三人裁决为 adjudicated labels。
4. 评测运行器接受 `--annotations` 和 `--adjudicated-labels`；只有 manifest 声明 `INDEPENDENTLY_ANNOTATED`、审计完整、至少 50 个留出查询且执行最终回答评测时，才可能解除当前工程门禁。V2.4.1 的 50 条生成式压力查询不会计入这个数字。
5. 50 例只是 GraphRAG/策略探索门槛，不是生产门槛。生产阈值仍建议至少 500 个独立标注查询，并按时间变化、危机语言、敏感信息、年龄/语言风格和长跨度切片报告。

即使检索指标通过，心理医疗产品上线前仍需独立的临床安全、隐私、伦理、人因和危机升级流程评审。
