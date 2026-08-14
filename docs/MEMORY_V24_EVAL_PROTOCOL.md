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

BGE-M3 是当前最佳**检索证据对照臂**，不是自动上线结论。对它追加 6 个本地 Qwen 最终回答后，答案契约遵循率为 `0.833`，错误记忆采纳率为 `0.000`。未通过项来自隐私/完整性场景的拒答措辞契约，而不是禁用记忆被模型采纳；这提示下一批标注应把“安全但不同措辞”与“事实污染”分开。

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

## 从工程集升级为独立金标

1. 先冻结查询、候选记忆和切片，再进行阈值实验，禁止用 TEST 反向调参。
2. 每个查询至少由两名互不知晓系统排序结果的标注者给出 0–3 级相关性，并单独标注禁止召回原因。
3. 保存原始 annotations，报告 exact agreement 和 quadratic weighted kappa；冲突由第三人裁决为 adjudicated labels。
4. 评测运行器接受 `--annotations` 和 `--adjudicated-labels`；只有 manifest 声明 `INDEPENDENTLY_ANNOTATED`、审计完整、至少 50 个留出查询且执行最终回答评测时，才可能解除当前工程门禁。
5. 50 例只是 GraphRAG/策略探索门槛，不是生产门槛。生产阈值仍建议至少 500 个独立标注查询，并按时间变化、危机语言、敏感信息、年龄/语言风格和长跨度切片报告。

即使检索指标通过，心理医疗产品上线前仍需独立的临床安全、隐私、伦理、人因和危机升级流程评审。
