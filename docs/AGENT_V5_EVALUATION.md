# Agent V5 可信记忆评测报告

评测日期：2026-08-21

评测版本：`memory-v5-eval-1.0`

发布门版本：`memory-v5-release-gate-1.0`

数据状态：`ENGINEERING_FIXTURE`

## 结论

- 场景：20/20 通过；
- `release_claim`: `ENGINEERING_DEMO_COMPLETE`；
- `engineering_complete`: `true`；
- `demo_ready`: `true`；
- `production_ready`: `false`。

这些结果证明仓库内可信记忆契约可重复执行，不证明临床有效性、真实用户获益、部署备份删除完整性或生产合规。

## 指标

| 指标 | 结果 | 工程门槛 |
|---|---:|---:|
| Case pass rate | 1.000 | 1.000 |
| Forgetting-aware memory accuracy (FAMA) | 1.000 | 1.000 |
| Stale-memory utilization rate | 0.000 | 0.000 |
| Counterfactual evidence benefit rate | 1.000 | 1.000 |
| Cross-user leakage rate | 0.000 | 0.000 |
| Unsolicited sensitive recall rate | 0.000 | 0.000 |
| Historical crisis adoption rate | 0.000 | 0.000 |
| Paused-memory bypass rate | 0.000 | 0.000 |
| Forbidden-use bypass rate | 0.000 | 0.000 |
| Cascade deletion completeness | 1.000 | 1.000 |
| Provenance completeness | 1.000 | 1.000 |
| Deterministic control p95 | 40.9358 ms | ≤ 100 ms |

p95 从测试数据准备完成后开始计时，包含 SQLite 控制操作与召回，不包含 Ollama、BGE 模型推理、STT、TTS、网络或数字人渲染。数值会随磁盘和主机负载变化，发布门使用 100 ms 上限。

## FAMA 与反事实边界

本工程的 FAMA 为当前事实正确召回与失效事实正确抑制的联合准确率。它不应替代 Recall@K、NDCG 或真实纵向数据上的遗忘评估。

Counterfactual evidence benefit 比较“治理记忆可提供预期证据”和“无记忆基线不提供证据”，只测试 Agent 是否获得正确背景，不评估最终生成质量，更不能解释为心理干预效果。

## 保持关闭的生产门

- 少于 500 条独立标注并留出的记忆案例；
- 缺少临床安全和隐私负责人签字；
- 部署数据库、缓存、索引和备份删除审计未通过；
- 纵向用户控制研究未完成；
- 独立 Memory red-team 未通过。

## 复现

```powershell
Set-Location services\orchestrator
.\.venv\Scripts\python.exe -m app.memory.run_v5_evaluation
```

固定场景位于 `evals\memory_v5_cases.jsonl`，完整一键验收使用 `scripts\verify-demo.ps1`。
