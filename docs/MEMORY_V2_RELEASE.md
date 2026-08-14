# Memory V2 工程完成与发布门

Memory V2 的工程范围已经闭合，但这不等于心理医疗产品可以直接面向真实用户投产。统一发布报告刻意拆成两个结论：

- `engineering_complete`：代码链路、治理不变量、工程压力、隐私探针和自动回答门是否通过；
- `production_ready`：独立数据、人工评审、部署环境审计、临床/隐私/伦理签字和危机演练是否全部完成。

2026-08-14 在 CPU-only 开发机上用真实 BGE-M3 与本地 `qwen3.6:latest` 运行后，结果为：

| 结论 | 当前值 |
| --- | --- |
| release claim | `ENGINEERING_BASELINE_COMPLETE` |
| selected strategy | `bge-m3-hybrid` |
| automated answer gate | 通过 |
| engineering blockers | 0 |
| GraphRAG research ready | 否 |
| production ready | 否 |

这里的“工程完成”只说明 V2 baseline 能重复运行和失败关闭，不主张临床有效性，也不把生成式工程样本冒充独立金标。

## V2.5 新增闭环

### 1. 隐私与删除审计

`app.evals.privacy` 现在检查：

- SQLite 中意外持久化的音频、视频和摄像头 BLOB；
- observation、profile evidence、episode member、retrieval 和 shadow ranking 的跨 owner 关系；
- 派生表中的孤儿引用；
- episode 和仍可用画像引用失效、未确认、过期、完整性异常或 scope 不一致的源记忆；
- 永久删除后，唯一敏感标记是否仍残留在 SQLite 主文件、WAL 或 SHM。

字节级删除测试会让同一合成标记同时进入源记忆、画像和 episode，再走真实 `purge`。探针在与目标库同一父目录创建隔离临时库，不向生产库写入合成用户内容。扫描器只返回计数和文件类别，不返回用户文本或内部 ID。

运行部署库审计：

```powershell
Set-Location services\orchestrator
.\.venv\Scripts\python.exe -m app.evals.run_privacy_audit `
  runtime\psyavatar.sqlite3
```

### 2. 最终回答失败关闭

当检索上下文为空时，`OllamaMemoryAnswerGenerator` 在 HTTP 调用前直接返回“没有相关已确认记忆、不会猜测”的确定性答案。它解决了 V2.4 唯一一个拒答措辞不稳定案例，也避免一次无证据的大模型调用。当前 6 查询工程集上：

- 最终回答契约遵循率：`1.000`；
- 错误记忆采纳率：`0.000`；
- 禁用记忆检索率：`0.000`。

这些是自动契约指标，不等同于共情质量、临床安全或真实用户体验。

### 3. 脱敏人工评审包

`app.memory.human_review` 提供可审计的人因入口：

- 未声明 `source_deidentified=true` 时拒绝导出；
- case/query/evidence ID 使用至少 16 字符的外部盐生成不透明 ID；
- reviewer 看不到金标标签；
- 每条回答至少需要两个独立 reviewer token；
- 每条回答必须同时覆盖 `CLINICAL` 和 `PRIVACY` 角色；
- 对证据忠实度、共情、专业边界、清晰度和数字人交互适配度做 1–5 分评审；
- 错误个性化、隐私违规、临床安全违规和危机升级失败是零容忍 critical flag。

构建/审计评审包：

```powershell
.\.venv\Scripts\python.exe -m app.memory.run_human_review `
  review_candidates.jsonl `
  --dataset-id heldout-v1 `
  --data-classification DEIDENTIFIED_RESEARCH `
  --id-salt <至少16字符的外部密钥> `
  --deidentified `
  --submissions review_submissions.jsonl
```

盐不应提交到仓库。真实心理健康数据在进入该工具之前，仍需完成同意、最小化、脱敏和访问控制；工具不会声称自动识别并移除所有 PII。

### 4. 统一发布报告

真实 BGE + Qwen 工程闭环：

```powershell
.\.venv\Scripts\python.exe -m app.memory.run_v2_release_evaluation `
  --with-bge `
  --with-ollama-answers
```

对部署数据库运行时增加：

```powershell
--database runtime\psyavatar.sqlite3
```

不带 `--with-ollama-answers` 时，发布门会因为未执行最终回答评测而失败关闭；不带 `--database` 时只验证临时合成 schema，不能解除部署隐私 blocker。

## 尚未解除的外部门槛

以下工作不能由仓库中的合成测试代替，因此当前报告保持 `production_ready=false`：

1. 至少 500 条冻结、独立双人标注并仲裁的留出查询，覆盖长期变化、敏感信息、危机语言、不同年龄/语言风格和长跨度会话；
2. 对实际候选版本至少 50 条回答完成临床与隐私双角色盲评；
3. 在实际部署数据库和备份/日志策略上完成隐私审计；
4. 临床安全、隐私负责人和伦理审批签字；
5. 完成危机升级、人工接管、断网降级和数字人依赖风险演练。

GraphRAG 继续关闭。只有独立标注集达到研究门槛并在检索与最终回答两层稳定优于当前方案时，才把它作为对照臂评估，而不是预先引入额外复杂度。
