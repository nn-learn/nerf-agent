# PsyAvatar Care「心桥」

面向成年用户的数字心理支持与随访工程演示：可打断语音 Agent、用户授权的摄像头视觉理解、RAD-NeRF 数字人，以及可审计的安全状态与模拟专业人员接管。

## 当前实现阶段

第一阶段建立 Python 3.12 Orchestrator、类型化事件、安全边界和 MockProvider。现有 `agent-mind` 与 `RAD-NeRF/RAD-NeRF` 作为只读需求及模型资产来源，不在新产品仓库内修改。

## Orchestrator 开发环境

```powershell
.\scripts\bootstrap.ps1 -PythonExe "C:\path\to\python.exe"
Set-Location services\orchestrator
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Python 必须是 3.12。真实模型凭证只能通过 `PSYAVATAR_*` 环境变量传入；默认 `provider_mode=mock`，无需云端凭证。

## 安全边界

- 本项目不是诊断、治疗、处方或紧急服务。
- 原始麦克风音频、摄像头视频和关键帧默认不落盘。
- 视觉观察不能独立诊断心理状态或提升风险等级。
- V1 不联系真实医院、紧急联系人或公共机构。

