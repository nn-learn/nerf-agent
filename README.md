# PsyAvatar Care「心澄」

面向成年用户的数字人心理支持工程演示：用户可以像视频通话一样与 AI 交流，在明确授权后临时开启摄像头，并在需要时进入可审计的模拟临床接管流程。

本项目是心理支持与随访演示，不是诊断、治疗、处方或紧急服务。

## 当前能力

- Python 3.12 FastAPI + LangGraph 显式安全路由
- GREEN / AMBER / RED / EMERGENCY 四级风险状态
- 评审知识库、BGE-M3 + BM25 混合检索和证据校验
- 需同意的长期记忆、可撤销记忆与高影响工具审批
- 单房间、短时效、最小发布权限的 LiveKit 令牌
- React 数字人视频通话界面和无需 GPU 的 mock 数字人
- 摄像头二次确认、权限拒绝降级和严格暂停顺序
- SQLite 追加式审计事件；拒绝持久化任何嵌套二进制媒体
- 模拟临床转介状态机，不声称已联系真实专业人员

旧项目 `agent-mind/` 和 `RAD-NeRF/RAD-NeRF/` 仅作为只读模型与需求来源。新产品不会修改其中的用户代码。

## 零 GPU 演示

默认 Web 模式是 `mock`，不需要 LiveKit、云端凭证或 GPU：

```powershell
Set-Location apps\web
npm install
npm run dev
```

打开 `http://localhost:5173`。摄像头不会自动启动，只有点击“开启视觉”并在说明弹窗中再次确认后，浏览器才会请求权限。

## 启用真实 WebRTC 房间

1. 启动本地 LiveKit：

```powershell
docker compose -f infra\docker-compose.yml up -d
```

2. 启动 Orchestrator：

```powershell
Set-Location services\orchestrator
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

3. 启动 LiveKit Web 模式：

```powershell
Set-Location apps\web
$env:VITE_MEDIA_MODE = "livekit"
npm run dev
```

本地配置仅用于开发。生产环境必须更换 LiveKit 密钥、启用应用身份认证，并使用可信 TLS/WSS 入口。

## 验证

后端：

```powershell
Set-Location services\orchestrator
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check app tests
.\.venv\Scripts\python.exe -m mypy app
```

前端：

```powershell
Set-Location apps\web
npm run test -- --run
npm run build
```

## 不可突破的边界

- 原始麦克风音频、摄像头视频和采样帧不落盘。
- 视觉观察 10 秒后失效，不能从外貌推断精神诊断、性格或风险。
- 用户令牌只能向一个生成的房间发布摄像头和麦克风轨道。
- 摄像头暂停顺序固定为：停发轨道、更新本地状态、撤销服务端同意。
- V1 不自动联系医院、紧急联系人、公共机构或真实临床人员。
- 真实模型凭证仅从 `PSYAVATAR_*` 环境变量读取，并以秘密字段隐藏。
