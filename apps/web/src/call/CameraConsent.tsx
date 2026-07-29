import { Camera, EyeOff, LockKeyhole, ShieldCheck, X } from "lucide-react";

interface CameraConsentProps {
  open: boolean;
  busy: boolean;
  error?: string;
  onConfirm: () => void;
  onClose: () => void;
}


export function CameraConsent({
  open,
  busy,
  error,
  onConfirm,
  onClose,
}: CameraConsentProps) {
  if (!open) return null;

  return (
    <div className="modal-backdrop" role="presentation">
      <section
        aria-labelledby="camera-consent-title"
        aria-modal="true"
        className="consent-card"
        role="dialog"
      >
        <button
          aria-label="关闭摄像头授权"
          className="icon-button consent-close"
          disabled={busy}
          onClick={onClose}
          type="button"
        >
          <X aria-hidden="true" size={20} />
        </button>
        <div className="consent-icon">
          <Camera aria-hidden="true" size={28} />
        </div>
        <p className="eyebrow">可选视觉能力</p>
        <h2 id="camera-consent-title">让小澄看见你想分享的内容？</h2>
        <p className="consent-copy">
          摄像头只用于当前会话中的即时理解，例如你主动展示睡眠记录卡。不会保存原始视频或截图。
        </p>
        <ul className="privacy-points">
          <li>
            <ShieldCheck aria-hidden="true" size={18} />
            视觉观察 10 秒后自动失效
          </li>
          <li>
            <LockKeyhole aria-hidden="true" size={18} />
            不从外貌推断诊断、性格或风险
          </li>
          <li>
            <EyeOff aria-hidden="true" size={18} />
            随时一键暂停，不影响语音和文字
          </li>
        </ul>
        {error && <p className="consent-error">{error}</p>}
        <div className="consent-actions">
          <button
            className="button button-secondary"
            disabled={busy}
            onClick={onClose}
            type="button"
          >
            暂不开启
          </button>
          <button
            className="button button-primary"
            disabled={busy}
            onClick={onConfirm}
            type="button"
          >
            {busy ? "正在请求权限…" : "同意并开启"}
          </button>
        </div>
      </section>
    </div>
  );
}
