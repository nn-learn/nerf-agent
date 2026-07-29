import {
  Camera,
  CameraOff,
  Hand,
  Mic,
  MicOff,
  PhoneOff,
} from "lucide-react";

interface CallControlsProps {
  cameraActive: boolean;
  microphoneEnabled: boolean;
  disabled?: boolean;
  onCamera: () => void;
  onMicrophone: () => void;
  onInterrupt: () => void;
  onHangUp: () => void;
}


export function CallControls({
  cameraActive,
  microphoneEnabled,
  disabled = false,
  onCamera,
  onMicrophone,
  onInterrupt,
  onHangUp,
}: CallControlsProps) {
  return (
    <nav aria-label="通话控制" className="call-controls">
      <button
        aria-label={microphoneEnabled ? "关闭麦克风" : "开启麦克风"}
        className={`control-button ${microphoneEnabled ? "" : "is-off"}`}
        disabled={disabled}
        onClick={onMicrophone}
        type="button"
      >
        {microphoneEnabled ? <Mic /> : <MicOff />}
        <span>{microphoneEnabled ? "静音" : "取消静音"}</span>
      </button>
      <button
        aria-label={cameraActive ? "暂停视觉" : "开启摄像头"}
        className={`control-button ${cameraActive ? "is-active" : ""}`}
        disabled={disabled}
        onClick={onCamera}
        type="button"
      >
        {cameraActive ? <CameraOff /> : <Camera />}
        <span>{cameraActive ? "暂停视觉" : "开启视觉"}</span>
      </button>
      <button
        aria-label="打断 AI"
        className="control-button interrupt-control"
        disabled={disabled}
        onClick={onInterrupt}
        type="button"
      >
        <Hand />
        <span>打断 AI</span>
      </button>
      <button
        aria-label="结束通话"
        className="control-button hangup-control"
        onClick={onHangUp}
        type="button"
      >
        <PhoneOff />
        <span>结束</span>
      </button>
    </nav>
  );
}
