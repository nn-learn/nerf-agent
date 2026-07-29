import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ConnectionState,
  LocalVideoTrack,
  RemoteVideoTrack,
  Room,
  RoomEvent,
  Track,
} from "livekit-client";

import { PsyAvatarApi } from "../api/client";
import { useSessionStore } from "../state/sessionStore";
import {
  type MediaConnectionState,
  type MediaSessionController,
  pauseVisionTransaction,
} from "./useMediaSession";

function mapConnectionState(state: ConnectionState): MediaConnectionState {
  if (state === ConnectionState.Connected) return "connected";
  if (state === ConnectionState.Reconnecting) return "reconnecting";
  if (state === ConnectionState.Disconnected) return "disconnected";
  return "connecting";
}


export function useLiveKitMediaSession(
  sessionId: string,
): MediaSessionController {
  const api = useMemo(() => new PsyAvatarApi(), []);
  const room = useMemo(
    () =>
      new Room({
        adaptiveStream: true,
        dynacast: true,
        stopLocalTrackOnUnpublish: true,
      }),
    [],
  );
  const visionState = useSessionStore((state) => state.visionState);
  const setVisionState = useSessionStore((state) => state.setVisionState);
  const agentState = useSessionStore((state) => state.agentState);
  const setAgentState = useSessionStore((state) => state.setAgentState);
  const [connectionState, setConnectionState] =
    useState<MediaConnectionState>("connecting");
  const [microphoneEnabled, setMicrophoneEnabled] = useState(false);
  const [avatarTrack, setAvatarTrack] = useState<RemoteVideoTrack>();
  const [localCameraTrack, setLocalCameraTrack] = useState<LocalVideoTrack>();
  const [errorMessage, setErrorMessage] = useState<string>();

  useEffect(() => {
    let cancelled = false;
    const attachedAudio = new Set<HTMLMediaElement>();

    const onConnectionStateChanged = (state: ConnectionState) => {
      setConnectionState(mapConnectionState(state));
    };
    const onTrackSubscribed = (
      track: Track,
      publication: { trackName: string },
    ) => {
      if (
        track.kind === Track.Kind.Video &&
        publication.trackName === "avatar-video"
      ) {
        setAvatarTrack(track as RemoteVideoTrack);
      }
      if (
        track.kind === Track.Kind.Audio &&
        publication.trackName === "assistant-audio"
      ) {
        const element = track.attach() as HTMLMediaElement;
        element.dataset.psyavatarAudio = "assistant";
        document.body.append(element);
        attachedAudio.add(element);
      }
    };
    const onTrackUnsubscribed = (track: Track) => {
      track.detach().forEach((element) => {
        attachedAudio.delete(element as HTMLMediaElement);
        element.remove();
      });
    };

    room.on(RoomEvent.ConnectionStateChanged, onConnectionStateChanged);
    room.on(RoomEvent.TrackSubscribed, onTrackSubscribed);
    room.on(RoomEvent.TrackUnsubscribed, onTrackUnsubscribed);
    room.on(RoomEvent.Disconnected, () => {
      setConnectionState("disconnected");
      setAgentState("disconnected");
    });

    void (async () => {
      try {
        const credentials = await api.getLiveKitToken(sessionId, "演示用户");
        await room.connect(
          credentials.server_url,
          credentials.participant_token,
        );
        if (cancelled) return;
        await api.setConsent(sessionId, "microphone", true);
        await room.localParticipant.setMicrophoneEnabled(true);
        setMicrophoneEnabled(true);
        setConnectionState("connected");
        setAgentState("listening");
      } catch (error) {
        if (cancelled) return;
        setConnectionState("error");
        setErrorMessage(
          error instanceof Error ? error.message : "实时房间连接失败",
        );
      }
    })();

    return () => {
      cancelled = true;
      attachedAudio.forEach((element) => element.remove());
      room.removeAllListeners();
      void room.disconnect();
    };
  }, [api, room, sessionId, setAgentState]);

  const publishCamera = useCallback(async () => {
    await api.setConsent(sessionId, "camera", true);
    try {
      await room.localParticipant.setCameraEnabled(true);
      const publication = room.localParticipant.getTrackPublication(
        Track.Source.Camera,
      );
      setLocalCameraTrack(publication?.track as LocalVideoTrack | undefined);
      setVisionState("active");
    } catch (error) {
      await api.setConsent(sessionId, "camera", false);
      setVisionState("disabled");
      throw error;
    }
  }, [api, room, sessionId, setVisionState]);

  const pauseVision = useCallback(async () => {
    await pauseVisionTransaction({
      stopCamera: async () => {
        await room.localParticipant.setCameraEnabled(false);
      },
      markPaused: () => {
        setLocalCameraTrack(undefined);
        setVisionState("paused");
      },
      revokeConsent: async () => {
        await api.setConsent(sessionId, "camera", false);
      },
    });
  }, [api, room, sessionId, setVisionState]);

  const toggleMicrophone = useCallback(async () => {
    const next = !microphoneEnabled;
    await room.localParticipant.setMicrophoneEnabled(next);
    setMicrophoneEnabled(next);
  }, [microphoneEnabled, room]);

  const interrupt = useCallback(async () => {
    setAgentState("listening");
    await api.interrupt(sessionId);
  }, [api, sessionId, setAgentState]);

  const hangUp = useCallback(async () => {
    if (visionState === "active") {
      await pauseVision();
    }
    await room.disconnect();
    setConnectionState("disconnected");
    setAgentState("disconnected");
  }, [pauseVision, room, setAgentState, visionState]);

  return {
    connectionState,
    microphoneEnabled,
    visionState,
    agentState,
    avatarTrack,
    localCameraTrack,
    errorMessage,
    publishCamera,
    pauseVision,
    toggleMicrophone,
    interrupt,
    hangUp,
  };
}
