import { Platform } from "react-native";
import { Audio } from "expo-av";
import api from "./api";

// ── Native-only import (expo-file-system not available on web) ─────
let FileSystem: typeof import("expo-file-system/legacy") | null = null;
if (Platform.OS !== "web") {
  FileSystem = require("expo-file-system/legacy");
}

// ── State ──────────────────────────────────────────────────────────
let _recording: Audio.Recording | null = null;   // native
let _mediaRecorder: MediaRecorder | null = null;  // web
let _audioChunks: Blob[] = [];                    // web
let _sound: Audio.Sound | null = null;            // native playback

// ── Helpers ────────────────────────────────────────────────────────

/** Pick best supported mimeType for browser MediaRecorder. */
function getBestMimeType(): string {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/ogg",
    "audio/mp4",
  ];
  for (const t of candidates) {
    if (typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(t)) {
      console.log("[Voice] MediaRecorder using mimeType:", t);
      return t;
    }
  }
  console.log("[Voice] MediaRecorder: no preferred type supported, using default");
  return "";
}

/** Derive a file extension from a mimeType string. */
function extFromMime(mime: string): string {
  if (mime.includes("webm")) return ".webm";
  if (mime.includes("ogg"))  return ".ogg";
  if (mime.includes("mp4"))  return ".mp4";
  return ".webm";
}

// ── Recording — start ─────────────────────────────────────────────

export const startRecording = async (): Promise<void> => {
  console.log("[Voice] startRecording — platform:", Platform.OS);
  await stopPlayback();

  if (Platform.OS === "web") {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Microphone not available in this browser.");
    }
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeType = getBestMimeType();
    _mediaRecorder = mimeType
      ? new MediaRecorder(stream, { mimeType })
      : new MediaRecorder(stream);
    _audioChunks = [];
    _mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) _audioChunks.push(e.data);
    };
    _mediaRecorder.start();
    console.log("[Voice] MediaRecorder started, state:", _mediaRecorder.state);
  } else {
    const { granted } = await Audio.requestPermissionsAsync();
    if (!granted) throw new Error("Microphone permission denied.");
    await Audio.setAudioModeAsync({ allowsRecordingIOS: true, playsInSilentModeIOS: true });
    const { recording } = await Audio.Recording.createAsync(
      Audio.RecordingOptionsPresets.HIGH_QUALITY
    );
    _recording = recording;
    console.log("[Voice] expo-av recording started");
  }
};

// ── Recording — stop & transcribe ─────────────────────────────────

export const stopRecording = async (): Promise<string> => {
  console.log("[Voice] stopRecording — platform:", Platform.OS);
  if (Platform.OS === "web") {
    return _stopRecordingWeb();
  }
  return _stopRecordingNative();
};

async function _stopRecordingNative(): Promise<string> {
  if (!_recording) throw new Error("No active recording.");
  await _recording.stopAndUnloadAsync();
  const uri = _recording.getURI();
  _recording = null;

  await Audio.setAudioModeAsync({ allowsRecordingIOS: false, playsInSilentModeIOS: true });
  if (!uri) throw new Error("Recording URI is null.");
  console.log("[Voice] native URI:", uri);

  const formData = new FormData();
  formData.append("audio", { uri, name: "recording.m4a", type: "audio/mp4" } as unknown as Blob);

  console.log("[Voice] posting to /voice/transcribe...");
  const res = await api.post("/voice/transcribe", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 60000,
  });
  console.log("[Voice] transcription response:", res.status, res.data);
  return res.data.text as string;
}

function _stopRecordingWeb(): Promise<string> {
  if (!_mediaRecorder) throw new Error("No active recording.");

  return new Promise((resolve, reject) => {
    const recorder = _mediaRecorder!;
    const mimeType = recorder.mimeType || "audio/webm";
    const ext = extFromMime(mimeType);

    recorder.onstop = async () => {
      try {
        console.log("[Voice] chunks collected:", _audioChunks.length, "mimeType:", mimeType);
        const blob = new Blob(_audioChunks, { type: mimeType });
        _audioChunks = [];

        console.log("[Voice] blob size:", blob.size, "type:", blob.type);
        if (blob.size === 0) throw new Error("Recorded audio is empty.");

        const formData = new FormData();
        formData.append("audio", blob, `recording${ext}`);

        const baseURL = api.defaults.baseURL ?? "http://localhost:8000";
        const token = api.defaults.headers.common["Authorization"] as string;
        console.log("[Voice] posting to:", baseURL + "/voice/transcribe");

        // Use native fetch instead of axios for FormData on web
        const res = await fetch(`${baseURL}/voice/transcribe`, {
          method: "POST",
          headers: { Authorization: token },
          body: formData,
        });

        console.log("[Voice] transcribe HTTP status:", res.status);
        if (!res.ok) {
          const errBody = await res.text();
          console.error("[Voice] transcribe error body:", errBody);
          throw new Error(`Transcription failed (${res.status}): ${errBody}`);
        }

        const data = await res.json();
        console.log("[Voice] transcription result:", data);
        resolve(data.text as string);
      } catch (e) {
        console.error("[Voice] transcription error:", e);
        reject(e);
      }
    };

    recorder.stop();
    recorder.stream.getTracks().forEach((t) => t.stop());
    _mediaRecorder = null;
  });
}

// ── Playback — stop ───────────────────────────────────────────────

export const stopPlayback = async (): Promise<void> => {
  if (_sound) {
    try { await _sound.stopAsync(); await _sound.unloadAsync(); } catch { /* already unloaded */ }
    _sound = null;
  }
};

// ── TTS ───────────────────────────────────────────────────────────

export const synthesizeSpeech = async (text: string): Promise<void> => {
  if (!text.trim()) return;
  console.log("[Voice] synthesizeSpeech:", text.slice(0, 60));

  const token = api.defaults.headers.common["Authorization"] as string;
  const baseURL = api.defaults.baseURL ?? "";
  const url = `${baseURL}/voice/synthesize_get?text=${encodeURIComponent(text)}`;

  if (Platform.OS === "web") {
    const res = await fetch(url, { headers: { Authorization: token } });
    if (!res.ok) throw new Error(`TTS failed: ${res.status}`);
    const blob = await res.blob();
    const objectUrl = URL.createObjectURL(blob);
    const audio = new window.Audio(objectUrl);
    audio.onended = () => URL.revokeObjectURL(objectUrl);
    await audio.play();
  } else {
    if (!FileSystem) throw new Error("FileSystem not available");
    const dest = FileSystem.cacheDirectory + `tts_${Date.now()}.wav`;
    const download = await FileSystem.downloadAsync(url, dest, {
      headers: { Authorization: token },
    });
    if (download.status !== 200) throw new Error(`TTS download failed: ${download.status}`);

    await stopPlayback();
    await Audio.setAudioModeAsync({
      allowsRecordingIOS: false,
      playsInSilentModeIOS: true,
      staysActiveInBackground: false,
    });
    const { sound } = await Audio.Sound.createAsync({ uri: download.uri }, { shouldPlay: true });
    _sound = sound;
    sound.setOnPlaybackStatusUpdate((status) => {
      if (status.isLoaded && status.didJustFinish) {
        sound.unloadAsync();
        _sound = null;
      }
    });
  }
};
