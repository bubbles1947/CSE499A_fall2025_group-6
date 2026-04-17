import React, { useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Animated,
  Platform,
  TouchableOpacity,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { startRecording, stopRecording } from "../services/voice";
import useChatStore from "../store/useChatStore";

type MicState = "idle" | "recording" | "transcribing" | "done";

interface Props {
  onTranscription: (text: string) => void;
}

export default function MicButton({ onTranscription }: Props) {
  const [state, setState] = useState<MicState>("idle");
  const { isStreaming } = useChatStore();

  // Pulse animation for recording state
  const pulse = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    if (state === "recording") {
      const anim = Animated.loop(
        Animated.sequence([
          Animated.timing(pulse, { toValue: 1.25, duration: 500, useNativeDriver: true }),
          Animated.timing(pulse, { toValue: 1,    duration: 500, useNativeDriver: true }),
        ])
      );
      anim.start();
      return () => anim.stop();
    } else {
      pulse.setValue(1);
    }
  }, [state]);

  // Auto-reset "done" state after 1.2 s
  useEffect(() => {
    if (state === "done") {
      const t = setTimeout(() => setState("idle"), 1200);
      return () => clearTimeout(t);
    }
  }, [state]);

  const handlePress = async () => {
    if (isStreaming || state === "transcribing") return;

    if (state === "recording") {
      setState("transcribing");
      try {
        const text = await stopRecording();
        setState("done");
        if (text) onTranscription(text);
        else setState("idle"); // empty transcription
      } catch (err: any) {
        const msg = err.message || "Transcription failed.";
        console.error("[MicButton] transcription error:", msg);
        setState("idle");
        if (Platform.OS === "web") {
          window.alert(`Transcription failed: ${msg}`);
        } else {
          Alert.alert("Voice Error", msg);
        }
      }
    } else {
      try {
        await startRecording();
        setState("recording");
      } catch (err: any) {
        const msg = err.message || "Recording failed.";
        console.error("[MicButton] recording error:", msg);
        setState("idle");
        if (Platform.OS === "web") {
          window.alert(`Mic error: ${msg}`);
        } else {
          Alert.alert("Voice Error", msg);
        }
      }
    }
  };

  const bg = {
    idle:          "#2A2A3E",
    recording:     "#EF4444",
    transcribing:  "#F59E0B",
    done:          "#22C55E",
  }[state];

  const iconName: any = {
    idle:          "mic",
    recording:     "mic",
    transcribing:  "mic",
    done:          "checkmark",
  }[state];

  return (
    <Animated.View style={{ transform: [{ scale: state === "recording" ? pulse : 1 }] }}>
      <TouchableOpacity
        onPress={handlePress}
        disabled={state === "transcribing"}
        style={{
          width: 42,
          height: 42,
          borderRadius: 21,
          backgroundColor: bg,
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {state === "transcribing" ? (
          <ActivityIndicator size="small" color="#fff" />
        ) : (
          <Ionicons name={iconName} size={20} color="#fff" />
        )}
      </TouchableOpacity>
    </Animated.View>
  );
}
