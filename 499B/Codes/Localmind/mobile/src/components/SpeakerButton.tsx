import React from "react";
import { TouchableOpacity } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import useChatStore from "../store/useChatStore";
import { stopPlayback } from "../services/voice";

/**
 * Toggles voice-reply (TTS) mode.
 * Purple when on, dark when off.
 * Tap while TTS is playing → stops playback immediately.
 */
export default function SpeakerButton() {
  const { voiceMode, setVoiceMode } = useChatStore();

  const handlePress = async () => {
    if (voiceMode) {
      // Turn off + stop any current playback
      await stopPlayback();
      setVoiceMode(false);
    } else {
      setVoiceMode(true);
    }
  };

  return (
    <TouchableOpacity
      onPress={handlePress}
      style={{
        width: 42,
        height: 42,
        borderRadius: 21,
        backgroundColor: voiceMode ? "#7C3AED" : "#2A2A3E",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <Ionicons
        name={voiceMode ? "volume-high" : "volume-mute"}
        size={20}
        color={voiceMode ? "#fff" : "#6B7280"}
      />
    </TouchableOpacity>
  );
}
