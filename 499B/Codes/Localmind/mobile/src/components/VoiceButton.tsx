import React, { useState } from "react";
import { ActivityIndicator, Alert, TouchableOpacity, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { startRecording, stopRecording } from "../services/voice";
import useChatStore from "../store/useChatStore";

interface Props {
  onTranscription: (text: string) => void;
}

export default function VoiceButton({ onTranscription }: Props) {
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const { voiceMode, setVoiceMode, isStreaming } = useChatStore();

  const handlePress = async () => {
    if (isStreaming) return;

    try {
      if (isRecording) {
        // Stop and transcribe
        setIsRecording(false);
        setIsTranscribing(true);
        const text = await stopRecording();
        setIsTranscribing(false);
        if (text) onTranscription(text);
      } else {
        // Start recording — also enable voice mode so response is spoken
        setVoiceMode(true);
        await startRecording();
        setIsRecording(true);
      }
    } catch (err: any) {
      setIsRecording(false);
      setIsTranscribing(false);
      Alert.alert("Voice Error", err.message || "Recording failed.");
    }
  };

  const handleLongPress = () => {
    // Long press toggles voice reply mode without recording
    setVoiceMode(!voiceMode);
  };

  // Determine button appearance
  const getBg = () => {
    if (isRecording) return "#EF4444";       // red — recording
    if (isTranscribing) return "#F59E0B";    // amber — processing
    if (voiceMode) return "#7C3AED";         // purple — voice reply on
    return "#2A2A3E";                         // default
  };

  const getIcon = (): any => {
    if (isRecording) return "stop";
    if (isTranscribing) return "hourglass-outline";
    if (voiceMode) return "volume-high";
    return "mic";
  };

  return (
    <TouchableOpacity
      onPress={handlePress}
      onLongPress={handleLongPress}
      disabled={isTranscribing}
      style={{
        width: 42,
        height: 42,
        borderRadius: 21,
        backgroundColor: getBg(),
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      {isTranscribing ? (
        <ActivityIndicator size="small" color="#fff" />
      ) : (
        <Ionicons name={getIcon()} size={20} color="#fff" />
      )}
    </TouchableOpacity>
  );
}
