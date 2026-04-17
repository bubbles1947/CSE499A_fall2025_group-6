import React, { useState } from "react";
import { Text, TouchableOpacity, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import Markdown from "react-native-markdown-display";
import { Message } from "../store/useChatStore";

interface Props {
  message: Message;
}

/** Strip DeepSeek R1 <think>...</think> reasoning chains before display. */
function stripThinkTags(text: string): string {
  return text.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
}

// Markdown style rules matched to the dark chat theme
const markdownStyles: any = {
  body: {
    color: "#E5E7EB",
    fontSize: 14,
    lineHeight: 20,
  },
  paragraph: {
    marginTop: 0,
    marginBottom: 6,
    color: "#E5E7EB",
    fontSize: 14,
    lineHeight: 20,
  },
  heading1: { color: "#F9FAFB", fontSize: 18, fontWeight: "700", marginTop: 4, marginBottom: 4 },
  heading2: { color: "#F9FAFB", fontSize: 16, fontWeight: "700", marginTop: 4, marginBottom: 4 },
  heading3: { color: "#F9FAFB", fontSize: 15, fontWeight: "600", marginTop: 2, marginBottom: 2 },
  strong: { fontWeight: "700", color: "#F9FAFB" },
  em: { fontStyle: "italic", color: "#E5E7EB" },
  code_inline: {
    backgroundColor: "#374151",
    color: "#A78BFA",
    fontFamily: "monospace",
    fontSize: 13,
    paddingHorizontal: 4,
    paddingVertical: 1,
    borderRadius: 4,
  },
  fence: {
    backgroundColor: "#1A1A2E",
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    marginVertical: 4,
    borderLeftWidth: 3,
    borderLeftColor: "#7C3AED",
  },
  code_block: {
    backgroundColor: "#1A1A2E",
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontFamily: "monospace",
    fontSize: 13,
    color: "#A78BFA",
  },
  bullet_list: { marginVertical: 4 },
  ordered_list: { marginVertical: 4 },
  list_item: { marginBottom: 2, color: "#E5E7EB" },
  blockquote: {
    backgroundColor: "#2A2A3E",
    borderLeftWidth: 3,
    borderLeftColor: "#7C3AED",
    paddingHorizontal: 10,
    paddingVertical: 4,
    marginVertical: 4,
    borderRadius: 4,
  },
  hr: { backgroundColor: "#374151", height: 1, marginVertical: 8 },
  link: { color: "#A78BFA", textDecorationLine: "underline" as const },
};

export default function MessageBubble({ message }: Props) {
  const isUser = message.role === "user";
  const content = isUser ? message.content : stripThinkTags(message.content);
  const hasSources = !isUser && message.sources && message.sources.length > 0;
  const [showSources, setShowSources] = useState(false);

  const time = new Date(message.created_at).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });

  if (isUser) {
    return (
      <View style={{ alignItems: "flex-end", marginBottom: 6, paddingHorizontal: 12 }}>
        <View
          style={{
            backgroundColor: "#7C3AED",
            borderRadius: 18,
            borderBottomRightRadius: 4,
            paddingHorizontal: 14,
            paddingTop: 9,
            paddingBottom: 7,
            maxWidth: "80%",
          }}
        >
          <Text style={{ color: "#fff", fontSize: 14, lineHeight: 20 }}>{content}</Text>
          <Text style={{ color: "#C4B5FD", fontSize: 10, marginTop: 3, textAlign: "right" }}>{time}</Text>
        </View>
      </View>
    );
  }

  return (
    <View style={{ alignItems: "flex-start", marginBottom: 6, paddingHorizontal: 12 }}>
      <View
        style={{
          backgroundColor: "#2A2A3E",
          borderRadius: 18,
          borderBottomLeftRadius: 4,
          paddingHorizontal: 14,
          paddingTop: 9,
          paddingBottom: 7,
          maxWidth: "85%",
        }}
      >
        <Markdown style={markdownStyles}>{content || " "}</Markdown>

        {/* RAG sources toggle */}
        {hasSources && (
          <TouchableOpacity
            onPress={() => setShowSources((v) => !v)}
            style={{
              flexDirection: "row",
              alignItems: "center",
              marginTop: 6,
              paddingTop: 6,
              borderTopWidth: 1,
              borderTopColor: "#374151",
            }}
          >
            <Ionicons name="document-text-outline" size={12} color="#6B7280" />
            <Text style={{ color: "#6B7280", fontSize: 11, marginLeft: 4 }}>
              {showSources ? "Hide" : "Show"} sources ({message.sources!.length})
            </Text>
            <Ionicons
              name={showSources ? "chevron-up" : "chevron-down"}
              size={11}
              color="#6B7280"
              style={{ marginLeft: 2 }}
            />
          </TouchableOpacity>
        )}

        {hasSources && showSources && (
          <View style={{ marginTop: 6 }}>
            {message.sources!.map((src, i) => (
              <View
                key={i}
                style={{
                  backgroundColor: "#1A1A2E",
                  borderRadius: 8,
                  padding: 8,
                  marginTop: 4,
                  borderLeftWidth: 2,
                  borderLeftColor: "#7C3AED",
                }}
              >
                <Text
                  style={{ color: "#9CA3AF", fontSize: 11, lineHeight: 16 }}
                  numberOfLines={4}
                >
                  {src.trim()}
                </Text>
              </View>
            ))}
          </View>
        )}

        <Text style={{ color: "#6B7280", fontSize: 10, marginTop: 4 }}>{time}</Text>
      </View>
    </View>
  );
}
