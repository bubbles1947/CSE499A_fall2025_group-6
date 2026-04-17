import React, { useEffect, useRef, useState } from "react";
import {
  Alert,
  FlatList,
  KeyboardAvoidingView,
  Modal,
  Platform,
  Text,
  TextInput,
  TouchableOpacity,
  View,
  FlatList as FlatListType,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";
import { NativeStackNavigationProp } from "@react-navigation/native-stack";
import { RouteProp, useNavigation, useRoute } from "@react-navigation/native";

import useChatStore, { Message } from "../store/useChatStore";
import MessageBubble    from "../components/MessageBubble";
import TypingIndicator  from "../components/TypingIndicator";
import MicButton        from "../components/MicButton";
import SpeakerButton    from "../components/SpeakerButton";
import ChatHistoryDrawer from "../components/ChatHistoryDrawer";
import { RootStackParamList } from "../navigation/AppNavigator";

type NavProp    = NativeStackNavigationProp<RootStackParamList, "Chat">;
type RoutePropT = RouteProp<RootStackParamList, "Chat">;

// Model display names
const MODEL_LABELS: Record<string, string> = {
  llama3:   "Llama 3.1 8B",
  mistral:  "Mistral 7B",
  deepseek: "DeepSeek R1",
};

export default function ChatScreen() {
  const navigation = useNavigation<NavProp>();
  const route      = useRoute<RoutePropT>();
  const paramConvId    = route.params?.conversationId;
  const paramDocId     = route.params?.documentId;
  const paramDocName   = route.params?.documentName;

  const {
    currentConversation,
    conversations,
    models,
    selectedModel,
    isStreaming,
    streamingContent,
    fetchConversation,
    fetchConversations,
    fetchModels,
    sendMessage,
    sendRagMessage,
    deleteConversation,
    clearCurrent,
    setSelectedModel,
  } = useChatStore();

  const [input,          setInput]          = useState("");
  const [convId,         setConvId]         = useState<string | undefined>(paramConvId);
  const [showHistory,    setShowHistory]    = useState(false);
  const [showModelMenu,  setShowModelMenu]  = useState(false);
  const flatListRef = useRef<FlatListType<Message>>(null);

  // ── Init ────────────────────────────────────────────────────────
  useEffect(() => {
    fetchModels();
    fetchConversations();
    if (paramConvId) {
      fetchConversation(paramConvId);
      setConvId(paramConvId);
    } else {
      clearCurrent();
      setConvId(undefined);
    }
  }, [paramConvId]);

  // ── Auto-select the only online model ───────────────────────────
  useEffect(() => {
    if (models.length === 0) return;
    const onlineModels = models.filter((m) => m.status === "online");
    const currentOnline = onlineModels.find((m) => m.name === selectedModel);
    if (!currentOnline && onlineModels.length === 1) {
      setSelectedModel(onlineModels[0].name);
    }
  }, [models]);

  // ── Display messages ────────────────────────────────────────────
  const messages: Message[] = currentConversation?.messages || [];

  // ── Auto-scroll ─────────────────────────────────────────────────
  useEffect(() => {
    if (messages.length > 0 || isStreaming) {
      setTimeout(() => flatListRef.current?.scrollToEnd({ animated: true }), 80);
    }
  }, [messages.length, streamingContent]);

  // ── Send ────────────────────────────────────────────────────────
  const handleSend = async () => {
    const text = input.trim();
    if (!text || isStreaming) return;
    setInput("");

    if (!currentConversation) {
      useChatStore.setState({
        currentConversation: {
          id: "temp",
          model_name: selectedModel,
          title: text.slice(0, 50),
          rag_enabled: !!paramDocId,
          created_at: new Date().toISOString(),
          messages: [],
        },
      });
    }

    try {
      if (paramDocId) {
        const newId = await sendRagMessage(text, paramDocId, convId);
        if (newId && newId !== "temp") setConvId(newId);
      } else {
        const newId = await sendMessage(text, convId);
        if (newId && newId !== "temp") setConvId(newId);
      }
    } catch (err: any) {
      const msg = err.response?.data?.detail || err.message || "Failed to send message.";
      Alert.alert("Send Failed", msg);
    }
  };

  // ── History actions ──────────────────────────────────────────────
  const handleSelectHistory = (id: string) => {
    setConvId(id);
    fetchConversation(id);
  };

  const handleNewChat = () => {
    clearCurrent();
    setConvId(undefined);
    setInput("");
  };

  const handleDeleteFromHistory = (id: string) => {
    Alert.alert("Delete Chat", "Remove this conversation?", [
      { text: "Cancel", style: "cancel" },
      { text: "Delete", style: "destructive", onPress: () => deleteConversation(id) },
    ]);
  };

  // ── Header title ────────────────────────────────────────────────
  const headerTitle = currentConversation?.title || "New Chat";
  const modelLabel  = MODEL_LABELS[selectedModel] ?? selectedModel;

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: "#15152A" }} edges={["top"]}>
      <KeyboardAvoidingView
        style={{ flex: 1 }}
        behavior={Platform.OS === "ios" ? "padding" : "height"}
        keyboardVerticalOffset={Platform.OS === "ios" ? 0 : 24}
      >

        {/* ── TOP BAR ──────────────────────────────────────────── */}
        <View
          style={{
            flexDirection: "row",
            alignItems: "center",
            paddingHorizontal: 12,
            paddingVertical: 10,
            borderBottomWidth: 1,
            borderBottomColor: "#2A2A3E",
            backgroundColor: "#1E1E2E",
          }}
        >
          {/* Back */}
          <TouchableOpacity
            onPress={() => navigation.goBack()}
            style={{ padding: 6, marginRight: 4 }}
            hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
          >
            <Ionicons name="arrow-back" size={22} color="#E5E7EB" />
          </TouchableOpacity>

          {/* Title + Model dropdown / RAG indicator */}
          <TouchableOpacity
            style={{ flex: 1, marginLeft: 4 }}
            onPress={() => !paramDocId && setShowModelMenu(true)}
            activeOpacity={paramDocId ? 1 : 0.7}
          >
            <Text style={{ color: "#fff", fontSize: 15, fontWeight: "700" }} numberOfLines={1}>
              {paramDocId ? (paramDocName ?? "Document Chat") : headerTitle}
            </Text>
            {paramDocId ? (
              <View style={{ flexDirection: "row", alignItems: "center", marginTop: 2 }}>
                <View style={{ backgroundColor: "#2D1F5E", borderRadius: 6, paddingHorizontal: 6, paddingVertical: 2, flexDirection: "row", alignItems: "center" }}>
                  <Ionicons name="document-text" size={10} color="#A78BFA" style={{ marginRight: 3 }} />
                  <Text style={{ color: "#A78BFA", fontSize: 10, fontWeight: "600" }}>RAG MODE</Text>
                </View>
                <Text style={{ color: "#6B7280", fontSize: 10, marginLeft: 6 }}>{modelLabel}</Text>
              </View>
            ) : (
              <View style={{ flexDirection: "row", alignItems: "center", marginTop: 1 }}>
                <View style={{ width: 7, height: 7, borderRadius: 4, backgroundColor: "#22C55E", marginRight: 5 }} />
                <Text style={{ color: "#9CA3AF", fontSize: 11 }}>{modelLabel}</Text>
                <Ionicons name="chevron-down" size={11} color="#6B7280" style={{ marginLeft: 3 }} />
              </View>
            )}
          </TouchableOpacity>

          {/* History icon */}
          <TouchableOpacity
            onPress={() => setShowHistory(true)}
            style={{
              padding: 8,
              marginLeft: 4,
              backgroundColor: "#2A2A3E",
              borderRadius: 20,
            }}
            hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
          >
            <Ionicons name="time-outline" size={20} color="#9CA3AF" />
          </TouchableOpacity>
        </View>

        {/* ── MODEL PICKER MODAL ───────────────────────────────── */}
        <Modal visible={showModelMenu} transparent animationType="fade">
          <TouchableOpacity
            style={{ flex: 1, backgroundColor: "#00000080", justifyContent: "center", paddingHorizontal: 32 }}
            activeOpacity={1}
            onPress={() => setShowModelMenu(false)}
          >
            <View style={{ backgroundColor: "#1E1E2E", borderRadius: 16, overflow: "hidden", borderWidth: 1, borderColor: "#2A2A3E" }}>
              <Text style={{ color: "#9CA3AF", fontSize: 11, fontWeight: "700", paddingHorizontal: 18, paddingTop: 14, paddingBottom: 8, letterSpacing: 1 }}>
                SELECT MODEL
              </Text>
              {(models.length > 0
                ? models
                : [
                    { name: "llama3",   display_name: "Llama 3.1 8B",  status: "offline" },
                    { name: "mistral",  display_name: "Mistral 7B",    status: "offline" },
                    { name: "deepseek", display_name: "DeepSeek R1",   status: "offline" },
                  ]
              ).map((m) => {
                const active = m.name === selectedModel;
                const online = m.status === "online";
                return (
                  <TouchableOpacity
                    key={m.name}
                    onPress={() => {
                      if (!online) {
                        const onlineNames = (models.length > 0 ? models : []).filter(x => x.status === "online").map(x => x.display_name).join(", ");
                        Alert.alert(
                          "Model Offline",
                          `${m.display_name} is not running.${onlineNames ? `\n\nCurrently online: ${onlineNames}` : "\n\nNo models are currently online. Run start_models.bat first."}`,
                        );
                        return;
                      }
                      setSelectedModel(m.name);
                      setShowModelMenu(false);
                    }}
                    style={{
                      flexDirection: "row",
                      alignItems: "center",
                      paddingHorizontal: 18,
                      paddingVertical: 13,
                      backgroundColor: active ? "#2D1F5E" : "transparent",
                      borderTopWidth: 1,
                      borderTopColor: "#2A2A3E",
                      opacity: online ? 1 : 0.5,
                    }}
                  >
                    <Ionicons name="hardware-chip-outline" size={18} color={active ? "#7C3AED" : "#6B7280"} />
                    <View style={{ flex: 1, marginLeft: 12 }}>
                      <Text style={{ color: active ? "#E9D5FF" : "#D1D5DB", fontSize: 14, fontWeight: active ? "600" : "400" }}>
                        {m.display_name}
                      </Text>
                      {!online && (
                        <Text style={{ color: "#EF4444", fontSize: 11, marginTop: 1 }}>offline</Text>
                      )}
                    </View>
                    <View style={{ flexDirection: "row", alignItems: "center" }}>
                      <View style={{ width: 6, height: 6, borderRadius: 3, backgroundColor: online ? "#22C55E" : "#EF4444", marginRight: 5 }} />
                      {active && <Ionicons name="checkmark" size={16} color="#7C3AED" />}
                    </View>
                  </TouchableOpacity>
                );
              })}
              <View style={{ height: 8 }} />
            </View>
          </TouchableOpacity>
        </Modal>

        {/* ── MESSAGES ─────────────────────────────────────────── */}
        <FlatList
          ref={flatListRef}
          data={messages}
          keyExtractor={(item) => item.id}
          renderItem={({ item }) => <MessageBubble message={item} />}
          contentContainerStyle={{ paddingTop: 12, paddingBottom: 8 }}
          showsVerticalScrollIndicator={false}
          ListEmptyComponent={
            <View style={{ flex: 1, alignItems: "center", paddingTop: 80 }}>
              <Ionicons name="chatbubble-ellipses-outline" size={52} color="#2A2A3E" />
              <Text style={{ color: "#4B5563", fontSize: 15, marginTop: 16, fontWeight: "500" }}>
                Start a conversation
              </Text>
              <Text style={{ color: "#374151", fontSize: 13, marginTop: 6 }}>
                Using {modelLabel}
              </Text>
            </View>
          }
          ListFooterComponent={isStreaming ? (
            <View style={{ paddingHorizontal: 12 }}>
              {/* Streaming bubble — builds token by token */}
              {streamingContent ? (
                <View style={{ alignItems: "flex-start", marginBottom: 6 }}>
                  <View style={{ backgroundColor: "#2A2A3E", borderRadius: 18, borderBottomLeftRadius: 4, paddingHorizontal: 14, paddingTop: 9, paddingBottom: 7, maxWidth: "80%" }}>
                    <Text style={{ color: "#E5E7EB", fontSize: 14, lineHeight: 20 }}>
                      {streamingContent.replace(/<think>[\s\S]*?<\/think>/g, "").replace(/<think>[\s\S]*/g, "").trim() || "..."}
                    </Text>
                  </View>
                </View>
              ) : (
                <TypingIndicator />
              )}
            </View>
          ) : null}
        />

        {/* ── INPUT BAR ────────────────────────────────────────── */}
        <View
          style={{
            flexDirection: "row",
            alignItems: "flex-end",
            paddingHorizontal: 10,
            paddingVertical: 10,
            paddingBottom: Platform.OS === "android" ? 14 : 10,
            borderTopWidth: 1,
            borderTopColor: "#2A2A3E",
            backgroundColor: "#1E1E2E",
            gap: 8,
          }}
        >
          {/* Text input */}
          <View
            style={{
              flex: 1,
              backgroundColor: "#2A2A3E",
              borderRadius: 22,
              paddingHorizontal: 16,
              paddingVertical: Platform.OS === "ios" ? 10 : 7,
              maxHeight: 120,
              justifyContent: "center",
            }}
          >
            <TextInput
              style={{ color: "#fff", fontSize: 14, lineHeight: 20, maxHeight: 100 }}
              placeholder="Message..."
              placeholderTextColor="#6B7280"
              multiline
              value={input}
              onChangeText={setInput}
              editable={!isStreaming}
              onSubmitEditing={Platform.OS !== "web" ? handleSend : undefined}
              blurOnSubmit={false}
              onKeyPress={(e: any) => {
                // Web: Enter sends; Shift+Enter inserts newline
                if (Platform.OS === "web" && e.nativeEvent.key === "Enter" && !e.nativeEvent.shiftKey) {
                  e.preventDefault?.();
                  handleSend();
                }
              }}
            />
          </View>

          {/* Mic — voice input (STT) */}
          <MicButton onTranscription={(t) => setInput((prev) => prev ? `${prev} ${t}` : t)} />

          {/* Speaker — toggle voice reply (TTS) */}
          <SpeakerButton />

          {/* Send button */}
          <TouchableOpacity
            onPress={handleSend}
            disabled={!input.trim() || isStreaming}
            style={{
              width: 42,
              height: 42,
              borderRadius: 21,
              backgroundColor: input.trim() && !isStreaming ? "#7C3AED" : "#2A2A3E",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <Ionicons
              name="send"
              size={17}
              color={input.trim() && !isStreaming ? "#fff" : "#4B5563"}
            />
          </TouchableOpacity>
        </View>

      </KeyboardAvoidingView>

      {/* ── HISTORY DRAWER (50% height) ──────────────────────── */}
      <ChatHistoryDrawer
        visible={showHistory}
        conversations={conversations}
        currentId={convId}
        onSelect={handleSelectHistory}
        onNewChat={handleNewChat}
        onDelete={handleDeleteFromHistory}
        onClose={() => setShowHistory(false)}
      />
    </SafeAreaView>
  );
}
