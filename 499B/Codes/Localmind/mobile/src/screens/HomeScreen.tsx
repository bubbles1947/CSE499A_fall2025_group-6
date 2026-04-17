import React, { useCallback, useRef } from "react";
import {
  Alert,
  Animated,
  FlatList,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";
import { useFocusEffect, useNavigation } from "@react-navigation/native";
import { NativeStackNavigationProp } from "@react-navigation/native-stack";

import useAuthStore from "../store/useAuthStore";
import useChatStore, { Conversation } from "../store/useChatStore";
import { RootStackParamList } from "../navigation/AppNavigator";

type NavProp = NativeStackNavigationProp<RootStackParamList>;

// ── Skeleton row ──────────────────────────────────────────────────
function SkeletonRow() {
  const opacity = useRef(new Animated.Value(0.3)).current;

  React.useEffect(() => {
    const anim = Animated.loop(
      Animated.sequence([
        Animated.timing(opacity, { toValue: 0.8, duration: 700, useNativeDriver: true }),
        Animated.timing(opacity, { toValue: 0.3, duration: 700, useNativeDriver: true }),
      ])
    );
    anim.start();
    return () => anim.stop();
  }, []);

  return (
    <Animated.View
      style={{
        backgroundColor: "#1E1E2E",
        borderRadius: 14,
        padding: 14,
        marginBottom: 10,
        opacity,
      }}
    >
      <View style={{ height: 14, width: "60%", backgroundColor: "#2A2A3E", borderRadius: 7 }} />
      <View style={{ height: 11, width: "35%", backgroundColor: "#2A2A3E", borderRadius: 6, marginTop: 8 }} />
    </Animated.View>
  );
}

// ── Model display names ───────────────────────────────────────────
const MODEL_LABELS: Record<string, string> = {
  llama3:   "Llama 3.1 8B",
  mistral:  "Mistral 7B",
  deepseek: "DeepSeek R1",
};

export default function HomeScreen() {
  const navigation = useNavigation<NavProp>();
  const { user, logout } = useAuthStore();
  const {
    conversations,
    models,
    selectedModel,
    fetchConversations,
    fetchModels,
    deleteConversation,
    clearCurrent,
    isLoadingConversations,
  } = useChatStore();

  useFocusEffect(
    useCallback(() => {
      fetchConversations();
      fetchModels();
    }, [])
  );

  const handleNewChat = () => {
    clearCurrent();
    navigation.navigate("Chat");
  };

  const handleOpenChat = (id: string) => {
    navigation.navigate("Chat", { conversationId: id });
  };

  const handleDelete = (id: string) => {
    Alert.alert("Delete Chat", "Are you sure?", [
      { text: "Cancel", style: "cancel" },
      { text: "Delete", style: "destructive", onPress: () => deleteConversation(id) },
    ]);
  };

  // Online model badge
  const onlineModel = models.find((m) => m.name === selectedModel && m.status === "online")
    || models.find((m) => m.status === "online");
  const modelLabel = onlineModel
    ? (MODEL_LABELS[onlineModel.name] ?? onlineModel.display_name)
    : null;

  const renderConversation = ({ item }: { item: Conversation }) => (
    <TouchableOpacity
      onPress={() => handleOpenChat(item.id)}
      onLongPress={() => handleDelete(item.id)}
      activeOpacity={0.7}
      style={{
        backgroundColor: "#1E1E2E",
        borderRadius: 14,
        paddingHorizontal: 14,
        paddingVertical: 13,
        marginBottom: 10,
        flexDirection: "row",
        alignItems: "center",
        borderWidth: 1,
        borderColor: "#2A2A3E",
      }}
    >
      {/* Icon */}
      <View style={{ width: 38, height: 38, borderRadius: 19, backgroundColor: "#2D1F5E", alignItems: "center", justifyContent: "center", marginRight: 12 }}>
        <Ionicons name={item.rag_enabled ? "document-text" : "chatbubble-ellipses-outline"} size={17} color="#7C3AED" />
      </View>

      {/* Text */}
      <View style={{ flex: 1 }}>
        <Text style={{ color: "#E5E7EB", fontSize: 14, fontWeight: "500" }} numberOfLines={1}>
          {item.title}
        </Text>
        <Text style={{ color: "#6B7280", fontSize: 12, marginTop: 3 }}>
          {MODEL_LABELS[item.model_name] ?? item.model_name}
          {" · "}
          {new Date(item.created_at).toLocaleDateString(undefined, { month: "short", day: "numeric" })}
          {item.rag_enabled ? " · RAG" : ""}
        </Text>
      </View>

      <Ionicons name="chevron-forward" size={16} color="#374151" />
    </TouchableOpacity>
  );

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: "#15152A" }} edges={["top"]}>
      <View style={{ flex: 1, paddingHorizontal: 20 }}>

        {/* Header */}
        <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginTop: 16, marginBottom: 24 }}>
          <View>
            <Text style={{ color: "#6B7280", fontSize: 13 }}>Welcome back,</Text>
            <Text style={{ color: "#fff", fontSize: 22, fontWeight: "700", marginTop: 1 }}>
              {user?.username || "User"}
            </Text>
          </View>
          <TouchableOpacity
            onPress={logout}
            style={{ width: 40, height: 40, borderRadius: 20, backgroundColor: "#1E1E2E", alignItems: "center", justifyContent: "center", borderWidth: 1, borderColor: "#2A2A3E" }}
          >
            <Ionicons name="log-out-outline" size={20} color="#EF4444" />
          </TouchableOpacity>
        </View>

        {/* Model status badge */}
        {modelLabel && (
          <View style={{ flexDirection: "row", alignItems: "center", backgroundColor: "#1E1E2E", borderRadius: 10, paddingHorizontal: 12, paddingVertical: 8, marginBottom: 20, borderWidth: 1, borderColor: "#2A2A3E" }}>
            <View style={{ width: 8, height: 8, borderRadius: 4, backgroundColor: "#22C55E", marginRight: 8 }} />
            <Text style={{ color: "#9CA3AF", fontSize: 13 }}>{modelLabel} is online</Text>
          </View>
        )}

        {/* New Chat Button */}
        <TouchableOpacity
          onPress={handleNewChat}
          style={{
            backgroundColor: "#7C3AED",
            borderRadius: 14,
            paddingVertical: 14,
            alignItems: "center",
            marginBottom: 28,
            flexDirection: "row",
            justifyContent: "center",
            shadowColor: "#7C3AED",
            shadowOffset: { width: 0, height: 4 },
            shadowOpacity: 0.4,
            shadowRadius: 8,
            elevation: 6,
          }}
        >
          <Ionicons name="add-circle-outline" size={22} color="#fff" />
          <Text style={{ color: "#fff", fontSize: 16, fontWeight: "600", marginLeft: 8 }}>
            New Chat
          </Text>
        </TouchableOpacity>

        {/* Conversations header */}
        <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
          <Text style={{ color: "#9CA3AF", fontSize: 13, fontWeight: "600", letterSpacing: 0.5 }}>
            RECENT CHATS
          </Text>
          {conversations.length > 0 && (
            <Text style={{ color: "#4B5563", fontSize: 12 }}>{conversations.length} total</Text>
          )}
        </View>

        {/* Conversation list / skeleton / empty */}
        {isLoadingConversations && conversations.length === 0 ? (
          <View>
            <SkeletonRow />
            <SkeletonRow />
            <SkeletonRow />
          </View>
        ) : (
          <FlatList
            data={conversations}
            keyExtractor={(item) => item.id}
            renderItem={renderConversation}
            showsVerticalScrollIndicator={false}
            ListEmptyComponent={
              <View style={{ alignItems: "center", paddingTop: 60 }}>
                <View style={{ width: 72, height: 72, borderRadius: 36, backgroundColor: "#1E1E2E", alignItems: "center", justifyContent: "center", marginBottom: 16 }}>
                  <Ionicons name="chatbubbles-outline" size={34} color="#374151" />
                </View>
                <Text style={{ color: "#6B7280", fontSize: 16, fontWeight: "600" }}>
                  No conversations yet
                </Text>
                <Text style={{ color: "#4B5563", fontSize: 13, marginTop: 6, textAlign: "center" }}>
                  Tap "New Chat" to start your{"\n"}first AI conversation
                </Text>
              </View>
            }
          />
        )}
      </View>
    </SafeAreaView>
  );
}
