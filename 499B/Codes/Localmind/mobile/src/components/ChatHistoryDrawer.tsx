import React, { useEffect, useRef } from "react";
import {
  Animated,
  Dimensions,
  FlatList,
  Modal,
  Text,
  TouchableOpacity,
  TouchableWithoutFeedback,
  View,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { Conversation } from "../store/useChatStore";

interface Props {
  visible: boolean;
  conversations: Conversation[];
  currentId?: string;
  onSelect: (id: string) => void;
  onNewChat: () => void;
  onDelete: (id: string) => void;
  onClose: () => void;
}

const SCREEN_HEIGHT = Dimensions.get("window").height;
const DRAWER_HEIGHT = SCREEN_HEIGHT * 0.52;

export default function ChatHistoryDrawer({
  visible,
  conversations,
  currentId,
  onSelect,
  onNewChat,
  onDelete,
  onClose,
}: Props) {
  const translateY = useRef(new Animated.Value(DRAWER_HEIGHT)).current;

  useEffect(() => {
    Animated.spring(translateY, {
      toValue: visible ? 0 : DRAWER_HEIGHT,
      useNativeDriver: true,
      bounciness: 4,
      speed: 16,
    }).start();
  }, [visible]);

  const renderItem = ({ item }: { item: Conversation }) => {
    const isActive = item.id === currentId;
    return (
      <TouchableOpacity
        onPress={() => { onSelect(item.id); onClose(); }}
        onLongPress={() => onDelete(item.id)}
        style={{
          flexDirection: "row",
          alignItems: "center",
          paddingHorizontal: 20,
          paddingVertical: 13,
          borderBottomWidth: 1,
          borderBottomColor: "#1E1E2E",
          backgroundColor: isActive ? "#2D1F5E" : "transparent",
        }}
        activeOpacity={0.7}
      >
        {/* Active indicator */}
        <View
          style={{
            width: 4,
            height: 36,
            borderRadius: 2,
            backgroundColor: isActive ? "#7C3AED" : "transparent",
            marginRight: 12,
          }}
        />

        {/* Icon */}
        <View
          style={{
            width: 36,
            height: 36,
            borderRadius: 18,
            backgroundColor: isActive ? "#7C3AED22" : "#2A2A3E",
            alignItems: "center",
            justifyContent: "center",
            marginRight: 12,
          }}
        >
          <Ionicons
            name="chatbubble-ellipses-outline"
            size={17}
            color={isActive ? "#7C3AED" : "#666"}
          />
        </View>

        {/* Text */}
        <View style={{ flex: 1 }}>
          <Text
            numberOfLines={1}
            style={{
              color: isActive ? "#E9D5FF" : "#D1D5DB",
              fontSize: 14,
              fontWeight: isActive ? "600" : "400",
            }}
          >
            {item.title}
          </Text>
          <Text style={{ color: "#6B7280", fontSize: 11, marginTop: 2 }}>
            {item.model_name} •{" "}
            {new Date(item.created_at).toLocaleDateString(undefined, {
              month: "short",
              day: "numeric",
            })}
            {item.rag_enabled ? " • RAG" : ""}
          </Text>
        </View>

        <Ionicons name="chevron-forward" size={15} color="#444" />
      </TouchableOpacity>
    );
  };

  return (
    <Modal
      visible={visible}
      transparent
      animationType="none"
      onRequestClose={onClose}
    >
      {/* Scrim */}
      <TouchableWithoutFeedback onPress={onClose}>
        <View style={{ flex: 1, backgroundColor: "#00000066" }} />
      </TouchableWithoutFeedback>

      {/* Drawer */}
      <Animated.View
        style={{
          position: "absolute",
          bottom: 0,
          left: 0,
          right: 0,
          height: DRAWER_HEIGHT,
          backgroundColor: "#1E1E2E",
          borderTopLeftRadius: 20,
          borderTopRightRadius: 20,
          transform: [{ translateY }],
          overflow: "hidden",
        }}
      >
        {/* Handle */}
        <View style={{ alignItems: "center", paddingTop: 10, paddingBottom: 4 }}>
          <View style={{ width: 36, height: 4, borderRadius: 2, backgroundColor: "#3F3F5F" }} />
        </View>

        {/* Header row */}
        <View
          style={{
            flexDirection: "row",
            alignItems: "center",
            justifyContent: "space-between",
            paddingHorizontal: 20,
            paddingVertical: 10,
            borderBottomWidth: 1,
            borderBottomColor: "#2A2A3E",
          }}
        >
          <Text style={{ color: "#fff", fontSize: 16, fontWeight: "700" }}>
            Chat History
          </Text>
          <TouchableOpacity
            onPress={() => { onNewChat(); onClose(); }}
            style={{
              flexDirection: "row",
              alignItems: "center",
              backgroundColor: "#7C3AED",
              paddingHorizontal: 14,
              paddingVertical: 7,
              borderRadius: 20,
            }}
          >
            <Ionicons name="add" size={16} color="#fff" />
            <Text style={{ color: "#fff", fontSize: 13, marginLeft: 4, fontWeight: "600" }}>
              New Chat
            </Text>
          </TouchableOpacity>
        </View>

        {/* List */}
        <FlatList
          data={conversations}
          keyExtractor={(c) => c.id}
          renderItem={renderItem}
          showsVerticalScrollIndicator={false}
          ListEmptyComponent={
            <View style={{ alignItems: "center", paddingTop: 40, paddingHorizontal: 24 }}>
              <Ionicons name="chatbubbles-outline" size={44} color="#2A2A3E" />
              <Text style={{ color: "#6B7280", marginTop: 14, fontSize: 15, fontWeight: "600" }}>
                No previous chats
              </Text>
              <Text style={{ color: "#4B5563", marginTop: 6, fontSize: 13, textAlign: "center" }}>
                Start a new conversation above and it will appear here
              </Text>
            </View>
          }
        />
      </Animated.View>
    </Modal>
  );
}
