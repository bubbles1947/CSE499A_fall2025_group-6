import React from "react";
import { View, Text, TouchableOpacity } from "react-native";
import { Ionicons } from "@expo/vector-icons";

interface Document {
  id: string;
  filename: string;
  status: string;
  uploaded_at: string;
}

interface Props {
  document: Document;
  onDelete: (id: string) => void;
  onChat: (id: string) => void;
}

export default function DocumentCard({ document, onDelete, onChat }: Props) {
  const statusColor =
    document.status === "ready"
      ? "text-green-500"
      : document.status === "processing"
      ? "text-yellow-500"
      : "text-red-500";

  const statusIcon =
    document.status === "ready"
      ? "checkmark-circle"
      : document.status === "processing"
      ? "time"
      : "alert-circle";

  return (
    <View className="bg-dark-100 rounded-xl px-4 py-4 mb-3">
      <View className="flex-row items-start justify-between">
        {/* File info */}
        <View className="flex-row items-center flex-1 mr-3">
          <Ionicons name="document-text" size={28} color="#6C63FF" />
          <View className="ml-3 flex-1">
            <Text className="text-white text-base font-medium" numberOfLines={1}>
              {document.filename}
            </Text>
            <View className="flex-row items-center mt-1">
              <Ionicons name={statusIcon as any} size={14} color={
                document.status === "ready" ? "#22C55E" :
                document.status === "processing" ? "#EAB308" : "#EF4444"
              } />
              <Text className={`text-xs ml-1 capitalize ${statusColor}`}>
                {document.status}
              </Text>
              <Text className="text-gray-600 text-xs ml-2">
                {new Date(document.uploaded_at).toLocaleDateString()}
              </Text>
            </View>
          </View>
        </View>

        {/* Actions */}
        <View className="flex-row items-center">
          {document.status === "ready" && (
            <TouchableOpacity
              className="bg-primary/20 rounded-full p-2 mr-2"
              onPress={() => onChat(document.id)}
            >
              <Ionicons name="chatbubble-ellipses" size={18} color="#6C63FF" />
            </TouchableOpacity>
          )}
          <TouchableOpacity
            className="bg-red-500/20 rounded-full p-2"
            onPress={() => onDelete(document.id)}
          >
            <Ionicons name="trash" size={18} color="#EF4444" />
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );
}
