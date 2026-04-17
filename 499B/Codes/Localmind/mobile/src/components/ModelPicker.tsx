import React, { useEffect, useState } from "react";
import { ActivityIndicator, Modal, Text, TouchableOpacity, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import useChatStore, { ModelInfo } from "../store/useChatStore";

const MODEL_LABELS: Record<string, string> = {
  llama3:   "Llama 3.1 8B",
  mistral:  "Mistral 7B",
  deepseek: "DeepSeek R1",
};

export default function ModelPicker() {
  const { models, selectedModel, setSelectedModel, fetchModels } = useChatStore();
  const [visible, setVisible] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    fetchModels().finally(() => setLoading(false));
  }, []);

  const selected = models.find((m) => m.name === selectedModel);
  const isOnline = selected?.status === "online";

  return (
    <>
      {/* Picker trigger */}
      <TouchableOpacity
        onPress={() => setVisible(true)}
        style={{
          backgroundColor: "#1E1E2E",
          borderRadius: 14,
          paddingHorizontal: 14,
          paddingVertical: 12,
          flexDirection: "row",
          alignItems: "center",
          justifyContent: "space-between",
          borderWidth: 1,
          borderColor: "#2A2A3E",
        }}
      >
        <View style={{ flexDirection: "row", alignItems: "center" }}>
          <View style={{ width: 36, height: 36, borderRadius: 10, backgroundColor: "#2D1F5E", alignItems: "center", justifyContent: "center", marginRight: 12 }}>
            <Ionicons name="hardware-chip" size={18} color="#7C3AED" />
          </View>
          <View>
            <Text style={{ color: "#E5E7EB", fontSize: 14, fontWeight: "500" }}>
              {loading ? "Loading models…" : (selected?.display_name || "Select Model")}
            </Text>
            {!loading && selected && (
              <View style={{ flexDirection: "row", alignItems: "center", marginTop: 2 }}>
                <View style={{ width: 6, height: 6, borderRadius: 3, backgroundColor: isOnline ? "#22C55E" : "#EF4444", marginRight: 5 }} />
                <Text style={{ color: isOnline ? "#22C55E" : "#EF4444", fontSize: 11 }}>
                  {isOnline ? "online" : "offline"}
                </Text>
              </View>
            )}
          </View>
        </View>
        {loading
          ? <ActivityIndicator size="small" color="#7C3AED" />
          : <Ionicons name="chevron-down" size={18} color="#6B7280" />
        }
      </TouchableOpacity>

      {/* Model picker modal */}
      <Modal visible={visible} transparent animationType="slide">
        <TouchableOpacity
          style={{ flex: 1, backgroundColor: "#00000066" }}
          activeOpacity={1}
          onPress={() => setVisible(false)}
        >
          <View style={{ marginTop: "auto", backgroundColor: "#1E1E2E", borderTopLeftRadius: 24, borderTopRightRadius: 24, paddingBottom: 32 }}>
            {/* Handle */}
            <View style={{ alignItems: "center", paddingTop: 12, paddingBottom: 6 }}>
              <View style={{ width: 36, height: 4, borderRadius: 2, backgroundColor: "#3F3F5F" }} />
            </View>

            <Text style={{ color: "#9CA3AF", fontSize: 11, fontWeight: "700", letterSpacing: 1, paddingHorizontal: 20, paddingBottom: 12 }}>
              SELECT MODEL
            </Text>

            {(models.length > 0
              ? models
              : [
                  { name: "llama3",   display_name: "Llama 3.1 8B",  status: "offline" },
                  { name: "mistral",  display_name: "Mistral 7B",    status: "offline" },
                  { name: "deepseek", display_name: "DeepSeek R1",   status: "offline" },
                ] as ModelInfo[]
            ).map((m) => {
              const active  = m.name === selectedModel;
              const online  = m.status === "online";
              return (
                <TouchableOpacity
                  key={m.name}
                  onPress={() => {
                    if (!online) return;
                    setSelectedModel(m.name);
                    setVisible(false);
                  }}
                  style={{
                    flexDirection: "row",
                    alignItems: "center",
                    paddingHorizontal: 20,
                    paddingVertical: 14,
                    backgroundColor: active ? "#2D1F5E" : "transparent",
                    borderTopWidth: 1,
                    borderTopColor: "#2A2A3E",
                    opacity: online ? 1 : 0.45,
                  }}
                >
                  <View style={{ width: 38, height: 38, borderRadius: 10, backgroundColor: active ? "#3D1F7E" : "#2A2A3E", alignItems: "center", justifyContent: "center", marginRight: 14 }}>
                    <Ionicons name="hardware-chip-outline" size={19} color={active ? "#A78BFA" : "#6B7280"} />
                  </View>
                  <View style={{ flex: 1 }}>
                    <Text style={{ color: active ? "#E9D5FF" : "#D1D5DB", fontSize: 15, fontWeight: active ? "600" : "400" }}>
                      {m.display_name}
                    </Text>
                    {!online && (
                      <Text style={{ color: "#EF4444", fontSize: 11, marginTop: 1 }}>offline — start the server first</Text>
                    )}
                  </View>
                  <View style={{ flexDirection: "row", alignItems: "center" }}>
                    <View style={{ width: 7, height: 7, borderRadius: 4, backgroundColor: online ? "#22C55E" : "#EF4444", marginRight: 6 }} />
                    {active && <Ionicons name="checkmark-circle" size={18} color="#7C3AED" />}
                  </View>
                </TouchableOpacity>
              );
            })}
          </View>
        </TouchableOpacity>
      </Modal>
    </>
  );
}
