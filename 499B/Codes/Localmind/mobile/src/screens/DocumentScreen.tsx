import React, { useCallback, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  FlatList,
  Platform,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";
import * as DocumentPicker from "expo-document-picker";
import { useFocusEffect, useNavigation } from "@react-navigation/native";
import { NativeStackNavigationProp } from "@react-navigation/native-stack";

import api from "../services/api";
import { RootStackParamList } from "../navigation/AppNavigator";

interface Document {
  id: string;
  filename: string;
  chroma_collection: string;
  status: "processing" | "ready" | "error";
  uploaded_at: string;
}

type NavProp = NativeStackNavigationProp<RootStackParamList>;

export default function DocumentScreen() {
  const navigation = useNavigation<NavProp>();
  const [documents, setDocuments] = useState<Document[]>([]);
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(true);

  const fetchDocuments = async () => {
    try {
      const res = await api.get("/rag/documents");
      setDocuments(res.data);
    } catch {
      Alert.alert("Error", "Failed to load documents.");
    } finally {
      setLoading(false);
    }
  };

  useFocusEffect(
    useCallback(() => {
      fetchDocuments();
    }, [])
  );

  const handleUpload = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: "application/pdf",
        copyToCacheDirectory: true,
      });
      if (result.canceled) return;

      const file = result.assets[0];
      setUploading(true);

      const formData = new FormData();
      formData.append("file", {
        uri: file.uri,
        name: file.name,
        type: "application/pdf",
      } as unknown as Blob);

      const res = await api.post("/rag/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 120000,
      });

      setDocuments((prev) => [res.data, ...prev]);
      Alert.alert("Success", `"${file.name}" processed and ready.`);
    } catch (err: any) {
      const msg = err.response?.data?.detail || err.message || "Upload failed.";
      Alert.alert("Upload Error", msg);
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = (id: string) => {
    Alert.alert("Delete Document", "This will remove the document and its embeddings.", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Delete",
        style: "destructive",
        onPress: async () => {
          try {
            await api.delete(`/rag/documents/${id}`);
            setDocuments((prev) => prev.filter((d) => d.id !== id));
          } catch {
            Alert.alert("Error", "Failed to delete document.");
          }
        },
      },
    ]);
  };

  const handleChat = (doc: Document) => {
    navigation.navigate("Chat", {
      documentId: doc.id,
      documentName: doc.filename,
    });
  };

  const statusColor = (s: string) =>
    s === "ready" ? "#22C55E" : s === "processing" ? "#F59E0B" : "#EF4444";

  const statusIcon = (s: string): any =>
    s === "ready" ? "checkmark-circle" : s === "processing" ? "time" : "alert-circle";

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: "#15152A" }} edges={["top"]}>
      <View style={{ flex: 1, paddingHorizontal: 20 }}>

        {/* Header */}
        <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginTop: 16, marginBottom: 24 }}>
          <View>
            <Text style={{ color: "#fff", fontSize: 20, fontWeight: "700" }}>Documents</Text>
            <Text style={{ color: "#6B7280", fontSize: 13, marginTop: 2 }}>Upload PDFs for RAG</Text>
          </View>
          <View style={{ backgroundColor: "#2A2A3E", borderRadius: 20, paddingHorizontal: 12, paddingVertical: 4 }}>
            <Text style={{ color: "#7C3AED", fontSize: 13, fontWeight: "600" }}>
              {documents.length} file{documents.length !== 1 ? "s" : ""}
            </Text>
          </View>
        </View>

        {/* Upload Button */}
        <TouchableOpacity
          onPress={handleUpload}
          disabled={uploading}
          style={{
            backgroundColor: uploading ? "#4C1D95" : "#7C3AED",
            borderRadius: 14,
            paddingVertical: 14,
            alignItems: "center",
            marginBottom: 20,
            flexDirection: "row",
            justifyContent: "center",
          }}
        >
          {uploading ? (
            <>
              <ActivityIndicator color="#fff" size="small" />
              <Text style={{ color: "#fff", fontSize: 15, fontWeight: "600", marginLeft: 8 }}>
                Uploading & Processing...
              </Text>
            </>
          ) : (
            <>
              <Ionicons name="cloud-upload-outline" size={20} color="#fff" />
              <Text style={{ color: "#fff", fontSize: 15, fontWeight: "600", marginLeft: 8 }}>
                Upload PDF
              </Text>
            </>
          )}
        </TouchableOpacity>

        {/* Document List */}
        {loading ? (
          <View style={{ flex: 1, alignItems: "center", justifyContent: "center" }}>
            <ActivityIndicator size="large" color="#7C3AED" />
          </View>
        ) : (
          <FlatList
            data={documents}
            keyExtractor={(item) => item.id}
            showsVerticalScrollIndicator={false}
            renderItem={({ item }) => (
              <View style={{ backgroundColor: "#1E1E2E", borderRadius: 14, padding: 14, marginBottom: 12, borderWidth: 1, borderColor: "#2A2A3E" }}>
                <View style={{ flexDirection: "row", alignItems: "flex-start" }}>
                  {/* File icon */}
                  <View style={{ width: 42, height: 42, borderRadius: 10, backgroundColor: "#2D1F5E", alignItems: "center", justifyContent: "center", marginRight: 12 }}>
                    <Ionicons name="document-text" size={22} color="#7C3AED" />
                  </View>

                  {/* File info */}
                  <View style={{ flex: 1 }}>
                    <Text style={{ color: "#E5E7EB", fontSize: 14, fontWeight: "600" }} numberOfLines={1}>
                      {item.filename}
                    </Text>
                    <View style={{ flexDirection: "row", alignItems: "center", marginTop: 4 }}>
                      <Ionicons name={statusIcon(item.status)} size={13} color={statusColor(item.status)} />
                      <Text style={{ color: statusColor(item.status), fontSize: 12, marginLeft: 4, textTransform: "capitalize" }}>
                        {item.status}
                      </Text>
                      <Text style={{ color: "#4B5563", fontSize: 12, marginLeft: 8 }}>
                        {new Date(item.uploaded_at).toLocaleDateString()}
                      </Text>
                    </View>
                  </View>

                  {/* Action buttons */}
                  <View style={{ flexDirection: "row", alignItems: "center", marginLeft: 8 }}>
                    {item.status === "ready" && (
                      <TouchableOpacity
                        onPress={() => handleChat(item)}
                        style={{ width: 34, height: 34, borderRadius: 17, backgroundColor: "#2D1F5E", alignItems: "center", justifyContent: "center", marginRight: 8 }}
                      >
                        <Ionicons name="chatbubble-ellipses" size={16} color="#7C3AED" />
                      </TouchableOpacity>
                    )}
                    <TouchableOpacity
                      onPress={() => handleDelete(item.id)}
                      style={{ width: 34, height: 34, borderRadius: 17, backgroundColor: "#3B1515", alignItems: "center", justifyContent: "center" }}
                    >
                      <Ionicons name="trash" size={16} color="#EF4444" />
                    </TouchableOpacity>
                  </View>
                </View>
              </View>
            )}
            ListEmptyComponent={
              <View style={{ alignItems: "center", paddingTop: 80 }}>
                <Ionicons name="document-outline" size={56} color="#2A2A3E" />
                <Text style={{ color: "#4B5563", fontSize: 15, marginTop: 16, fontWeight: "500" }}>
                  No documents uploaded
                </Text>
                <Text style={{ color: "#374151", fontSize: 13, marginTop: 6 }}>
                  Upload a PDF to chat with it using RAG
                </Text>
              </View>
            }
          />
        )}
      </View>
    </SafeAreaView>
  );
}
