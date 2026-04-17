import { create } from "zustand";
import api from "../services/api";
import { synthesizeSpeech } from "../services/voice";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  voice_used: boolean;
  created_at: string;
  sources?: string[];   // RAG source chunks (assistant messages only)
}

export interface Conversation {
  id: string;
  model_name: string;
  title: string;
  rag_enabled: boolean;
  created_at: string;
  messages?: Message[];
}

export interface ModelInfo {
  name: string;
  display_name: string;
  server_url: string;
  status: "online" | "offline";
}

interface ChatState {
  conversations: Conversation[];
  currentConversation: Conversation | null;
  models: ModelInfo[];
  selectedModel: string;
  isStreaming: boolean;
  streamingContent: string;
  voiceMode: boolean;
  isLoadingConversations: boolean;

  // Actions
  fetchConversations: () => Promise<void>;
  fetchConversation: (id: string) => Promise<void>;
  deleteConversation: (id: string) => Promise<void>;
  fetchModels: () => Promise<void>;
  setSelectedModel: (model: string) => void;
  setVoiceMode: (enabled: boolean) => void;
  sendMessage: (message: string, conversationId?: string) => Promise<string>;
  sendRagMessage: (message: string, documentId: string, conversationId?: string) => Promise<string>;
  clearCurrent: () => void;
}

const useChatStore = create<ChatState>((set, get) => ({
  conversations: [],
  currentConversation: null,
  models: [],
  selectedModel: "llama3",
  isStreaming: false,
  streamingContent: "",
  voiceMode: false,
  isLoadingConversations: false,

  fetchConversations: async () => {
    set({ isLoadingConversations: true });
    try {
      const res = await api.get("/chat/conversations");
      set({ conversations: res.data });
    } finally {
      set({ isLoadingConversations: false });
    }
  },

  fetchConversation: async (id) => {
    const res = await api.get(`/chat/conversations/${id}`);
    set({ currentConversation: res.data });
  },

  deleteConversation: async (id) => {
    await api.delete(`/chat/conversations/${id}`);
    set((state) => ({
      conversations: state.conversations.filter((c) => c.id !== id),
      currentConversation:
        state.currentConversation?.id === id ? null : state.currentConversation,
    }));
  },

  fetchModels: async () => {
    const res = await api.get("/chat/models");
    set({ models: res.data });
  },

  setSelectedModel: (model) => set({ selectedModel: model }),
  setVoiceMode: (enabled) => set({ voiceMode: enabled }),

  sendMessage: async (message, conversationId?) => {
    const { selectedModel } = get();
    set({ isStreaming: true, streamingContent: "" });

    const body: Record<string, unknown> = {
      model_name: selectedModel,
      message,
      rag_enabled: false,
    };
    if (conversationId) {
      body.conversation_id = conversationId;
    }

    // Add user message to local state immediately
    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content: message,
      voice_used: false,
      created_at: new Date().toISOString(),
    };

    set((state) => {
      const conv = state.currentConversation;
      if (conv) {
        return {
          currentConversation: {
            ...conv,
            messages: [...(conv.messages || []), userMsg],
          },
        };
      }
      return {};
    });

    let fullResponse = "";
    let newConversationId = conversationId || "";

    try {
      // Use fetch for SSE streaming (axios doesn't support streaming well on RN)
      const token = api.defaults.headers.common["Authorization"];
      const baseURL = api.defaults.baseURL;

      const response = await fetch(`${baseURL}/chat/send`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: token as string,
        },
        body: JSON.stringify(body),
      });

      // Get conversation ID from response headers
      newConversationId =
        response.headers.get("X-Conversation-Id") || newConversationId;

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (reader) {
        let done = false;
        while (!done) {
          const { value, done: readerDone } = await reader.read();
          done = readerDone;
          if (value) {
            const chunk = decoder.decode(value, { stream: true });
            // Parse SSE lines
            const lines = chunk.split("\n");
            for (const line of lines) {
              if (line.startsWith("data: ") && line.trim() !== "data: [DONE]") {
                try {
                  const payload = JSON.parse(line.slice(6));
                  const delta = payload.choices?.[0]?.delta;
                  if (delta?.content) {
                    fullResponse += delta.content;
                    set({ streamingContent: fullResponse });
                  }
                } catch {
                  // Skip malformed JSON lines
                }
              }
            }
          }
        }
      }
    } finally {
      // Add assistant message to local state immediately (optimistic)
      const assistantMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: fullResponse || "No response received.",
        voice_used: false,
        created_at: new Date().toISOString(),
      };

      set((state) => {
        const conv = state.currentConversation;
        if (conv) {
          return {
            isStreaming: false,
            streamingContent: "",
            currentConversation: {
              ...conv,
              id: newConversationId || conv.id,
              messages: [...(conv.messages || []), assistantMsg],
            },
          };
        }
        return { isStreaming: false, streamingContent: "" };
      });

      // Refresh conversation list so history drawer stays up-to-date
      try {
        const res = await api.get("/chat/conversations");
        set({ conversations: res.data });
      } catch {
        // Non-fatal — drawer will update on next open
      }

      // Auto-play TTS if voice mode is on and we got a response
      if (get().voiceMode && fullResponse) {
        // Strip <think> tags before speaking
        const speakText = fullResponse
          .replace(/<think>[\s\S]*?<\/think>/g, "")
          .trim();
        if (speakText) {
          synthesizeSpeech(speakText).catch(() => {
            // Non-fatal — TTS failure shouldn't break chat
          });
        }
      }
    }

    return newConversationId;
  },

  sendRagMessage: async (message, documentId, conversationId?) => {
    const { selectedModel } = get();
    set({ isStreaming: true, streamingContent: "" });

    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content: message,
      voice_used: false,
      created_at: new Date().toISOString(),
    };

    set((state) => {
      const conv = state.currentConversation;
      if (conv) {
        return {
          currentConversation: { ...conv, messages: [...(conv.messages || []), userMsg] },
        };
      }
      return {};
    });

    try {
      const body: Record<string, unknown> = {
        document_id: documentId,
        message,
        model_name: selectedModel,
      };
      if (conversationId) body.conversation_id = conversationId;

      const res = await api.post("/rag/chat", body);
      const { answer, sources, conversation_id: newConvId } = res.data;

      const assistantMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: answer,
        sources: sources || [],
        voice_used: false,
        created_at: new Date().toISOString(),
      };

      set((state) => {
        const conv = state.currentConversation;
        if (conv) {
          return {
            isStreaming: false,
            streamingContent: "",
            currentConversation: {
              ...conv,
              id: newConvId || conv.id,
              messages: [...(conv.messages || []), assistantMsg],
            },
          };
        }
        return { isStreaming: false, streamingContent: "" };
      });

      try {
        const res2 = await api.get("/chat/conversations");
        set({ conversations: res2.data });
      } catch {
        // Non-fatal
      }

      if (get().voiceMode && answer) {
        const speakText = answer.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
        if (speakText) {
          synthesizeSpeech(speakText).catch(() => {});
        }
      }

      return newConvId || "";
    } catch (err) {
      set({ isStreaming: false, streamingContent: "" });
      throw err;
    }
  },

  clearCurrent: () => set({ currentConversation: null, streamingContent: "" }),
}));

export default useChatStore;
