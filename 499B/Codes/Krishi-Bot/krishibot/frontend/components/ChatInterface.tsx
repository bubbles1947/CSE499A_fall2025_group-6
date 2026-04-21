"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { v4 as uuidv4 } from "uuid";
import { Send, Trash2, Loader2, Leaf, Plus, Cpu } from "lucide-react";
import MessageBubble from "@/components/MessageBubble";
import {
  sendChatMessage,
  getChatHistory,
  clearChatSession,
  type ChatMessage,
} from "@/lib/api";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SUGGESTED_QUESTIONS = [
  "My rice leaves are turning yellow, what should I do?",
  "How often should I water tomato plants?",
  "What fertilizer is best for potato crops?",
  "How do I prevent pest damage on my crops?",
] as const;

const SESSION_KEY = "krishibot_session_id";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getOrCreateSessionId(): string {
  if (typeof window === "undefined") return uuidv4();
  const stored = sessionStorage.getItem(SESSION_KEY);
  if (stored) return stored;
  const id = uuidv4();
  sessionStorage.setItem(SESSION_KEY, id);
  return id;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function ChatInterface() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputText, setInputText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string>("");
  const [language, setLanguage] = useState<"en" | "bn">("en");
  const [streamingId, setStreamingId] = useState<string | null>(null);

  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // ── Initialise session and load history ────────────────────────────────
  useEffect(() => {
    const id = getOrCreateSessionId();
    setSessionId(id);

    getChatHistory(id)
      .then((history) => {
        if (history.length > 0) setMessages(history);
      })
      .catch(() => {
        // History fetch failure is non-fatal — start fresh.
      });
  }, []);

  // ── Auto-scroll to bottom on new messages ─────────────────────────────
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ── Auto-grow textarea ─────────────────────────────────────────────────
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [inputText]);

  // ── Send message ───────────────────────────────────────────────────────
  const handleSend = useCallback(
    async (overrideText?: string) => {
      const text = (overrideText ?? inputText).trim();
      if (!text || isLoading) return;

      setInputText("");

      const userMsg: ChatMessage = {
        role: "user",
        content: text,
        timestamp: new Date().toISOString(),
      };

      // Placeholder for the streaming assistant reply
      const assistantId = uuidv4();
      const assistantMsg: ChatMessage = {
        id: assistantId as unknown as number,
        role: "assistant",
        content: "",
        timestamp: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setIsLoading(true);
      setStreamingId(assistantId);

      try {
        await sendChatMessage(
          sessionId,
          text,
          language,
          // onToken — append each token to the streaming assistant bubble
          (token) => {
            setMessages((prev) =>
              prev.map((m) =>
                (m.id as unknown as string) === assistantId
                  ? { ...m, content: m.content + token }
                  : m,
              ),
            );
          },
          // onDone
          () => {
            setStreamingId(null);
            setIsLoading(false);
          },
        );
      } catch (err) {
        const errorText =
          err instanceof Error ? err.message : "Something went wrong. Please try again.";
        setMessages((prev) =>
          prev.map((m) =>
            (m.id as unknown as string) === assistantId
              ? { ...m, content: `⚠️ ${errorText}` }
              : m,
          ),
        );
        setStreamingId(null);
        setIsLoading(false);
      }
    },
    [inputText, isLoading, sessionId, language],
  );

  // ── Clear session ──────────────────────────────────────────────────────
  const handleClear = useCallback(async () => {
    if (isLoading) return;
    try {
      await clearChatSession(sessionId);
    } catch {
      // Ignore — session may not exist yet on the server.
    }
    setMessages([]);
  }, [sessionId, isLoading]);

  // ── Keyboard handler ───────────────────────────────────────────────────
  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  // ── Render ─────────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col h-full bg-gray-50">

      {/* ── Toolbar ── */}
      <div className="flex items-center justify-between px-4 py-2.5 bg-white border-b border-gray-200 shrink-0">
        <div className="flex items-center gap-2 text-primary-700">
          <Leaf size={18} strokeWidth={2} />
          <span className="text-sm font-semibold">KrishiBot Chat</span>
          {sessionId && (
            <span className="hidden sm:inline text-xs text-gray-400 font-mono">
              #{sessionId.slice(0, 8)}
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {/* Language toggle */}
          <button
            type="button"
            onClick={() => setLanguage((l) => (l === "en" ? "bn" : "en"))}
            className="px-3 py-1.5 rounded-lg text-xs font-semibold border border-primary-200 text-primary-700 hover:bg-primary-50 transition-colors"
            title="Toggle language"
          >
            {language === "en" ? "EN" : "বাংলা"}
          </button>

          {/* New Chat */}
          <button
            type="button"
            onClick={handleClear}
            disabled={isLoading}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-primary-600 text-white hover:bg-primary-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors shadow-sm"
            title="Start a new chat"
          >
            <Plus size={14} strokeWidth={2.5} />
            <span className="hidden sm:inline">New Chat</span>
          </button>

          {/* Clear */}
          <button
            type="button"
            onClick={handleClear}
            disabled={isLoading || messages.length === 0}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-gray-500 hover:text-red-600 hover:bg-red-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            title="Clear conversation"
          >
            <Trash2 size={14} />
            <span className="hidden sm:inline">Clear</span>
          </button>
        </div>
      </div>

      {/* ── Message list ── */}
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
        {messages.length === 0 ? (
          /* Empty state — suggested questions */
          <div className="flex flex-col items-center justify-center h-full gap-8 text-center">
            <div>
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-primary-100 border border-primary-200 mb-4">
                <Leaf size={32} className="text-primary-600" strokeWidth={1.75} />
              </div>
              <h3 className="text-xl font-bold text-gray-800 mb-1">
                {language === "bn" ? "আমি কীভাবে সাহায্য করতে পারি?" : "How can I help you?"}
              </h3>
              <p className="text-gray-400 text-sm">
                {language === "bn"
                  ? "নিচের প্রশ্নগুলো থেকে বেছে নিন বা নিজে লিখুন"
                  : "Choose a question below or type your own"}
              </p>
            </div>

            <div className="flex flex-wrap justify-center gap-2 w-full max-w-2xl">
              {SUGGESTED_QUESTIONS.map((q) => (
                <button
                  key={q}
                  type="button"
                  onClick={() => handleSend(q)}
                  className="px-4 py-2 rounded-full bg-primary-50 border border-primary-200 text-sm text-primary-800 hover:bg-primary-100 hover:border-primary-300 transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((msg, i) => (
              <MessageBubble
                key={msg.id ?? `msg-${i}`}
                message={msg}
                isStreaming={
                  msg.role === "assistant" &&
                  (msg.id as unknown as string) === streamingId
                }
              />
            ))}

            {/* "Thinking" indicator — shown until the first token arrives */}
            {isLoading && streamingId !== null && messages.at(-1)?.content === "" && (
              <div className="flex items-center gap-2 text-gray-400 text-sm pl-11">
                <Loader2 size={15} className="animate-spin" />
                KrishiBot is thinking…
              </div>
            )}
          </>
        )}

        {/* Scroll anchor */}
        <div ref={bottomRef} />
      </div>

      {/* ── Input area ── */}
      <div className="shrink-0 bg-white border-t border-gray-200 px-4 py-3">
        <div className="flex items-end gap-3 max-w-4xl mx-auto">
          <textarea
            ref={textareaRef}
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
            placeholder={
              language === "bn"
                ? "আপনার প্রশ্ন লিখুন… (Enter পাঠাতে, Shift+Enter নতুন লাইন)"
                : "Type your question… (Enter to send, Shift+Enter for newline)"
            }
            rows={1}
            className="flex-1 resize-none rounded-xl border border-gray-300 bg-gray-50 px-4 py-3 text-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-primary-400 focus:border-transparent disabled:opacity-60 transition-colors overflow-hidden"
          />

          <button
            type="button"
            onClick={() => handleSend()}
            disabled={isLoading || !inputText.trim()}
            className="shrink-0 inline-flex items-center justify-center w-11 h-11 rounded-xl bg-primary-600 text-white hover:bg-primary-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors shadow-sm"
            aria-label="Send message"
          >
            {isLoading
              ? <Loader2 size={18} className="animate-spin" />
              : <Send size={18} />
            }
          </button>
        </div>

        <div className="flex flex-col items-center gap-1.5 mt-2">
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-primary-50 border border-primary-100 text-[10px] font-medium text-primary-700">
            <Cpu size={10} strokeWidth={2.5} />
            Powered by Qwen2.5 3B
          </span>
          <p className="text-center text-xs text-gray-400">
            KrishiBot can make mistakes — always verify advice with a local agricultural officer.
          </p>
        </div>
      </div>
    </div>
  );
}
