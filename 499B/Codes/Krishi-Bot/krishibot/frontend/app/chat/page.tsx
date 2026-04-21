import type { Metadata } from "next";
import ChatInterface from "@/components/ChatInterface";

export const metadata: Metadata = {
  title: "Chat — KrishiBot",
  description:
    "Ask KrishiBot any farming question in English or Bangla. Get expert advice on crop diseases, irrigation, fertilizer, and pest control.",
};

export default function ChatPage() {
  return (
    /*
      The outer div takes up the full viewport height minus the Navbar (64px)
      and Footer heights. Using dvh (dynamic viewport height) handles mobile
      browser chrome correctly; vh is the fallback.
    */
    <div className="flex flex-col flex-1 bg-gray-50 overflow-hidden">
      {/* Page header — compact, doesn't scroll with messages */}
      <div className="shrink-0 bg-white border-b border-gray-100 px-4 sm:px-6 py-3">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-base font-semibold text-gray-900">
            Agriculture Chat Assistant
          </h1>
          <p className="text-xs text-gray-500 mt-0.5">
            Ask anything about crops, diseases, irrigation, fertilizer or pest control —
            in English or Bangla.
          </p>
        </div>
      </div>

      {/* Chat interface fills remaining space */}
      <div className="flex-1 overflow-hidden max-w-4xl w-full mx-auto">
        <ChatInterface />
      </div>
    </div>
  );
}
