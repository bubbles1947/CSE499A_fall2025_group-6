import { Leaf, User } from "lucide-react";
import type { ChatMessage } from "@/lib/api";

interface Props {
  message: ChatMessage;
  isStreaming?: boolean;
}

// ---------------------------------------------------------------------------
// Inline markdown renderer
// Supports: **bold**, line breaks (\n). No external dependency needed.
// ---------------------------------------------------------------------------

function renderContent(text: string): React.ReactNode[] {
  // Split on newlines first, then process each line for bold spans.
  return text.split("\n").flatMap((line, lineIdx, lines) => {
    const parts: React.ReactNode[] = [];
    const boldRegex = /\*\*(.+?)\*\*/g;
    let lastIndex = 0;
    let match: RegExpExecArray | null;

    while ((match = boldRegex.exec(line)) !== null) {
      if (match.index > lastIndex) {
        parts.push(line.slice(lastIndex, match.index));
      }
      parts.push(
        <strong key={`bold-${lineIdx}-${match.index}`} className="font-semibold">
          {match[1]}
        </strong>,
      );
      lastIndex = boldRegex.lastIndex;
    }

    if (lastIndex < line.length) {
      parts.push(line.slice(lastIndex));
    }

    // Add <br> between lines (but not after the last one).
    if (lineIdx < lines.length - 1) {
      parts.push(<br key={`br-${lineIdx}`} />);
    }

    return parts;
  });
}

// ---------------------------------------------------------------------------
// Timestamp formatter
// ---------------------------------------------------------------------------

function formatTime(iso?: string): string {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function MessageBubble({ message, isStreaming = false }: Props) {
  const isUser = message.role === "user";

  return (
    <div
      className={`flex items-end gap-2.5 ${isUser ? "flex-row-reverse" : "flex-row"}`}
    >
      {/* Avatar */}
      <div
        className={`shrink-0 flex items-center justify-center w-8 h-8 rounded-full mb-5
          ${isUser
            ? "bg-primary-600 text-white"
            : "bg-primary-50 border border-primary-200 text-primary-700"
          }`}
        aria-hidden
      >
        {isUser
          ? <User size={15} strokeWidth={2} />
          : <Leaf size={15} strokeWidth={2} />
        }
      </div>

      {/* Bubble + timestamp wrapper */}
      <div className={`flex flex-col gap-1 max-w-[75%] ${isUser ? "items-end" : "items-start"}`}>
        <div
          className={`px-4 py-3 text-sm leading-relaxed break-words
            ${isUser
              ? "bg-primary-600 text-white rounded-2xl rounded-br-none shadow-sm"
              : "bg-white border border-gray-200 text-gray-800 rounded-2xl rounded-bl-none shadow-sm"
            }`}
        >
          {renderContent(message.content)}

          {/* Blinking cursor while the assistant is still streaming */}
          {isStreaming && !isUser && (
            <span
              className="inline-block w-0.5 h-4 ml-0.5 align-middle bg-primary-500 animate-pulse"
              aria-hidden
            />
          )}
        </div>

        {/* Timestamp */}
        {message.timestamp && (
          <span className="text-xs text-gray-400 px-1">
            {formatTime(message.timestamp)}
          </span>
        )}
      </div>
    </div>
  );
}
