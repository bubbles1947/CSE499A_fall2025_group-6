import { Leaf, Github } from "lucide-react";
import Link from "next/link";

const QUICK_LINKS = [
  { href: "/",         label: "Home"     },
  { href: "/chat",     label: "Chat"     },
  { href: "/analyze",  label: "Analyze"  },
  { href: "/advisory", label: "Advisory" },
  { href: "/about",    label: "About"    },
] as const;

const BUILT_WITH = [
  "Next.js 14",
  "FastAPI",
  "Ollama · Qwen2.5",
  "Tailwind CSS",
] as const;

export default function Footer() {
  return (
    <footer className="bg-primary-900 text-white">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12">

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-8">

          {/* ── Column 1 — Brand + tagline ── */}
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Leaf size={22} className="text-primary-300" strokeWidth={2} />
              <span className="text-lg font-bold text-white tracking-tight">
                KrishiBot
              </span>
            </div>
            <p className="text-primary-200 text-sm leading-relaxed">
              AI Agriculture Assistant — helping South Asian farmers grow smarter
              with local, private AI.
            </p>
            <Link
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 text-primary-200 hover:text-white text-sm transition-colors"
              aria-label="GitHub repository"
            >
              <Github size={16} />
              <span>GitHub</span>
            </Link>
          </div>

          {/* ── Column 2 — Quick links ── */}
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wide text-primary-300 mb-4">
              Quick Links
            </h3>
            <ul className="space-y-2">
              {QUICK_LINKS.map(({ href, label }) => (
                <li key={href}>
                  <Link
                    href={href}
                    className="text-primary-200 hover:text-white text-sm transition-colors"
                  >
                    {label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* ── Column 3 — Built with ── */}
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wide text-primary-300 mb-4">
              Built With
            </h3>
            <ul className="space-y-2">
              {BUILT_WITH.map((item) => (
                <li key={item} className="text-primary-200 text-sm">
                  {item}
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* ── Copyright row ── */}
        <div className="mt-10 pt-6 border-t border-primary-800 flex flex-col sm:flex-row items-center justify-between gap-2">
          <p className="text-primary-400 text-xs">
            KrishiBot &copy; 2025 — AI Agriculture Assistant
          </p>
          <p className="text-primary-500 text-xs">
            Built as a Final Year Project
          </p>
        </div>

      </div>
    </footer>
  );
}
