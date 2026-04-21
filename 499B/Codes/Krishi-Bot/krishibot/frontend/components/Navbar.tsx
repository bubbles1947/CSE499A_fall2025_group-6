"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Leaf, Menu, X } from "lucide-react";

const NAV_LINKS = [
  { href: "/",          label: "Home"      },
  { href: "/chat",      label: "Chat"      },
  { href: "/analyze",   label: "Analyze"   },
  { href: "/advisory",  label: "Advisory"  },
  { href: "/community", label: "Community" },
  { href: "/about",     label: "About"     },
] as const;

export default function Navbar() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  function isActive(href: string): boolean {
    if (href === "/") return pathname === "/";
    return pathname.startsWith(href);
  }

  function linkClass(href: string): string {
    const base = "px-3 py-2 rounded-lg text-sm font-medium transition-colors duration-150";
    return isActive(href)
      ? `${base} bg-primary-100 text-primary-800`
      : `${base} text-gray-600 hover:text-primary-700 hover:bg-primary-50`;
  }

  return (
    <header className="sticky top-0 z-50 bg-white border-b border-gray-200 shadow-sm">
      <nav className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">

          {/* ── Logo ── */}
          <Link
            href="/"
            className="flex items-center gap-2 group"
            aria-label="KrishiBot home"
          >
            <Leaf
              size={28}
              className="text-primary-600 group-hover:text-primary-700 transition-colors"
              strokeWidth={2}
            />
            <span className="text-xl font-bold text-primary-800 tracking-tight">
              KrishiBot
            </span>
          </Link>

          {/* ── Desktop links ── */}
          <div className="hidden md:flex items-center gap-1">
            {NAV_LINKS.map(({ href, label }) => (
              <Link key={href} href={href} className={linkClass(href)}>
                {label}
              </Link>
            ))}
          </div>

          {/* ── Mobile hamburger button ── */}
          <button
            type="button"
            className="md:hidden p-2 rounded-lg text-gray-600 hover:text-primary-700 hover:bg-primary-50 transition-colors"
            onClick={() => setMobileOpen((prev) => !prev)}
            aria-label={mobileOpen ? "Close menu" : "Open menu"}
            aria-expanded={mobileOpen}
          >
            {mobileOpen ? <X size={22} /> : <Menu size={22} />}
          </button>
        </div>

        {/* ── Mobile dropdown ── */}
        {mobileOpen && (
          <div className="md:hidden border-t border-gray-100 py-2 pb-4 space-y-1">
            {NAV_LINKS.map(({ href, label }) => (
              <Link
                key={href}
                href={href}
                className={`block ${linkClass(href)}`}
                onClick={() => setMobileOpen(false)}
              >
                {label}
              </Link>
            ))}
          </div>
        )}
      </nav>
    </header>
  );
}
