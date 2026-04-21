import Link from "next/link";
import {
  MessageSquare,
  Camera,
  BookOpen,
  Leaf,
  ArrowRight,
  Upload,
  Cpu,
  CheckCircle,
} from "lucide-react";
import WeatherRiskCard from "@/components/WeatherRiskCard";
import CommunityStatsSection from "@/components/CommunityStatsSection";

// ---------------------------------------------------------------------------
// Static data
// ---------------------------------------------------------------------------

const FEATURES = [
  {
    icon: MessageSquare,
    title: "Agriculture Chatbot",
    description:
      "Ask questions in English or Bangla and get expert farming advice tailored to South Asian crops and conditions.",
    color: "text-primary-600",
    bg: "bg-primary-50",
    border: "border-primary-200",
  },
  {
    icon: Camera,
    title: "Disease Detection",
    description:
      "Upload a photo of a leaf or crop. The AI vision model instantly identifies diseases and suggests treatment steps.",
    color: "text-accent-600",
    bg: "bg-amber-50",
    border: "border-amber-200",
  },
  {
    icon: BookOpen,
    title: "Crop Advisory",
    description:
      "Access structured guides on irrigation scheduling, fertilizer dosage, pest control, and harvest timing.",
    color: "text-primary-700",
    bg: "bg-primary-50",
    border: "border-primary-200",
  },
] as const;

const STEPS = [
  {
    number: "01",
    icon: Upload,
    title: "Ask or Upload",
    description: "Type your farming question in any language, or upload a photo of a diseased leaf or crop.",
  },
  {
    number: "02",
    icon: Cpu,
    title: "AI Analyzes",
    description: "Local AI models running on your machine process the query — your data never leaves your device.",
  },
  {
    number: "03",
    icon: CheckCircle,
    title: "Get Guidance",
    description: "Receive clear, actionable advice: diagnosis, treatment steps, and preventive tips.",
  },
] as const;

const CROPS = [
  { emoji: "🌾", name: "Rice",   nameBn: "ধান"    },
  { emoji: "🍅", name: "Tomato", nameBn: "টমেটো" },
  { emoji: "🥔", name: "Potato", nameBn: "আলু"   },
  { emoji: "🌿", name: "Wheat",  nameBn: "গম"     },
  { emoji: "🌱", name: "Jute",   nameBn: "পাট"    },
] as const;

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function HomePage() {
  return (
    <>
      {/* ================================================================
          HERO
      ================================================================ */}
      <section className="relative bg-gradient-to-br from-primary-800 via-primary-700 to-primary-600 text-white overflow-hidden">
        {/* Decorative background icons */}
        <div aria-hidden className="pointer-events-none absolute inset-0 overflow-hidden">
          <Leaf
            size={320}
            strokeWidth={0.4}
            className="absolute -top-16 -right-20 text-primary-500 opacity-20 rotate-12"
          />
          <Leaf
            size={200}
            strokeWidth={0.4}
            className="absolute bottom-0 -left-10 text-primary-400 opacity-15 -rotate-20"
          />
          <Leaf
            size={120}
            strokeWidth={0.5}
            className="absolute top-1/2 left-1/3 text-primary-300 opacity-10 rotate-45"
          />
        </div>

        <div className="relative max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-24 sm:py-32">
          <div className="max-w-2xl">
            {/* Badge */}
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-primary-600 border border-primary-400 text-primary-100 text-xs font-medium mb-6">
              <Leaf size={12} />
              Powered by Local AI — 100% Private
            </span>

            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold leading-tight tracking-tight mb-6">
              Smart Agriculture Assistant{" "}
              <span className="text-primary-200">Powered by Local AI</span>
            </h1>

            <p className="text-lg sm:text-xl text-primary-100 leading-relaxed mb-10 max-w-xl">
              Get expert advice on crop diseases, irrigation, fertilizer and more.
              Works offline with local AI models — your farm data stays private.
            </p>

            <div className="flex flex-col sm:flex-row gap-4">
              <Link
                href="/chat"
                className="inline-flex items-center justify-center gap-2 px-7 py-3.5 rounded-xl bg-white text-primary-800 font-semibold text-base hover:bg-primary-50 transition-colors shadow-lg shadow-primary-900/20"
              >
                <MessageSquare size={18} />
                Start Chatting
              </Link>
              <Link
                href="/analyze"
                className="inline-flex items-center justify-center gap-2 px-7 py-3.5 rounded-xl bg-transparent border-2 border-white text-white font-semibold text-base hover:bg-white/10 transition-colors"
              >
                <Camera size={18} />
                Analyze Crop Disease
                <ArrowRight size={16} />
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* ================================================================
          FEATURES
      ================================================================ */}
      <section className="py-20 bg-white">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-14">
            <h2 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-4">
              Everything a Farmer Needs
            </h2>
            <p className="text-gray-500 text-lg max-w-xl mx-auto">
              Three powerful tools in one assistant, designed for real farming challenges.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-8">
            {FEATURES.map(({ icon: Icon, title, description, color, bg, border }) => (
              <div
                key={title}
                className={`rounded-2xl border ${border} ${bg} p-8 flex flex-col gap-4 hover:shadow-md transition-shadow`}
              >
                <div className={`inline-flex items-center justify-center w-12 h-12 rounded-xl bg-white shadow-sm ${color}`}>
                  <Icon size={24} strokeWidth={1.75} />
                </div>
                <h3 className="text-xl font-bold text-gray-900">{title}</h3>
                <p className="text-gray-600 text-sm leading-relaxed">{description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ================================================================
          LIVE DISEASE RISK FORECAST
      ================================================================ */}
      <section className="py-20 bg-white border-t border-gray-100">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-10">
            <h2 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-3">
              Live Disease Risk Forecast
            </h2>
            <p className="text-gray-500 text-lg max-w-xl mx-auto">
              Based on current weather conditions in your area.
            </p>
          </div>

          <div className="max-w-2xl mx-auto">
            <WeatherRiskCard />
          </div>
        </div>
      </section>

      {/* ================================================================
          COMMUNITY DISEASE REPORTS
      ================================================================ */}
      <CommunityStatsSection />

      {/* ================================================================
          HOW IT WORKS
      ================================================================ */}
      <section className="py-20 bg-primary-50">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-14">
            <h2 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-4">
              How It Works
            </h2>
            <p className="text-gray-500 text-lg max-w-xl mx-auto">
              From question to actionable advice in three simple steps.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-8 relative">
            {/* Connector line (desktop only) */}
            <div
              aria-hidden
              className="hidden sm:block absolute top-10 left-[calc(16.67%+1rem)] right-[calc(16.67%+1rem)] h-0.5 bg-primary-200"
            />

            {STEPS.map(({ number, icon: Icon, title, description }) => (
              <div key={number} className="relative flex flex-col items-center text-center gap-4">
                {/* Step circle */}
                <div className="relative z-10 flex items-center justify-center w-20 h-20 rounded-full bg-white border-2 border-primary-200 shadow-sm">
                  <Icon size={28} className="text-primary-600" strokeWidth={1.75} />
                  <span className="absolute -top-1.5 -right-1.5 w-6 h-6 rounded-full bg-primary-600 text-white text-xs font-bold flex items-center justify-center">
                    {number.slice(1)}
                  </span>
                </div>
                <h3 className="text-lg font-bold text-gray-900">{title}</h3>
                <p className="text-gray-500 text-sm leading-relaxed max-w-xs">{description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ================================================================
          SUPPORTED CROPS
      ================================================================ */}
      <section className="py-20 bg-white">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-4">
              Supported Crops
            </h2>
            <p className="text-gray-500 text-lg">
              Expert knowledge for the most common South Asian staple and cash crops.
            </p>
          </div>

          <div className="flex flex-wrap justify-center gap-4">
            {CROPS.map(({ emoji, name, nameBn }) => (
              <div
                key={name}
                className="flex flex-col items-center gap-2 px-8 py-6 rounded-2xl bg-primary-50 border border-primary-100 hover:border-primary-300 hover:shadow-sm transition-all w-32 sm:w-36"
              >
                <span className="text-4xl" role="img" aria-label={name}>
                  {emoji}
                </span>
                <span className="font-semibold text-gray-800 text-sm">{name}</span>
                <span className="text-primary-600 text-xs font-medium">{nameBn}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ================================================================
          BOTTOM CTA
      ================================================================ */}
      <section className="py-20 bg-gradient-to-br from-primary-800 to-primary-600 text-white">
        <div className="max-w-2xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <Leaf size={40} className="mx-auto text-primary-300 mb-6" strokeWidth={1.5} />
          <h2 className="text-3xl sm:text-4xl font-bold mb-4">
            Ready to get farming advice?
          </h2>
          <p className="text-primary-200 text-lg mb-8">
            Ask anything about your crops — in English or Bangla. KrishiBot is here to help.
          </p>
          <Link
            href="/chat"
            className="inline-flex items-center gap-2 px-8 py-4 rounded-xl bg-white text-primary-800 font-bold text-base hover:bg-primary-50 transition-colors shadow-lg shadow-primary-900/20"
          >
            <MessageSquare size={20} />
            Chat with KrishiBot
            <ArrowRight size={18} />
          </Link>
        </div>
      </section>
    </>
  );
}
