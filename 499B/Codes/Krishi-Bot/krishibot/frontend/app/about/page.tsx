import {
  Leaf,
  Shield,
  Wifi,
  Monitor,
  Server,
  Brain,
  ArrowRight,
  CheckCircle,
  Cpu,
  MemoryStick,
  HardDrive,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Static data
// ---------------------------------------------------------------------------

const TECH_STACK: ReadonlyArray<{ layer: string; items: string[] }> = [
  {
    layer: "Frontend",
    items: ["Next.js 14", "React 18", "TypeScript", "Tailwind CSS"],
  },
  {
    layer: "Backend",
    items: ["FastAPI", "Python 3.11", "SQLAlchemy 2", "Pydantic v2"],
  },
  {
    layer: "AI Models",
    items: ["Ollama", "Qwen2.5:3b"],
  },
  {
    layer: "Database",
    items: ["SQLite", "PostgreSQL (optional)"],
  },
] as const;

const PROJECT_STATS = [
  { value: "5",     label: "Crops"           },
  { value: "2",     label: "Languages"       },
  { value: "Local", label: "AI"              },
  { value: "No",    label: "Cloud Needed"    },
] as const;

const USE_CASES = [
  "Rice blast, brown spot, and bacterial blight detection",
  "Tomato early blight, late blight, and leaf curl diagnosis",
  "Potato late blight and common scab identification",
  "Irrigation scheduling and water-use optimization",
  "Fertilizer (NPK) dosage and application timing",
  "Integrated pest management and pesticide guidance",
  "Harvest timing and post-harvest storage advice",
  "Bangla-language support for local farmers",
] as const;

const REQUIREMENTS = [
  { icon: Monitor,    label: "Python",   value: "3.11 or newer"     },
  { icon: Server,     label: "Node.js",  value: "18 or newer"       },
  { icon: Brain,      label: "Ollama",   value: "Latest version"    },
  { icon: MemoryStick,label: "RAM",      value: "16 GB recommended" },
  { icon: Cpu,        label: "GPU",      value: "4 GB VRAM (optional, for speed)" },
  { icon: HardDrive,  label: "Storage",  value: "~10 GB for models" },
] as const;

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function AboutPage() {
  return (
    <div className="bg-white">

      {/* ================================================================
          PAGE HEADER
      ================================================================ */}
      <section className="bg-gradient-to-br from-primary-800 to-primary-600 text-white py-16 sm:py-20">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-primary-600 border border-primary-400 mb-6">
            <Leaf size={32} className="text-primary-100" strokeWidth={1.75} />
          </div>
          <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight mb-4">
            About KrishiBot
          </h1>
          <p className="text-primary-200 text-lg max-w-2xl mx-auto">
            An AI-powered agricultural assistant built as a Final Year Project
            to bring modern AI capabilities to smallholder farmers.
          </p>
        </div>
      </section>

      {/* ================================================================
          PROJECT STATS
      ================================================================ */}
      <section className="py-12 sm:py-14 bg-white border-b border-gray-100">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {PROJECT_STATS.map(({ value, label }) => (
              <div
                key={label}
                className="bg-primary-50 border border-primary-100 rounded-2xl px-5 py-6 text-center"
              >
                <div className="text-3xl sm:text-4xl font-extrabold text-primary-700 tracking-tight">
                  {value}
                </div>
                <div className="text-xs sm:text-sm font-semibold text-gray-600 mt-1 uppercase tracking-wide">
                  {label}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ================================================================
          PROJECT DESCRIPTION
      ================================================================ */}
      <section className="py-16 sm:py-20">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-8">The Problem</h2>
          <div className="space-y-5 text-gray-600 text-lg leading-relaxed">
            <p>
              Smallholder farmers in Bangladesh and across South Asia face significant
              challenges in accessing timely, reliable agricultural advice. Crop diseases
              can devastate an entire harvest if not identified early, yet many farmers
              lack access to agronomists or extension officers — especially in rural areas
              with limited connectivity.
            </p>
            <p>
              Existing online tools either require a constant internet connection, charge
              subscription fees, or produce advice too generic to be actionable for a
              specific crop in a specific region. Language is another barrier: most
              agricultural AI tools work only in English, excluding a large portion of
              the farming population.
            </p>
            <p>
              KrishiBot addresses these gaps by running entirely on local hardware using
              open-source large language models via Ollama. It supports both English and
              Bangla, works offline once the models are downloaded, and covers the most
              common crops and diseases found in South Asia — from rice blast to tomato
              late blight.
            </p>
          </div>
        </div>
      </section>

      {/* ================================================================
          WHY LOCAL AI
      ================================================================ */}
      <section className="py-16 sm:py-20 bg-primary-50">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">Why Local AI?</h2>
          <p className="text-gray-500 text-lg mb-10">
            Most AI assistants send your data to remote servers. KrishiBot takes a different approach.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            {/* Privacy */}
            <div className="bg-white rounded-2xl border border-primary-100 p-7 flex flex-col gap-3">
              <div className="inline-flex items-center justify-center w-11 h-11 rounded-xl bg-primary-100">
                <Shield size={22} className="text-primary-700" />
              </div>
              <h3 className="text-lg font-bold text-gray-900">Complete Privacy</h3>
              <p className="text-gray-500 text-sm leading-relaxed">
                Your crop photos and farm data never leave your machine. There are no
                accounts, no telemetry, and no third-party API calls. The model runs
                entirely in your local environment.
              </p>
            </div>

            {/* Offline */}
            <div className="bg-white rounded-2xl border border-primary-100 p-7 flex flex-col gap-3">
              <div className="inline-flex items-center justify-center w-11 h-11 rounded-xl bg-primary-100">
                <Wifi size={22} className="text-primary-700" />
              </div>
              <h3 className="text-lg font-bold text-gray-900">Works Offline</h3>
              <p className="text-gray-500 text-sm leading-relaxed">
                Once Ollama models are downloaded, KrishiBot works without an internet
                connection — critical for rural areas with unreliable connectivity. Only
                the initial model download requires internet access.
              </p>
            </div>

            {/* No cost */}
            <div className="bg-white rounded-2xl border border-primary-100 p-7 flex flex-col gap-3">
              <div className="inline-flex items-center justify-center w-11 h-11 rounded-xl bg-primary-100">
                <CheckCircle size={22} className="text-primary-700" />
              </div>
              <h3 className="text-lg font-bold text-gray-900">No API Costs</h3>
              <p className="text-gray-500 text-sm leading-relaxed">
                Running a cloud LLM at scale is expensive. By using Ollama with
                open-weight models, KrishiBot can be deployed at no per-query cost —
                making it economically viable for agricultural NGOs and extension offices.
              </p>
            </div>

            {/* Control */}
            <div className="bg-white rounded-2xl border border-primary-100 p-7 flex flex-col gap-3">
              <div className="inline-flex items-center justify-center w-11 h-11 rounded-xl bg-primary-100">
                <Cpu size={22} className="text-primary-700" />
              </div>
              <h3 className="text-lg font-bold text-gray-900">Full Control</h3>
              <p className="text-gray-500 text-sm leading-relaxed">
                Swap models by changing a single environment variable. Upgrade from
                Qwen2.5:3b to a 7B or 14B model as hardware allows — no code changes
                required.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ================================================================
          TECH STACK
      ================================================================ */}
      <section className="py-16 sm:py-20">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-10">Technology Stack</h2>

          <div className="space-y-6">
            {TECH_STACK.map(({ layer, items }) => (
              <div key={layer}>
                <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">
                  {layer}
                </h3>
                <div className="flex flex-wrap gap-2">
                  {items.map((tech) => (
                    <span
                      key={tech}
                      className="px-3 py-1.5 rounded-full bg-primary-100 text-primary-800 text-sm font-medium"
                    >
                      {tech}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ================================================================
          ARCHITECTURE
      ================================================================ */}
      <section className="py-16 sm:py-20 bg-primary-50">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">Architecture Overview</h2>
          <p className="text-gray-500 text-lg mb-12">
            A simple three-tier architecture — all components run on the same machine.
          </p>

          {/* Architecture diagram */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            {/* Box 1 — Frontend (purple tint) */}
            <div className="w-full sm:w-48 rounded-2xl bg-purple-50 border-2 border-purple-200 p-6 text-center shadow-sm">
              <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-purple-100 mb-3">
                <Monitor size={24} className="text-purple-700" />
              </div>
              <h3 className="font-bold text-gray-900 text-sm mb-1">Frontend</h3>
              <p className="text-gray-500 text-xs">Next.js 14</p>
              <p className="text-gray-400 text-xs">localhost:3000</p>
            </div>

            {/* Arrow */}
            <div className="flex sm:flex-col items-center gap-1 text-primary-400">
              <ArrowRight size={20} className="sm:rotate-90" />
              <span className="text-xs font-medium text-primary-500">REST / SSE</span>
            </div>

            {/* Box 2 — Backend (blue tint) */}
            <div className="w-full sm:w-48 rounded-2xl bg-blue-50 border-2 border-blue-200 p-6 text-center shadow-sm">
              <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-blue-100 mb-3">
                <Server size={24} className="text-blue-700" />
              </div>
              <h3 className="font-bold text-gray-900 text-sm mb-1">Backend</h3>
              <p className="text-gray-500 text-xs">FastAPI + SQLite</p>
              <p className="text-gray-400 text-xs">localhost:8000</p>
            </div>

            {/* Arrow */}
            <div className="flex sm:flex-col items-center gap-1 text-primary-400">
              <ArrowRight size={20} className="sm:rotate-90" />
              <span className="text-xs font-medium text-primary-500">HTTP / NDJSON</span>
            </div>

            {/* Box 3 — AI (green tint) */}
            <div className="w-full sm:w-48 rounded-2xl bg-primary-50 border-2 border-primary-300 p-6 text-center shadow-sm">
              <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-primary-100 mb-3">
                <Brain size={24} className="text-primary-700" />
              </div>
              <h3 className="font-bold text-gray-900 text-sm mb-1">Ollama AI</h3>
              <p className="text-gray-500 text-xs">Qwen2.5 3B</p>
              <p className="text-gray-400 text-xs">Local Inference</p>
            </div>
          </div>

          <p className="text-center text-gray-400 text-sm mt-8">
            All three components run locally — no cloud dependency after setup.
          </p>
        </div>
      </section>

      {/* ================================================================
          SUPPORTED USE CASES
      ================================================================ */}
      <section className="py-16 sm:py-20">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-10">Supported Use Cases</h2>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {USE_CASES.map((useCase) => (
              <div
                key={useCase}
                className="flex items-start gap-3 p-4 rounded-xl bg-primary-50 border border-primary-100"
              >
                <CheckCircle
                  size={18}
                  className="text-primary-600 mt-0.5 shrink-0"
                  strokeWidth={2}
                />
                <span className="text-gray-700 text-sm leading-relaxed">{useCase}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ================================================================
          SYSTEM REQUIREMENTS
      ================================================================ */}
      <section className="py-16 sm:py-20 bg-primary-50">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">System Requirements</h2>
          <p className="text-gray-500 text-lg mb-10">
            KrishiBot runs on any reasonably modern laptop or desktop computer.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {REQUIREMENTS.map(({ icon: Icon, label, value }) => (
              <div
                key={label}
                className="flex items-center gap-4 bg-white rounded-xl border border-primary-100 px-5 py-4 shadow-sm"
              >
                <div className="inline-flex items-center justify-center w-10 h-10 rounded-lg bg-primary-100 shrink-0">
                  <Icon size={18} className="text-primary-700" />
                </div>
                <div>
                  <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide">{label}</p>
                  <p className="text-sm font-medium text-gray-800">{value}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ================================================================
          TEAM / PROJECT INFO
      ================================================================ */}
      <section className="py-16 sm:py-20">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-10">Project Information</h2>

          <div className="rounded-2xl border border-gray-200 overflow-hidden">
            {[
              { label: "Project Type",  value: "Final Year Project (FYP)" },
              { label: "Domain",        value: "Agricultural AI / Precision Farming" },
              { label: "AI Approach",   value: "Local LLM inference via Ollama (no cloud)" },
              { label: "Language",      value: "English and Bangla (বাংলা)" },
              { label: "License",       value: "MIT — open source" },
              { label: "Target Users",  value: "Smallholder farmers, agricultural officers, NGOs" },
            ].map(({ label, value }, i) => (
              <div
                key={label}
                className={`flex flex-col sm:flex-row sm:items-center px-6 py-4 gap-1 sm:gap-6 border-b border-gray-100 last:border-0 ${
                  i % 2 === 1 ? "bg-gray-50/50" : "bg-white"
                }`}
              >
                <span className="text-sm font-semibold text-gray-400 uppercase tracking-wide sm:w-40 shrink-0">
                  {label}
                </span>
                <span className="text-gray-800 text-sm">{value}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

    </div>
  );
}
