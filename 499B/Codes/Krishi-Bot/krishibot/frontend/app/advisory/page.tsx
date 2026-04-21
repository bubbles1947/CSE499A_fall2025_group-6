"use client";

import { useEffect, useState, useCallback } from "react";
import {
  BookOpen,
  ChevronRight,
  Sprout,
  Droplets,
  Bug,
  Shield,
  Wheat,
  type LucideIcon,
} from "lucide-react";
import { getCrops, getAdvisory, type Crop, type AdvisoryResponse } from "@/lib/api";
import WeatherRiskCard from "@/components/WeatherRiskCard";

// ---------------------------------------------------------------------------
// Static data
// ---------------------------------------------------------------------------

const TOPICS: ReadonlyArray<{ id: string; label: string; icon: LucideIcon }> = [
  { id: "irrigation",         label: "Irrigation",         icon: Droplets },
  { id: "fertilizer",         label: "Fertilizer",         icon: Sprout   },
  { id: "pest_control",       label: "Pest Control",       icon: Bug      },
  { id: "disease_prevention", label: "Disease Prevention", icon: Shield   },
  { id: "harvest",            label: "Harvest",            icon: Wheat    },
] as const;

type TopicId =
  | "irrigation"
  | "fertilizer"
  | "pest_control"
  | "disease_prevention"
  | "harvest";

// Static quick-tip content per crop — shown in the sidebar.
const QUICK_TIPS: Record<string, string[]> = {
  rice: [
    "Maintain 2–5 cm water depth during tillering stage.",
    "Apply nitrogen fertilizer in split doses to avoid lodging.",
    "Watch for brown planthopper — yellow sticky traps help monitor.",
    "Harvest when 80–85% of grains turn golden-yellow.",
  ],
  tomato: [
    "Water at the base; avoid wetting foliage to prevent blight.",
    "Stake plants early to support fruit weight.",
    "Remove suckers weekly to improve air circulation.",
    "Apply calcium to prevent blossom-end rot.",
  ],
  potato: [
    "Hill soil around stems when plants reach 20 cm to prevent greening.",
    "Avoid over-watering — waterlogged soil causes tuber rot.",
    "Rotate with non-solanaceous crops each season.",
    "Cure harvested tubers at 15–20°C for 2 weeks before storage.",
  ],
  wheat: [
    "Sow at optimal density (~100 kg/ha) to avoid lodging.",
    "Apply phosphorus at planting for strong root development.",
    "First irrigation at crown root initiation stage is critical.",
    "Monitor for rust — apply fungicide at first sign of infection.",
  ],
  jute: [
    "Sow seeds in well-drained loamy soil in April–May.",
    "Thin seedlings to 5–7 cm spacing after germination.",
    "Ret harvested stalks in slow-moving water for 15–20 days.",
    "Avoid waterlogging — use raised beds in flood-prone areas.",
  ],
};

// ---------------------------------------------------------------------------
// Markdown-lite renderer (bold + line breaks)
// ---------------------------------------------------------------------------

function renderAdvisoryText(text: string): React.ReactNode[] {
  return text.split("\n").flatMap((line, li, lines) => {
    const parts: React.ReactNode[] = [];
    const boldRegex = /\*\*(.+?)\*\*/g;
    let last = 0;
    let m: RegExpExecArray | null;

    while ((m = boldRegex.exec(line)) !== null) {
      if (m.index > last) parts.push(line.slice(last, m.index));
      parts.push(
        <strong key={`b-${li}-${m.index}`} className="font-semibold text-gray-900">
          {m[1]}
        </strong>,
      );
      last = boldRegex.lastIndex;
    }
    if (last < line.length) parts.push(line.slice(last));
    if (li < lines.length - 1) parts.push(<br key={`br-${li}`} />);
    return parts;
  });
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function CropCard({
  crop,
  selected,
  onClick,
}: {
  crop: Crop;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex flex-col items-center gap-1.5 px-5 py-4 rounded-2xl border-2 shrink-0 transition-colors ${
        selected
          ? "bg-primary-600 border-primary-600 text-white shadow-md"
          : "bg-white border-gray-200 text-gray-700 hover:border-primary-400 hover:bg-primary-50"
      }`}
    >
      <span className="text-3xl" role="img" aria-label={crop.name}>
        {crop.icon}
      </span>
      <span className="text-sm font-semibold leading-tight">{crop.name}</span>
      <span className={`text-xs ${selected ? "text-primary-200" : "text-gray-400"}`}>
        {crop.name_bn}
      </span>
    </button>
  );
}

function TopicTab({
  label,
  icon: Icon,
  active,
  onClick,
}: {
  label: string;
  icon: LucideIcon;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`whitespace-nowrap flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
        active
          ? "border-primary-600 text-primary-700"
          : "border-transparent text-gray-500 hover:text-primary-600 hover:border-primary-300"
      }`}
    >
      <Icon size={15} strokeWidth={2} />
      {label}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function AdvisoryPage() {
  const [crops, setCrops] = useState<Crop[]>([]);
  const [selectedCrop, setSelectedCrop] = useState<Crop | null>(null);
  const [selectedTopic, setSelectedTopic] = useState<TopicId>("irrigation");
  const [advisory, setAdvisory] = useState<AdvisoryResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cropsError, setCropsError] = useState(false);

  // ── Fetch crops on mount ───────────────────────────────────────────────
  useEffect(() => {
    getCrops()
      .then(setCrops)
      .catch(() => setCropsError(true));
  }, []);

  // ── Fetch advisory whenever crop or topic changes ──────────────────────
  const fetchAdvisory = useCallback(
    async (crop: Crop, topic: TopicId) => {
      setIsLoading(true);
      setError(null);
      setAdvisory(null);
      try {
        const result = await getAdvisory(crop.id, topic);
        setAdvisory(result);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Failed to get advisory. Please try again.",
        );
      } finally {
        setIsLoading(false);
      }
    },
    [],
  );

  function handleCropSelect(crop: Crop) {
    setSelectedCrop(crop);
    setAdvisory(null);
    setError(null);
    fetchAdvisory(crop, selectedTopic);
  }

  function handleTopicSelect(topic: TopicId) {
    setSelectedTopic(topic);
    if (selectedCrop) fetchAdvisory(selectedCrop, topic);
  }

  const tips = selectedCrop ? (QUICK_TIPS[selectedCrop.id] ?? []) : [];

  // ── Render ─────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-6xl mx-auto space-y-6">

        {/* ── Page header ── */}
        <div>
          <div className="flex items-center gap-3 mb-1">
            <div className="inline-flex items-center justify-center w-10 h-10 rounded-xl bg-primary-100 border border-primary-200">
              <BookOpen size={20} className="text-primary-700" strokeWidth={1.75} />
            </div>
            <h1 className="text-2xl sm:text-3xl font-bold text-gray-900">Crop Advisory</h1>
          </div>
          <p className="text-gray-500 text-base ml-[3.25rem]">
            Select a crop and a topic to get expert, AI-generated farming guidance.
          </p>
        </div>

        {/* ── Weather risk banner ── */}
        <WeatherRiskCard compact />

        {/* ── Crop selector ── */}
        <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-5">
          <h2 className="text-sm font-semibold text-gray-700 mb-4">Select a Crop</h2>

          {cropsError ? (
            <p className="text-sm text-red-500">
              Could not load crops. Make sure the backend is running.
            </p>
          ) : crops.length === 0 ? (
            <div className="flex gap-3">
              {[1, 2, 3, 4, 5].map((n) => (
                <div
                  key={n}
                  className="w-24 h-24 rounded-2xl bg-gray-100 animate-pulse shrink-0"
                />
              ))}
            </div>
          ) : (
            <div className="flex gap-3 overflow-x-auto pb-1 scrollbar-none">
              {crops.map((crop) => (
                <CropCard
                  key={crop.id}
                  crop={crop}
                  selected={selectedCrop?.id === crop.id}
                  onClick={() => handleCropSelect(crop)}
                />
              ))}
            </div>
          )}
        </div>

        {/* ── Topic tabs + content + sidebar ── */}
        {selectedCrop ? (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">

            {/* Main column */}
            <div className="lg:col-span-2 space-y-4">

              {/* Topic tab bar */}
              <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
                <div className="flex overflow-x-auto border-b border-gray-100 px-2 scrollbar-none">
                  {TOPICS.map(({ id, label, icon }) => (
                    <TopicTab
                      key={id}
                      label={label}
                      icon={icon}
                      active={selectedTopic === id}
                      onClick={() => handleTopicSelect(id as TopicId)}
                    />
                  ))}
                </div>

                {/* Advisory content */}
                <div className="p-6">
                  {isLoading && (
                    <div className="space-y-3 py-4 animate-pulse" aria-label="Loading advisory">
                      <div className="h-3 w-full rounded bg-gray-200" />
                      <div className="h-3 w-5/6 rounded bg-gray-200" />
                      <div className="h-3 w-4/6 rounded bg-gray-200" />
                    </div>
                  )}

                  {!isLoading && error && (
                    <div className="rounded-xl bg-red-50 border border-red-200 px-5 py-4 text-sm text-red-600">
                      {error}
                    </div>
                  )}

                  {!isLoading && advisory && (
                    <div className="border-l-4 border-primary-300 pl-4 text-gray-700 leading-relaxed text-sm">
                      {renderAdvisoryText(advisory.content)}
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Quick tips sidebar — desktop only */}
            {tips.length > 0 && (
              <aside className="hidden lg:block space-y-4">
                <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-5">
                  <div className="flex items-center gap-2 mb-4">
                    <Sprout size={16} className="text-primary-600" strokeWidth={2} />
                    <h3 className="text-sm font-semibold text-gray-800">
                      Quick Tips — {selectedCrop.name}
                    </h3>
                  </div>
                  <ul className="space-y-3">
                    {tips.map((tip, i) => (
                      <li key={i} className="flex items-start gap-2.5">
                        <ChevronRight
                          size={14}
                          className="text-primary-500 mt-0.5 shrink-0"
                          strokeWidth={2.5}
                        />
                        <span className="text-sm text-gray-600 leading-snug">{tip}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Selected context pill */}
                <div className="bg-primary-50 border border-primary-100 rounded-2xl px-5 py-4 text-center">
                  <div className="text-3xl mb-2">{selectedCrop.icon}</div>
                  <p className="text-xs font-semibold text-primary-700">
                    {selectedCrop.name} · {TOPICS.find((t) => t.id === selectedTopic)?.label}
                  </p>
                  <p className="text-xs text-primary-500 mt-0.5">{selectedCrop.name_bn}</p>
                </div>
              </aside>
            )}
          </div>
        ) : (
          /* No crop selected placeholder */
          <div className="bg-white rounded-2xl border-2 border-dashed border-gray-200 flex flex-col items-center justify-center gap-4 py-20 text-center">
            <div className="text-5xl">🌾</div>
            <div>
              <p className="font-semibold text-gray-700">Select a crop to get started</p>
              <p className="text-sm text-gray-400 mt-1">
                Choose a crop above, then pick a topic to see expert advice.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
