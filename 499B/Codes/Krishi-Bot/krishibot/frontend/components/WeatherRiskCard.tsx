"use client";

import { useEffect, useState } from "react";
import {
  CloudSun,
  Thermometer,
  Droplets,
  AlertTriangle,
  ShieldCheck,
  Info,
} from "lucide-react";
import {
  getWeatherRisk,
  type WeatherRiskResponse,
  type WeatherRiskAlert,
  type WeatherRiskLevel,
} from "@/lib/api";

// ---------------------------------------------------------------------------
// Risk-level styling
// ---------------------------------------------------------------------------

const RISK_STYLES: Record<
  WeatherRiskLevel,
  {
    border: string;
    badge:  string;
    text:   string;
    icon:   typeof AlertTriangle;
    label:  string;
  }
> = {
  HIGH: {
    border: "border-l-red-500",
    badge:  "bg-red-100 text-red-700 border border-red-200",
    text:   "text-red-700",
    icon:   AlertTriangle,
    label:  "HIGH",
  },
  MEDIUM: {
    border: "border-l-amber-500",
    badge:  "bg-amber-100 text-amber-700 border border-amber-200",
    text:   "text-amber-700",
    icon:   Info,
    label:  "MEDIUM",
  },
  LOW: {
    border: "border-l-primary-500",
    badge:  "bg-primary-100 text-primary-700 border border-primary-200",
    text:   "text-primary-700",
    icon:   ShieldCheck,
    label:  "LOW",
  },
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface Props {
  lat?: number;
  lon?: number;
  /** Compact variant — smaller header, only risk badges (for advisory page). */
  compact?: boolean;
}

export default function WeatherRiskCard({ lat, lon, compact = false }: Props) {
  const [data, setData] = useState<WeatherRiskResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);

  useEffect(() => {
    let cancelled = false;

    setIsLoading(true);
    setHasError(false);

    getWeatherRisk(lat, lon)
      .then((res) => {
        if (!cancelled) setData(res);
      })
      .catch(() => {
        if (!cancelled) setHasError(true);
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [lat, lon]);

  // ── Loading ─────────────────────────────────────────────────────────────
  if (isLoading) {
    return compact ? (
      <CompactSkeleton />
    ) : (
      <FullSkeleton />
    );
  }

  // ── Error ───────────────────────────────────────────────────────────────
  if (hasError || !data) {
    return (
      <div className="rounded-2xl border border-gray-200 bg-white px-5 py-4 text-sm text-gray-400 text-center">
        Weather data unavailable
      </div>
    );
  }

  // ── Compact variant — risk badges only ──────────────────────────────────
  if (compact) {
    return (
      <div className="rounded-2xl border border-gray-200 bg-white shadow-sm px-5 py-4">
        <div className="flex items-center gap-2 mb-3">
          <CloudSun size={16} className="text-primary-600" strokeWidth={2} />
          <h3 className="text-sm font-semibold text-gray-800">
            Weather Disease Risk — Next 48 h
          </h3>
          <span className="ml-auto text-xs text-gray-400">
            {data.current.temperature.toFixed(0)}°C · {data.current.humidity.toFixed(0)}%
          </span>
        </div>

        <div className="flex flex-wrap gap-2">
          {data.risks.map((alert, i) => {
            const style = RISK_STYLES[alert.risk];
            return (
              <span
                key={i}
                className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold ${style.badge}`}
                title={alert.reason}
              >
                <style.icon size={12} strokeWidth={2.5} />
                {alert.risk} · {alert.disease}
              </span>
            );
          })}
        </div>
      </div>
    );
  }

  // ── Full variant ────────────────────────────────────────────────────────
  return (
    <div className="rounded-2xl border border-gray-200 bg-white shadow-sm overflow-hidden">

      {/* ── Header ── */}
      <div className="flex items-center gap-3 px-6 py-5 bg-gradient-to-r from-primary-50 to-white border-b border-gray-100">
        <div className="inline-flex items-center justify-center w-11 h-11 rounded-xl bg-white border border-primary-200 shadow-sm">
          <CloudSun size={22} className="text-primary-700" strokeWidth={1.75} />
        </div>
        <div>
          <h2 className="text-base font-bold text-gray-900">Weather Disease Risk</h2>
          <p className="text-xs text-gray-500">Next 48 hours</p>
        </div>
      </div>

      {/* ── Current conditions ── */}
      <div className="grid grid-cols-2 divide-x divide-gray-100 border-b border-gray-100">
        <div className="flex items-center gap-3 px-6 py-4">
          <Thermometer size={18} className="text-red-500" strokeWidth={2} />
          <div>
            <p className="text-[11px] uppercase tracking-wide font-semibold text-gray-400">
              Temperature
            </p>
            <p className="text-sm font-semibold text-gray-800">
              {data.current.temperature.toFixed(1)}°C{" "}
              <span className="text-xs font-normal text-gray-400">
                (48h avg {data.forecast_summary.avg_temp.toFixed(1)}°C)
              </span>
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 px-6 py-4">
          <Droplets size={18} className="text-blue-500" strokeWidth={2} />
          <div>
            <p className="text-[11px] uppercase tracking-wide font-semibold text-gray-400">
              Humidity
            </p>
            <p className="text-sm font-semibold text-gray-800">
              {data.current.humidity.toFixed(0)}%{" "}
              <span className="text-xs font-normal text-gray-400">
                (48h avg {data.forecast_summary.avg_humidity.toFixed(0)}%)
              </span>
            </p>
          </div>
        </div>
      </div>

      {/* ── Risk alerts ── */}
      <div className="px-6 py-5 space-y-3">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-500">
          Risk Alerts
        </h3>

        {data.risks.map((alert, i) => (
          <RiskRow key={i} alert={alert} />
        ))}
      </div>

      {/* ── Footer ── */}
      <div className="px-6 py-3 bg-gray-50 border-t border-gray-100">
        <p className="text-[11px] text-gray-400">
          Based on Open-Meteo forecast data · Updated{" "}
          {new Date(data.updated_at).toLocaleTimeString([], {
            hour:   "2-digit",
            minute: "2-digit",
          })}
        </p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function RiskRow({ alert }: { alert: WeatherRiskAlert }) {
  const style = RISK_STYLES[alert.risk];
  return (
    <div
      className={`border-l-4 ${style.border} bg-gray-50 rounded-r-xl px-4 py-3 flex items-start gap-3`}
    >
      <style.icon size={18} className={`${style.text} mt-0.5 shrink-0`} strokeWidth={2} />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap mb-1">
          <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${style.badge}`}>
            {style.label}
          </span>
          <span className="text-sm font-bold text-gray-900">{alert.disease}</span>
          <span className="text-xs text-gray-500">· {alert.crop}</span>
        </div>
        <p className="text-xs text-gray-600 leading-relaxed">{alert.reason}</p>
      </div>
    </div>
  );
}

function FullSkeleton() {
  return (
    <div className="rounded-2xl border border-gray-200 bg-white shadow-sm overflow-hidden animate-pulse">
      <div className="h-[72px] bg-gray-100 border-b border-gray-100" />
      <div className="grid grid-cols-2 divide-x divide-gray-100 border-b border-gray-100">
        <div className="h-16 bg-gray-50" />
        <div className="h-16 bg-gray-50" />
      </div>
      <div className="px-6 py-5 space-y-3">
        <div className="h-3 w-24 rounded bg-gray-200" />
        <div className="h-14 rounded-xl bg-gray-100" />
        <div className="h-14 rounded-xl bg-gray-100" />
      </div>
    </div>
  );
}

function CompactSkeleton() {
  return (
    <div className="rounded-2xl border border-gray-200 bg-white shadow-sm px-5 py-4 animate-pulse">
      <div className="flex items-center gap-2 mb-3">
        <div className="h-4 w-4 rounded bg-gray-200" />
        <div className="h-4 w-48 rounded bg-gray-200" />
      </div>
      <div className="flex gap-2">
        <div className="h-6 w-32 rounded-full bg-gray-100" />
        <div className="h-6 w-28 rounded-full bg-gray-100" />
      </div>
    </div>
  );
}
