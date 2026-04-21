"use client";

import { useState } from "react";
import {
  AlertTriangle,
  CheckCircle,
  AlertCircle,
  Shield,
  Pill,
  Download,
  Loader2,
} from "lucide-react";
import type { AnalysisResponse } from "@/lib/api";

const API_BASE = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

interface Props {
  result: AnalysisResponse;
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function ConfidenceBar({ confidence }: { confidence: "High" | "Medium" | "Low" }) {
  const config = {
    High:   { width: "w-[88%]",  bg: "bg-primary-500", text: "text-primary-700", label: "High Confidence"   },
    Medium: { width: "w-[56%]",  bg: "bg-amber-400",   text: "text-amber-700",   label: "Medium Confidence" },
    Low:    { width: "w-[24%]",  bg: "bg-red-400",     text: "text-red-700",     label: "Low Confidence"    },
  }[confidence];

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between text-xs font-medium">
        <span className="text-gray-500">Confidence Level</span>
        <span className={config.text}>{config.label}</span>
      </div>
      <div className="h-2 w-full rounded-full bg-gray-100 overflow-hidden">
        <div
          className={`h-full rounded-full ${config.bg} ${config.width} transition-all duration-700 ease-out`}
        />
      </div>
    </div>
  );
}

function SectionHeading({
  icon: Icon,
  label,
  iconClass,
}: {
  icon: React.ElementType;
  label: string;
  iconClass: string;
}) {
  return (
    <div className="flex items-center gap-2 mb-3">
      <Icon size={16} className={iconClass} strokeWidth={2} />
      <h3 className="text-sm font-bold text-gray-800">{label}</h3>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function DiseaseReportCard({ result }: Props) {
  const { is_diseased, disease_name, confidence, symptoms, treatment, prevention } = result;
  const [isDownloading, setIsDownloading] = useState(false);

  async function handleDownloadPDF() {
    setIsDownloading(true);
    try {
      const response = await fetch(`${API_BASE}/api/analyze/report/pdf`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          disease_name,
          is_diseased,
          confidence,
          symptoms,
          treatment,
          prevention,
          filename: "crop_image",
        }),
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const blob = await response.blob();
      const url  = window.URL.createObjectURL(blob);
      const a    = document.createElement("a");
      a.href     = url;
      a.download = `krishibot_report_${Date.now()}.pdf`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error("PDF download failed:", error);
    } finally {
      setIsDownloading(false);
    }
  }

  return (
    <div className="rounded-2xl border border-gray-200 overflow-hidden shadow-sm bg-white">

      {/* ── Header ── */}
      <div
        className={`px-6 py-5 border-b ${
          is_diseased
            ? "bg-red-50 border-red-100"
            : "bg-primary-50 border-primary-100"
        }`}
      >
        <div className="flex items-center justify-between flex-wrap gap-3 mb-3">
          <span
            className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold ${
              is_diseased
                ? "bg-red-100 text-red-700 border border-red-200"
                : "bg-primary-100 text-primary-700 border border-primary-200"
            }`}
          >
            {is_diseased ? (
              <AlertTriangle size={12} strokeWidth={2.5} />
            ) : (
              <CheckCircle size={12} strokeWidth={2.5} />
            )}
            {is_diseased ? "Disease Detected" : "Healthy Crop"}
          </span>

          <span
            className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold border ${
              confidence === "High"
                ? "bg-primary-100 text-primary-700 border-primary-200"
                : confidence === "Medium"
                ? "bg-amber-100 text-amber-700 border-amber-200"
                : "bg-red-100 text-red-700 border-red-200"
            }`}
          >
            {confidence}
          </span>
        </div>

        <h2 className="text-xl font-bold text-gray-900">{disease_name}</h2>
      </div>

      {/* ── Body ── */}
      <div className="px-6 py-5 space-y-6">

        <ConfidenceBar confidence={confidence} />

        {symptoms.length > 0 && (
          <div>
            <SectionHeading icon={AlertCircle} label="Symptoms Observed" iconClass="text-amber-500" />
            <ul className="space-y-2">
              {symptoms.map((symptom, i) => (
                <li key={i} className="flex items-start gap-2.5 text-sm text-gray-700">
                  <span className="mt-1.5 shrink-0 w-1.5 h-1.5 rounded-full bg-amber-400" />
                  {symptom}
                </li>
              ))}
            </ul>
          </div>
        )}

        {treatment.length > 0 && (
          <div>
            <SectionHeading icon={Pill} label="Recommended Treatment" iconClass="text-red-500" />
            <ol className="space-y-2">
              {treatment.map((step, i) => (
                <li
                  key={i}
                  className="flex items-start gap-3 px-4 py-3 rounded-xl bg-red-50 border border-red-100 text-sm text-gray-800"
                >
                  <span className="shrink-0 flex items-center justify-center w-5 h-5 rounded-full bg-red-200 text-red-700 text-xs font-bold mt-0.5">
                    {i + 1}
                  </span>
                  {step}
                </li>
              ))}
            </ol>
          </div>
        )}

        {prevention.length > 0 && (
          <div>
            <SectionHeading icon={Shield} label="Prevention Tips" iconClass="text-primary-600" />
            <ul className="space-y-2">
              {prevention.map((tip, i) => (
                <li
                  key={i}
                  className="flex items-start gap-3 px-4 py-3 rounded-xl bg-primary-50 border border-primary-100 text-sm text-gray-800"
                >
                  <CheckCircle size={15} className="text-primary-500 mt-0.5 shrink-0" strokeWidth={2.5} />
                  {tip}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Disclaimer */}
        <div className="rounded-xl bg-amber-50 border border-amber-200 px-4 py-3 flex items-start gap-2.5">
          <AlertTriangle size={14} className="text-amber-600 mt-0.5 shrink-0" strokeWidth={2} />
          <p className="text-xs italic text-amber-800 leading-relaxed">
            This is an AI-assisted analysis, not a confirmed diagnosis. Always consult
            a local agricultural officer or plant pathologist for verification.
          </p>
        </div>

        {/* Download PDF button */}
        <button
          type="button"
          onClick={handleDownloadPDF}
          disabled={isDownloading}
          className="w-full inline-flex items-center justify-center gap-2 px-5 py-3 rounded-xl bg-green-700 hover:bg-green-800 text-white font-semibold text-sm transition-colors shadow-sm disabled:opacity-60 disabled:cursor-not-allowed"
        >
          {isDownloading ? (
            <>
              <Loader2 size={16} className="animate-spin" />
              Generating PDF…
            </>
          ) : (
            <>
              <Download size={16} />
              Download PDF Report
            </>
          )}
        </button>

      </div>
    </div>
  );
}
