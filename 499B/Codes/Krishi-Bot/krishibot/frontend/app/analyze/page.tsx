"use client";

import { useState, useCallback } from "react";
import {
  Camera,
  ImageIcon,
  Loader2,
  AlertTriangle,
  Lightbulb,
  CheckCircle,
} from "lucide-react";
import ImageUploader from "@/components/ImageUploader";
import DiseaseReportCard from "@/components/DiseaseReportCard";
import { analyzeImage, type AnalysisResponse } from "@/lib/api";

// ---------------------------------------------------------------------------
// Static tips data
// ---------------------------------------------------------------------------

const TIPS = [
  "Use clear, well-lit photos — avoid shadows over the affected area.",
  "Focus on the affected leaves or stem as closely as possible.",
  "Include both healthy and diseased parts in the same frame if possible.",
  "Supported formats: JPG, PNG, WEBP (max 10 MB).",
] as const;

// ---------------------------------------------------------------------------
// Skeleton loader
// ---------------------------------------------------------------------------

function SkeletonCard() {
  return (
    <div className="rounded-2xl border border-gray-200 overflow-hidden animate-pulse">
      {/* Header skeleton */}
      <div className="px-6 py-5 border-b bg-gray-50 space-y-3">
        <div className="flex items-center justify-between">
          <div className="h-6 w-36 rounded-full bg-gray-200" />
          <div className="h-6 w-20 rounded-full bg-gray-200" />
        </div>
        <div className="h-7 w-56 rounded-lg bg-gray-200" />
      </div>
      {/* Body skeleton */}
      <div className="px-6 py-5 space-y-5 bg-white">
        <div className="space-y-2">
          <div className="h-3 w-full rounded bg-gray-100" />
          <div className="h-2 w-full rounded-full bg-gray-100">
            <div className="h-2 w-3/4 rounded-full bg-gray-200" />
          </div>
        </div>
        {[1, 2, 3].map((n) => (
          <div key={n} className="space-y-2">
            <div className="h-4 w-40 rounded bg-gray-200" />
            {[1, 2].map((m) => (
              <div key={m} className="h-12 w-full rounded-xl bg-gray-100" />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function AnalyzePage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResponse | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleImageSelected = useCallback((file: File) => {
    setSelectedFile(file);
    // Clear previous results when a new image is chosen.
    setAnalysisResult(null);
    setError(null);
  }, []);

  async function handleAnalyze() {
    if (!selectedFile || isAnalyzing) return;

    setIsAnalyzing(true);
    setError(null);
    setAnalysisResult(null);

    try {
      const result = await analyzeImage(selectedFile);
      setAnalysisResult(result);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Analysis failed. Make sure the backend and Ollama are running.",
      );
    } finally {
      setIsAnalyzing(false);
    }
  }

  // ── Analyze button label / state ─────────────────────────────────────
  const canAnalyze = !!selectedFile && !isAnalyzing;

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-6xl mx-auto">

        {/* ── Page header ── */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="inline-flex items-center justify-center w-10 h-10 rounded-xl bg-primary-100 border border-primary-200">
              <Camera size={20} className="text-primary-700" strokeWidth={1.75} />
            </div>
            <h1 className="text-2xl sm:text-3xl font-bold text-gray-900">
              Crop Disease Detection
            </h1>
          </div>
          <p className="text-gray-500 text-base ml-[3.25rem]">
            Upload a crop or leaf photo — AI will detect likely diseases
            and provide treatment recommendations
          </p>
        </div>

        {/* ── Two-column layout ── */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">

          {/* ============================================================
              LEFT COLUMN — upload controls
          ============================================================ */}
          <div className="space-y-6">

            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-6 space-y-5">
              <h2 className="text-base font-semibold text-gray-800">Upload Crop Image</h2>

              <ImageUploader
                onImageSelected={handleImageSelected}
                isAnalyzing={isAnalyzing}
              />

              {/* Analyze button */}
              <button
                type="button"
                onClick={handleAnalyze}
                disabled={!canAnalyze}
                className={`w-full flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl font-semibold text-sm transition-colors ${
                  isAnalyzing
                    ? "bg-primary-400 text-white cursor-not-allowed"
                    : canAnalyze
                    ? "bg-primary-600 text-white hover:bg-primary-700 shadow-sm"
                    : "bg-gray-100 text-gray-400 cursor-not-allowed"
                }`}
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 size={17} className="animate-spin" />
                    Analyzing…
                  </>
                ) : (
                  <>
                    <Camera size={17} />
                    Analyze Image
                  </>
                )}
              </button>
            </div>

            {/* Tips card */}
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-6">
              <div className="flex items-center gap-2 mb-4">
                <Lightbulb size={16} className="text-accent-500" strokeWidth={2} />
                <h2 className="text-sm font-semibold text-gray-800">
                  Tips for Best Results
                </h2>
              </div>
              <ul className="space-y-2.5">
                {TIPS.map((tip, i) => (
                  <li key={i} className="flex items-start gap-2.5 text-sm text-gray-600">
                    <CheckCircle
                      size={14}
                      className="text-primary-500 mt-0.5 shrink-0"
                      strokeWidth={2.5}
                    />
                    {tip}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* ============================================================
              RIGHT COLUMN — result area
          ============================================================ */}
          <div>
            {/* Skeleton while analyzing */}
            {isAnalyzing && <SkeletonCard />}

            {/* Result card */}
            {!isAnalyzing && analysisResult && (
              <DiseaseReportCard result={analysisResult} />
            )}

            {/* Error card */}
            {!isAnalyzing && error && (
              <div className="rounded-2xl border border-red-200 bg-red-50 px-6 py-5 flex items-start gap-4">
                <div className="shrink-0 flex items-center justify-center w-10 h-10 rounded-xl bg-red-100 border border-red-200">
                  <AlertTriangle size={20} className="text-red-600" strokeWidth={1.75} />
                </div>
                <div className="flex-1">
                  <h3 className="text-sm font-bold text-red-800 mb-1">Analysis Failed</h3>
                  <p className="text-sm text-red-600 leading-relaxed">{error}</p>
                  <p className="text-xs text-red-400 mt-2">
                    Make sure the backend server and Ollama are both running, then try again.
                  </p>
                  <button
                    type="button"
                    onClick={handleAnalyze}
                    disabled={!selectedFile}
                    className="mt-4 inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-amber-500 text-white text-xs font-semibold hover:bg-amber-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-sm"
                  >
                    Try Again
                  </button>
                </div>
              </div>
            )}

            {/* Placeholder — no file selected or not yet analyzed */}
            {!isAnalyzing && !analysisResult && !error && (
              <div className="rounded-2xl border-2 border-dashed border-gray-200 bg-white flex flex-col items-center justify-center gap-4 px-8 py-16 text-center">
                <div className="flex items-center justify-center w-16 h-16 rounded-2xl bg-gray-100 border border-gray-200">
                  <ImageIcon size={28} className="text-gray-400" strokeWidth={1.5} />
                </div>
                <div>
                  <p className="text-gray-600 font-semibold text-sm">
                    No analysis yet
                  </p>
                  <p className="text-gray-400 text-xs mt-1">
                    Upload an image and click{" "}
                    <span className="font-medium text-primary-600">Analyze Image</span>{" "}
                    to see the disease report here.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
