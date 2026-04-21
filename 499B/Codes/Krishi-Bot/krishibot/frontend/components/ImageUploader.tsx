"use client";

import { useCallback, useRef, useState } from "react";
import { Upload, ImageIcon, X, RefreshCw, Loader2 } from "lucide-react";

interface Props {
  onImageSelected: (file: File) => void;
  isAnalyzing: boolean;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function ImageUploader({ onImageSelected, isAnalyzing }: Props) {
  const [preview, setPreview] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [typeError, setTypeError] = useState<string | null>(null);

  const inputRef = useRef<HTMLInputElement>(null);

  // ── File acceptance ────────────────────────────────────────────────────
  const acceptFile = useCallback(
    (file: File) => {
      setTypeError(null);

      if (!file.type.startsWith("image/")) {
        setTypeError(`"${file.name}" is not an image file. Please upload a JPG, PNG, or WEBP.`);
        return;
      }

      // Revoke previous object URL to avoid memory leaks.
      if (preview) URL.revokeObjectURL(preview);

      const url = URL.createObjectURL(file);
      setPreview(url);
      setSelectedFile(file);
      onImageSelected(file);
    },
    [preview, onImageSelected],
  );

  // ── Input change ───────────────────────────────────────────────────────
  function handleInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) acceptFile(file);
    // Reset so the same file can be re-selected.
    e.target.value = "";
  }

  // ── Drag events ────────────────────────────────────────────────────────
  function handleDragOver(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  }

  function handleDragLeave(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) acceptFile(file);
  }

  // ── Clear selection ────────────────────────────────────────────────────
  function handleClear() {
    if (preview) URL.revokeObjectURL(preview);
    setPreview(null);
    setSelectedFile(null);
    setTypeError(null);
  }

  // ── Drop zone classes ──────────────────────────────────────────────────
  const dropzoneClass = [
    "relative flex flex-col items-center justify-center w-full rounded-2xl border-2 border-dashed transition-colors cursor-pointer",
    isDragOver
      ? "border-primary-500 bg-primary-50"
      : "border-primary-200 bg-primary-50/40 hover:border-primary-400 hover:bg-primary-50",
  ].join(" ");

  // ---------------------------------------------------------------------------
  // Render: preview state
  // ---------------------------------------------------------------------------
  if (preview && selectedFile) {
    return (
      <div className="flex flex-col gap-3">
        {/* Image preview with optional analyzing overlay */}
        <div className="relative rounded-xl overflow-hidden border border-gray-200 bg-gray-50">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={preview}
            alt="Selected crop"
            className="w-full max-h-64 object-contain mx-auto block"
          />

          {/* Analyzing overlay */}
          {isAnalyzing && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-black/50 rounded-xl">
              <Loader2 size={36} className="text-white animate-spin" />
              <span className="text-white text-sm font-medium">Analyzing image…</span>
            </div>
          )}

          {/* Clear button — hidden while analyzing */}
          {!isAnalyzing && (
            <button
              type="button"
              onClick={handleClear}
              className="absolute top-2 right-2 flex items-center justify-center w-7 h-7 rounded-full bg-black/60 text-white hover:bg-black/80 transition-colors"
              aria-label="Remove image"
            >
              <X size={14} />
            </button>
          )}
        </div>

        {/* File info + change button */}
        <div className="flex items-center justify-between px-1">
          <div className="flex items-center gap-2 text-sm text-gray-500 min-w-0">
            <ImageIcon size={14} className="shrink-0 text-primary-500" />
            <span className="truncate">{selectedFile.name}</span>
            <span className="shrink-0 text-gray-400">· {formatBytes(selectedFile.size)}</span>
          </div>

          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            disabled={isAnalyzing}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-primary-700 border border-primary-200 hover:bg-primary-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shrink-0 ml-3"
          >
            <RefreshCw size={12} />
            Change Image
          </button>
        </div>

        {/* Hidden file input for "Change Image" */}
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleInputChange}
        />
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // Render: drop zone (no image selected yet)
  // ---------------------------------------------------------------------------
  return (
    <div className="flex flex-col gap-2">
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload crop image"
        className={dropzoneClass}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="flex flex-col items-center gap-4 px-6 py-12">
          <div className="flex items-center justify-center w-16 h-16 rounded-2xl bg-primary-100 border border-primary-200">
            <Upload size={28} className="text-primary-600" strokeWidth={1.75} />
          </div>

          <div className="text-center">
            <p className="font-semibold text-gray-800 text-sm">
              {isDragOver ? "Drop the image here" : "Drag & drop an image here"}
            </p>
            <p className="text-gray-400 text-xs mt-1">or click to browse files</p>
          </div>

          <span className="text-xs text-gray-400 border border-gray-200 rounded-lg px-3 py-1 bg-white">
            JPG · PNG · WEBP · BMP
          </span>
        </div>

        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleInputChange}
        />
      </div>

      {/* Type error */}
      {typeError && (
        <p className="text-xs text-red-600 bg-red-50 border border-red-200 rounded-lg px-3 py-2">
          {typeError}
        </p>
      )}
    </div>
  );
}
