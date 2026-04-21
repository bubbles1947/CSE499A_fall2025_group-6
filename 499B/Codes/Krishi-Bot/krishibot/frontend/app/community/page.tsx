"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  Calendar,
  CheckCircle2,
  Filter,
  Leaf,
  MapPin,
  Send,
  Sprout,
  Users,
  X,
} from "lucide-react";
import {
  CommunityReport,
  CommunityStats,
  getCommunityReports,
  getCommunityStats,
  getDivisions,
  seedCommunityReports,
  submitCommunityReport,
} from "@/lib/api";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const SEVERITIES = ["Low", "Medium", "High"] as const;
type Severity = (typeof SEVERITIES)[number];

const SEVERITY_STYLES: Record<string, string> = {
  High:   "bg-red-100 text-red-800 border-red-200",
  Medium: "bg-amber-100 text-amber-800 border-amber-200",
  Low:    "bg-green-100 text-green-800 border-green-200",
};

function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60_000);
  if (mins < 1)    return "just now";
  if (mins < 60)   return `${mins} min ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24)    return `${hrs} hr${hrs === 1 ? "" : "s"} ago`;
  const days = Math.floor(hrs / 24);
  if (days < 30)   return `${days} day${days === 1 ? "" : "s"} ago`;
  const months = Math.floor(days / 30);
  return `${months} mo${months === 1 ? "" : "s"} ago`;
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function CommunityPage() {
  const [reports,   setReports]   = useState<CommunityReport[]>([]);
  const [stats,     setStats]     = useState<CommunityStats | null>(null);
  const [divisions, setDivisions] = useState<string[]>([]);

  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState<string | null>(null);

  // Filters
  const [filterDivision, setFilterDivision] = useState<string>("");
  const [filterCrop,     setFilterCrop]     = useState<string>("");

  // Submission form
  const [formOpen,    setFormOpen]    = useState(false);
  const [submitting,  setSubmitting]  = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitOk,    setSubmitOk]    = useState(false);

  const [form, setForm] = useState({
    disease_name:  "",
    crop_name:     "",
    location_name: "",
    division:      "Dhaka",
    severity:      "Medium" as Severity,
    notes:         "",
  });

  // --- data loading -------------------------------------------------------

  const loadAll = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [reps, st, divs] = await Promise.all([
        getCommunityReports({
          division: filterDivision || undefined,
          crop:     filterCrop     || undefined,
          limit:    30,
        }),
        getCommunityStats(),
        getDivisions(),
      ]);
      setReports(reps);
      setStats(st);
      setDivisions(divs);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load community board.");
    } finally {
      setLoading(false);
    }
  }, [filterDivision, filterCrop]);

  useEffect(() => {
    void loadAll();
  }, [loadAll]);

  // --- actions ------------------------------------------------------------

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setSubmitError(null);
    setSubmitOk(false);
    try {
      await submitCommunityReport({
        disease_name:  form.disease_name.trim(),
        crop_name:     form.crop_name.trim(),
        location_name: form.location_name.trim(),
        division:      form.division,
        severity:      form.severity,
        notes:         form.notes.trim() || null,
      });
      setSubmitOk(true);
      setForm({
        disease_name:  "",
        crop_name:     "",
        location_name: "",
        division:      "Dhaka",
        severity:      "Medium",
        notes:         "",
      });
      await loadAll();
      // Auto-close form after a short confirmation moment
      setTimeout(() => {
        setFormOpen(false);
        setSubmitOk(false);
      }, 1500);
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : "Failed to submit report.");
    } finally {
      setSubmitting(false);
    }
  }

  async function handleSeed() {
    try {
      await seedCommunityReports();
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to seed sample data.");
    }
  }

  const showSeedButton = useMemo(
    () => !loading && !error && (stats?.total_reports ?? 0) === 0,
    [loading, error, stats],
  );

  // --- render -------------------------------------------------------------

  return (
    <div className="min-h-screen bg-primary-50/40">
      {/* ── Hero ────────────────────────────────────────────────────── */}
      <section className="bg-gradient-to-br from-primary-800 via-primary-700 to-primary-600 text-white">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-14 sm:py-16">
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-primary-600 border border-primary-400 text-primary-100 text-xs font-medium mb-4">
            <Users size={12} />
            Anonymous Community Board
          </span>
          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold tracking-tight mb-3">
            Community Disease Reports
          </h1>
          <p className="text-primary-100 text-lg max-w-2xl">
            See what diseases are spreading around Bangladesh — and anonymously share
            what you&apos;re seeing in your field so others can stay ahead.
          </p>
        </div>
      </section>

      {/* ── Main grid ───────────────────────────────────────────────── */}
      <section className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">

          {/* ============== LEFT: reports (60%) ===================== */}
          <div className="lg:col-span-3 space-y-6">
            {/* Action bar */}
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h2 className="text-xl font-bold text-gray-900 flex items-center gap-2">
                <Leaf size={20} className="text-primary-600" />
                Recent Reports
                {stats && (
                  <span className="text-sm font-normal text-gray-500">
                    ({stats.total_reports} total)
                  </span>
                )}
              </h2>
              <button
                type="button"
                onClick={() => setFormOpen((v) => !v)}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-primary-600 text-white text-sm font-medium hover:bg-primary-700 transition-colors shadow-sm"
              >
                {formOpen ? <X size={16} /> : <Send size={16} />}
                {formOpen ? "Cancel" : "Report a disease"}
              </button>
            </div>

            {/* Submission form */}
            {formOpen && (
              <form
                onSubmit={handleSubmit}
                className="bg-white rounded-2xl border border-primary-200 shadow-sm p-6 space-y-4"
              >
                <h3 className="text-lg font-bold text-gray-900">
                  Share an anonymous report
                </h3>
                <p className="text-xs text-gray-500 -mt-2">
                  No name, no photo, no personal info — just what you&apos;re seeing.
                </p>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Disease name *
                    </label>
                    <input
                      required
                      value={form.disease_name}
                      onChange={(e) => setForm({ ...form, disease_name: e.target.value })}
                      placeholder="e.g. Late Blight"
                      className="w-full px-3 py-2 rounded-lg border border-gray-300 focus:border-primary-500 focus:ring-1 focus:ring-primary-500 outline-none text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Crop *
                    </label>
                    <input
                      required
                      value={form.crop_name}
                      onChange={(e) => setForm({ ...form, crop_name: e.target.value })}
                      placeholder="e.g. Tomato"
                      className="w-full px-3 py-2 rounded-lg border border-gray-300 focus:border-primary-500 focus:ring-1 focus:ring-primary-500 outline-none text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Location (village / town) *
                    </label>
                    <input
                      required
                      value={form.location_name}
                      onChange={(e) => setForm({ ...form, location_name: e.target.value })}
                      placeholder="e.g. Gazipur"
                      className="w-full px-3 py-2 rounded-lg border border-gray-300 focus:border-primary-500 focus:ring-1 focus:ring-primary-500 outline-none text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Division
                    </label>
                    <select
                      value={form.division}
                      onChange={(e) => setForm({ ...form, division: e.target.value })}
                      className="w-full px-3 py-2 rounded-lg border border-gray-300 focus:border-primary-500 focus:ring-1 focus:ring-primary-500 outline-none text-sm bg-white"
                    >
                      {divisions.map((d) => (
                        <option key={d} value={d}>{d}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Severity
                    </label>
                    <div className="flex gap-2">
                      {SEVERITIES.map((s) => (
                        <button
                          key={s}
                          type="button"
                          onClick={() => setForm({ ...form, severity: s })}
                          className={`flex-1 px-3 py-2 rounded-lg border text-sm font-medium transition-colors ${
                            form.severity === s
                              ? SEVERITY_STYLES[s]
                              : "bg-white text-gray-600 border-gray-300 hover:bg-gray-50"
                          }`}
                        >
                          {s}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="sm:col-span-2">
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Notes <span className="text-gray-400 font-normal">(optional, max 200 chars)</span>
                    </label>
                    <textarea
                      value={form.notes}
                      onChange={(e) =>
                        setForm({ ...form, notes: e.target.value.slice(0, 200) })
                      }
                      rows={2}
                      placeholder="Anything useful — spread rate, weather, what you tried…"
                      className="w-full px-3 py-2 rounded-lg border border-gray-300 focus:border-primary-500 focus:ring-1 focus:ring-primary-500 outline-none text-sm resize-none"
                    />
                    <div className="text-right text-xs text-gray-400 mt-1">
                      {form.notes.length}/200
                    </div>
                  </div>
                </div>

                {submitError && (
                  <div className="flex items-start gap-2 text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg px-3 py-2">
                    <AlertTriangle size={16} className="mt-0.5 shrink-0" />
                    <span>{submitError}</span>
                  </div>
                )}
                {submitOk && (
                  <div className="flex items-start gap-2 text-sm text-green-700 bg-green-50 border border-green-200 rounded-lg px-3 py-2">
                    <CheckCircle2 size={16} className="mt-0.5 shrink-0" />
                    <span>Report submitted. Thank you!</span>
                  </div>
                )}

                <div className="flex justify-end gap-2 pt-2">
                  <button
                    type="button"
                    onClick={() => setFormOpen(false)}
                    className="px-4 py-2 rounded-lg text-sm font-medium text-gray-600 hover:bg-gray-100 transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={submitting}
                    className="inline-flex items-center gap-2 px-5 py-2 rounded-lg bg-primary-600 text-white text-sm font-medium hover:bg-primary-700 transition-colors shadow-sm disabled:opacity-60"
                  >
                    <Send size={14} />
                    {submitting ? "Submitting…" : "Submit report"}
                  </button>
                </div>
              </form>
            )}

            {/* Filter bar */}
            <div className="bg-white rounded-xl border border-gray-200 p-4 flex flex-wrap items-center gap-3">
              <Filter size={16} className="text-gray-400" />
              <span className="text-sm font-medium text-gray-600">Filter:</span>
              <select
                value={filterDivision}
                onChange={(e) => setFilterDivision(e.target.value)}
                className="px-3 py-1.5 rounded-lg border border-gray-300 text-sm bg-white focus:border-primary-500 focus:ring-1 focus:ring-primary-500 outline-none"
              >
                <option value="">All divisions</option>
                {divisions.map((d) => (
                  <option key={d} value={d}>{d}</option>
                ))}
              </select>
              <input
                value={filterCrop}
                onChange={(e) => setFilterCrop(e.target.value)}
                placeholder="Crop name…"
                className="px-3 py-1.5 rounded-lg border border-gray-300 text-sm focus:border-primary-500 focus:ring-1 focus:ring-primary-500 outline-none flex-1 min-w-[8rem]"
              />
              {(filterDivision || filterCrop) && (
                <button
                  type="button"
                  onClick={() => { setFilterDivision(""); setFilterCrop(""); }}
                  className="text-xs text-gray-500 hover:text-gray-700 underline"
                >
                  Clear
                </button>
              )}
            </div>

            {/* Reports list */}
            {loading ? (
              <ReportsSkeleton />
            ) : error ? (
              <div className="bg-red-50 border border-red-200 rounded-xl p-6 text-sm text-red-700 flex items-start gap-2">
                <AlertTriangle size={16} className="mt-0.5 shrink-0" />
                <span>{error}</span>
              </div>
            ) : reports.length === 0 ? (
              <div className="bg-white border border-gray-200 rounded-xl p-10 text-center">
                <Sprout size={32} className="mx-auto text-primary-400 mb-3" />
                <p className="text-gray-600 mb-4">
                  No reports yet
                  {(filterDivision || filterCrop) ? " match these filters" : ""}.
                </p>
                {showSeedButton && (
                  <button
                    type="button"
                    onClick={handleSeed}
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-primary-600 text-white text-sm font-medium hover:bg-primary-700 transition-colors"
                  >
                    Load sample data
                  </button>
                )}
              </div>
            ) : (
              <div className="space-y-3">
                {reports.map((r) => (
                  <ReportCard key={r.id} report={r} />
                ))}
              </div>
            )}
          </div>

          {/* ============== RIGHT: stats (40%) ====================== */}
          <aside className="lg:col-span-2 space-y-6">
            <StatsSidebar stats={stats} loading={loading} />
          </aside>
        </div>
      </section>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function ReportCard({ report }: { report: CommunityReport }) {
  const sev = SEVERITY_STYLES[report.severity] ?? SEVERITY_STYLES.Medium;
  return (
    <article className="bg-white rounded-xl border border-gray-200 p-4 hover:shadow-sm transition-shadow">
      <div className="flex items-start justify-between gap-3 mb-2">
        <div className="min-w-0">
          <h3 className="font-semibold text-gray-900 truncate">{report.disease_name}</h3>
          <p className="text-sm text-gray-600">
            on <span className="font-medium">{report.crop_name}</span>
          </p>
        </div>
        <span className={`shrink-0 text-xs font-semibold px-2 py-0.5 rounded-full border ${sev}`}>
          {report.severity}
        </span>
      </div>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-gray-500">
        <span className="inline-flex items-center gap-1">
          <MapPin size={12} />
          {report.location_name}
          {report.division ? `, ${report.division}` : ""}
        </span>
        <span className="inline-flex items-center gap-1">
          <Calendar size={12} />
          {timeAgo(report.created_at)}
        </span>
      </div>

      {report.notes && (
        <p className="mt-2 text-sm text-gray-700 bg-gray-50 rounded-lg px-3 py-2 border border-gray-100">
          {report.notes}
        </p>
      )}
    </article>
  );
}

function StatsSidebar({
  stats,
  loading,
}: {
  stats: CommunityStats | null;
  loading: boolean;
}) {
  if (loading || !stats) return <StatsSkeleton />;

  return (
    <>
      {/* Total */}
      <div className="bg-gradient-to-br from-primary-600 to-primary-700 text-white rounded-2xl p-6 shadow-sm">
        <p className="text-primary-100 text-xs uppercase tracking-wider font-semibold">
          Total reports
        </p>
        <p className="text-4xl font-extrabold mt-1">{stats.total_reports}</p>
        <p className="text-primary-100 text-sm mt-1">
          shared by farmers across Bangladesh
        </p>
      </div>

      {/* Top diseases */}
      <StatList
        title="Top diseases"
        icon={<Leaf size={16} className="text-primary-600" />}
        rows={stats.top_diseases.map((d) => ({
          label: d.disease_name,
          count: d.count,
        }))}
        emptyText="No diseases reported yet."
      />

      {/* Top locations */}
      <StatList
        title="Hotspot locations"
        icon={<MapPin size={16} className="text-red-500" />}
        rows={stats.top_locations.map((l) => ({
          label: l.location_name,
          count: l.count,
        }))}
        emptyText="No locations yet."
      />

      {/* Per-division */}
      <StatList
        title="Reports by division"
        icon={<Sprout size={16} className="text-amber-600" />}
        rows={stats.reports_by_division.map((d) => ({
          label: d.division,
          count: d.count,
        }))}
        emptyText="No divisions yet."
      />
    </>
  );
}

function StatList({
  title,
  icon,
  rows,
  emptyText,
}: {
  title: string;
  icon: React.ReactNode;
  rows: { label: string; count: number }[];
  emptyText: string;
}) {
  const max = rows.reduce((m, r) => Math.max(m, r.count), 0) || 1;
  return (
    <div className="bg-white rounded-2xl border border-gray-200 p-5">
      <h3 className="text-sm font-semibold text-gray-900 flex items-center gap-2 mb-3">
        {icon}
        {title}
      </h3>
      {rows.length === 0 ? (
        <p className="text-sm text-gray-500">{emptyText}</p>
      ) : (
        <ul className="space-y-2">
          {rows.map(({ label, count }) => (
            <li key={label}>
              <div className="flex items-center justify-between text-sm mb-1">
                <span className="text-gray-700 truncate">{label}</span>
                <span className="text-gray-500 font-medium">{count}</span>
              </div>
              <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary-500 rounded-full"
                  style={{ width: `${(count / max) * 100}%` }}
                />
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Skeletons
// ---------------------------------------------------------------------------

function ReportsSkeleton() {
  return (
    <div className="space-y-3">
      {[0, 1, 2].map((i) => (
        <div key={i} className="bg-white border border-gray-200 rounded-xl p-4 animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-1/2 mb-2" />
          <div className="h-3 bg-gray-200 rounded w-1/3 mb-3" />
          <div className="h-3 bg-gray-100 rounded w-2/3" />
        </div>
      ))}
    </div>
  );
}

function StatsSkeleton() {
  return (
    <>
      <div className="h-28 bg-gradient-to-br from-primary-300 to-primary-400 rounded-2xl animate-pulse" />
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className="bg-white border border-gray-200 rounded-2xl p-5 animate-pulse space-y-3"
        >
          <div className="h-4 bg-gray-200 rounded w-1/3" />
          <div className="h-3 bg-gray-100 rounded w-full" />
          <div className="h-3 bg-gray-100 rounded w-5/6" />
          <div className="h-3 bg-gray-100 rounded w-4/6" />
        </div>
      ))}
    </>
  );
}
