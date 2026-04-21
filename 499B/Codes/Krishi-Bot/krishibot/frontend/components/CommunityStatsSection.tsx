"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowRight, Leaf, MapPin, Users } from "lucide-react";
import { CommunityStats, getCommunityStats } from "@/lib/api";

export default function CommunityStatsSection() {
  const [stats,   setStats]   = useState<CommunityStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [failed,  setFailed]  = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const s = await getCommunityStats();
        if (!cancelled) setStats(s);
      } catch {
        if (!cancelled) setFailed(true);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // Hide the section entirely if the backend call failed or no reports exist.
  if (failed) return null;
  if (!loading && (!stats || stats.total_reports === 0)) return null;

  const topDisease  = stats?.top_diseases?.[0];
  const topLocation = stats?.top_locations?.[0];

  return (
    <section className="py-20 bg-primary-50 border-t border-primary-100">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-10">
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-primary-100 border border-primary-200 text-primary-700 text-xs font-medium mb-3">
            <Users size={12} />
            Community insights
          </span>
          <h2 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-3">
            What Farmers Are Reporting
          </h2>
          <p className="text-gray-500 text-lg max-w-xl mx-auto">
            Anonymous disease reports shared by farmers across Bangladesh.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 mb-8">
          <MetricCard
            loading={loading}
            icon={<Users size={22} className="text-primary-600" />}
            label="Total reports"
            value={stats?.total_reports.toString() ?? "—"}
            hint="shared anonymously"
          />
          <MetricCard
            loading={loading}
            icon={<Leaf size={22} className="text-red-500" />}
            label="Most reported disease"
            value={topDisease?.disease_name ?? "—"}
            hint={topDisease ? `${topDisease.count} report${topDisease.count === 1 ? "" : "s"}` : "no data yet"}
          />
          <MetricCard
            loading={loading}
            icon={<MapPin size={22} className="text-amber-600" />}
            label="Top hotspot"
            value={topLocation?.location_name ?? "—"}
            hint={topLocation ? `${topLocation.count} report${topLocation.count === 1 ? "" : "s"}` : "no data yet"}
          />
        </div>

        <div className="text-center">
          <Link
            href="/community"
            className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-white border border-primary-200 text-primary-700 font-semibold text-sm hover:bg-primary-100 transition-colors shadow-sm"
          >
            <Users size={16} />
            View all community reports
            <ArrowRight size={16} />
          </Link>
        </div>
      </div>
    </section>
  );
}

function MetricCard({
  loading,
  icon,
  label,
  value,
  hint,
}: {
  loading: boolean;
  icon: React.ReactNode;
  label: string;
  value: string;
  hint: string;
}) {
  return (
    <div className="bg-white rounded-2xl border border-primary-100 p-6 flex flex-col gap-2 hover:shadow-md transition-shadow">
      <div className="flex items-center gap-2">
        <div className="inline-flex items-center justify-center w-10 h-10 rounded-lg bg-primary-50">
          {icon}
        </div>
        <span className="text-sm font-medium text-gray-500">{label}</span>
      </div>
      {loading ? (
        <>
          <div className="h-8 w-24 bg-gray-200 rounded animate-pulse" />
          <div className="h-3 w-32 bg-gray-100 rounded animate-pulse" />
        </>
      ) : (
        <>
          <p className="text-2xl sm:text-3xl font-extrabold text-gray-900 truncate">
            {value}
          </p>
          <p className="text-xs text-gray-500">{hint}</p>
        </>
      )}
    </div>
  );
}
