import type { LucideIcon } from "lucide-react";

interface Props {
  title: string;
  content: string;
  icon?: LucideIcon;
  variant?: "default" | "warning" | "success";
}

const VARIANT_STYLES = {
  default: {
    card: "bg-white border-gray-200",
    icon: "bg-gray-100 text-gray-600",
    title: "text-gray-900",
  },
  warning: {
    card: "bg-amber-50 border-amber-200",
    icon: "bg-amber-100 text-amber-700",
    title: "text-amber-900",
  },
  success: {
    card: "bg-primary-50 border-primary-200",
    icon: "bg-primary-100 text-primary-700",
    title: "text-primary-900",
  },
} as const;

export default function AdvisoryCard({
  title,
  content,
  icon: Icon,
  variant = "default",
}: Props) {
  const styles = VARIANT_STYLES[variant];

  return (
    <div className={`rounded-2xl border ${styles.card} p-5 shadow-sm`}>
      <div className="flex items-start gap-3 mb-3">
        {Icon && (
          <div
            className={`shrink-0 flex items-center justify-center w-8 h-8 rounded-lg ${styles.icon}`}
          >
            <Icon size={16} strokeWidth={2} />
          </div>
        )}
        <h3 className={`font-semibold text-sm pt-1 ${styles.title}`}>{title}</h3>
      </div>
      <p className="text-sm text-gray-600 leading-relaxed">{content}</p>
    </div>
  );
}
