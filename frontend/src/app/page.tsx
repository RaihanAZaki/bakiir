
"use client";

import {
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type ReactNode,
} from "react";

// ==========================================
// TYPES
// ==========================================

type FreshnessClass = "Fresh" | "Half-Fresh" | "Rotten";
type Page = "dashboard" | "classification" | "history";
type Probabilities = Record<FreshnessClass, number>;

type ImageQuality = {
  status: "passed" | "warning";
  width: number;
  height: number;
  blur_score: number;
  brightness: number;
  warnings: string[];
};

type EggDetection = {
  status: string;
  candidate_found?: boolean;
};


type GradCamExplanation = {
  available: boolean;
  image: string | null;
  source_image?: string | null;
  layer: string | null;
  message: string;
};


type PredictionResult = {
  prediction: FreshnessClass;
  confidence: number;
  probabilities: Probabilities;
  model: string;
  inference_method: string;
  tta_angles: number[];
  image_quality: ImageQuality | null;
  egg_detection: EggDetection | null;
  warnings: string[];
  explanation?: GradCamExplanation;
};

type HistoryItem = {
  id: string;
  filename: string;
  timestamp: string;
  prediction: FreshnessClass;
  confidence: number;
  probabilities: Probabilities;
};

type IconName =
  | "dashboard"
  | "scan"
  | "history"
  | "egg"
  | "upload"
  | "arrow"
  | "trash"
  | "check"
  | "clock"
  | "chart"
  | "info"
  | "image"
  | "close"
  | "sparkles";

// ==========================================
// CONFIG
// ==========================================

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ??
  "http://127.0.0.1:8000";

const STORAGE_KEY = "eggvision-history-v1";
const MAX_FILE_SIZE = 10 * 1024 * 1024;
const MAX_HISTORY = 100;

const CLASSES: FreshnessClass[] = [
  "Fresh",
  "Half-Fresh",
  "Rotten",
];

const LABELS: Record<FreshnessClass, string> = {
  Fresh: "Segar",
  "Half-Fresh": "Setengah Segar",
  Rotten: "Busuk",
};

const BADGE_COLORS: Record<FreshnessClass, string> = {
  Fresh: "bg-emerald-50 text-emerald-700 ring-emerald-200",
  "Half-Fresh": "bg-amber-50 text-amber-700 ring-amber-200",
  Rotten: "bg-rose-50 text-rose-700 ring-rose-200",
};

const BAR_COLORS: Record<FreshnessClass, string> = {
  Fresh: "bg-emerald-500",
  "Half-Fresh": "bg-amber-500",
  Rotten: "bg-rose-500",
};

const DOT_COLORS: Record<FreshnessClass, string> = {
  Fresh: "bg-emerald-500",
  "Half-Fresh": "bg-amber-500",
  Rotten: "bg-rose-500",
};

// ==========================================
// HELPERS
// ==========================================

function formatPercent(value: number) {
  return `${(value * 100).toFixed(2)}%`;
}

function formatDate(value: string) {
  return new Date(value).toLocaleString("id-ID", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function isClass(value: unknown): value is FreshnessClass {
  return (
    value === "Fresh" ||
    value === "Half-Fresh" ||
    value === "Rotten"
  );
}

function isProbabilities(
  value: unknown
): value is Probabilities {
  if (!value || typeof value !== "object") return false;

  const data = value as Partial<Probabilities>;

  return CLASSES.every((label) => {
    const score = data[label];

    return (
      typeof score === "number" &&
      Number.isFinite(score) &&
      score >= 0 &&
      score <= 1
    );
  });
}

function isPrediction(
  value: unknown
): value is PredictionResult {
  if (!value || typeof value !== "object") return false;

  const data = value as Partial<PredictionResult>;

  return (
    isClass(data.prediction) &&
    typeof data.confidence === "number" &&
    Number.isFinite(data.confidence) &&
    data.confidence >= 0 &&
    data.confidence <= 1 &&
    isProbabilities(data.probabilities)
  );
}

function isHistoryItem(value: unknown): value is HistoryItem {
  if (!value || typeof value !== "object") return false;

  const data = value as Partial<HistoryItem>;

  return (
    typeof data.id === "string" &&
    typeof data.filename === "string" &&
    typeof data.timestamp === "string" &&
    !Number.isNaN(Date.parse(data.timestamp)) &&
    isClass(data.prediction) &&
    typeof data.confidence === "number" &&
    Number.isFinite(data.confidence) &&
    data.confidence >= 0 &&
    data.confidence <= 1 &&
    isProbabilities(data.probabilities)
  );
}

// ==========================================
// ICON COMPONENT
// ==========================================

function Icon({
  name,
  className = "h-5 w-5",
}: {
  name: IconName;
  className?: string;
}) {
  const icons: Record<IconName, ReactNode> = {
    dashboard: (
      <>
        <rect x="3" y="3" width="7" height="7" rx="1.5" />
        <rect x="14" y="3" width="7" height="7" rx="1.5" />
        <rect x="3" y="14" width="7" height="7" rx="1.5" />
        <rect x="14" y="14" width="7" height="7" rx="1.5" />
      </>
    ),
    scan: (
      <>
        <path d="M4 9V5a1 1 0 0 1 1-1h4" />
        <path d="M15 4h4a1 1 0 0 1 1 1v4" />
        <path d="M20 15v4a1 1 0 0 1-1 1h-4" />
        <path d="M9 20H5a1 1 0 0 1-1-1v-4" />
        <path d="M8 12h8" />
      </>
    ),
    history: (
      <>
        <path d="M3 12a9 9 0 1 0 3-6.7" />
        <path d="M3 3v5h5" />
        <path d="M12 7v5l3 2" />
      </>
    ),
    egg: (
      <path d="M12 22c-5 0-8-3.7-8-8.3C4 8.4 7.6 2 12 2s8 6.4 8 11.7C20 18.3 17 22 12 22Z" />
    ),
    upload: (
      <>
        <path d="M12 16V4" />
        <path d="m7 9 5-5 5 5" />
        <path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" />
      </>
    ),
    arrow: (
      <>
        <path d="M5 12h14" />
        <path d="m13 6 6 6-6 6" />
      </>
    ),
    trash: (
      <>
        <path d="M3 6h18" />
        <path d="M8 6V4h8v2" />
        <path d="M6 6l1 15h10l1-15" />
        <path d="M10 10v7" />
        <path d="M14 10v7" />
      </>
    ),
    check: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="m8 12 3 3 5-6" />
      </>
    ),
    clock: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 7v5l3 2" />
      </>
    ),
    chart: (
      <>
        <path d="M4 20V10" />
        <path d="M10 20V4" />
        <path d="M16 20v-8" />
        <path d="M22 20v-5" />
      </>
    ),
    info: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 11v5" />
        <path d="M12 8h.01" />
      </>
    ),
    image: (
      <>
        <rect x="3" y="3" width="18" height="18" rx="2" />
        <circle cx="8.5" cy="8.5" r="1.5" />
        <path d="m21 15-5-5L5 21" />
      </>
    ),
    close: (
      <>
        <path d="M5 5l14 14" />
        <path d="M19 5 5 19" />
      </>
    ),
    sparkles: (
      <>
        <path d="m12 3 2 7 7 2-7 2-2 7-2-7-7-2 7-2 2-7Z" />
      </>
    ),
  };

  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      {icons[name]}
    </svg>
  );
}

// ==========================================
// SHARED COMPONENTS
// ==========================================

function ResultBadge({
  value,
}: {
  value: FreshnessClass;
}) {
  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-semibold ring-1 ring-inset ${BADGE_COLORS[value]}`}
    >
      <span
        className={`h-1.5 w-1.5 rounded-full ${DOT_COLORS[value]}`}
      />
      {LABELS[value]}
    </span>
  );
}

function Sidebar({
  page,
  onNavigate,
}: {
  page: Page;
  onNavigate: (value: Page) => void;
}) {
  const menus: {
    id: Page;
    title: string;
    description: string;
    icon: IconName;
  }[] = [
    {
      id: "dashboard",
      title: "Dashboard",
      description: "Overview",
      icon: "dashboard",
    },
    {
      id: "classification",
      title: "Pengecekan Klasifikasi",
      description: "Analyze egg",
      icon: "scan",
    },
    {
      id: "history",
      title: "Histori",
      description: "Activity logs",
      icon: "history",
    },
  ];

  return (
    <aside className="flex w-full shrink-0 flex-col border-b border-slate-200 bg-white md:sticky md:top-0 md:h-screen md:w-72 md:border-b-0 md:border-r">
      {/* BRAND */}
      <div className="flex items-center gap-3 px-5 py-6 md:px-7 md:py-8">
        <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-emerald-600 text-white shadow-lg shadow-emerald-100">
          <Icon name="egg" className="h-7 w-7" />
        </div>

        <div>
          <h1 className="text-xl font-extrabold tracking-tight text-slate-900">
            Egg<span className="text-emerald-600">Vision</span>
          </h1>
          <p className="mt-0.5 text-[10px] font-semibold tracking-wide text-slate-400">
            EGG FRESHNESS ANALYTICS
          </p>
        </div>
      </div>

      <div className="hidden px-7 pb-3 pt-5 text-[11px] font-bold uppercase tracking-[0.2em] text-slate-400 md:block">
        Main Menu
      </div>

      {/* NAVIGATION */}
      <nav
        aria-label="Navigasi utama"
        className="flex gap-2 overflow-x-auto px-3 pb-4 md:flex-col md:gap-2 md:px-4"
      >
        {menus.map((menu) => {
          const active = page === menu.id;

          return (
            <button
              key={menu.id}
              type="button"
              onClick={() => onNavigate(menu.id)}
              aria-current={active ? "page" : undefined}
              className={`group flex shrink-0 items-center gap-3 rounded-xl px-4 py-3.5 text-left transition-all duration-200 ${
                active
                  ? "bg-emerald-50 text-emerald-700"
                  : "text-slate-500 hover:bg-slate-50 hover:text-slate-900"
              }`}
            >
              <div
                className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg transition ${
                  active
                    ? "bg-emerald-600 text-white"
                    : "bg-slate-100 text-slate-500 group-hover:bg-slate-200"
                }`}
              >
                <Icon
                  name={menu.icon}
                  className="h-5 w-5"
                />
              </div>

              <div className="min-w-0">
                <p className="whitespace-nowrap text-sm font-semibold">
                  {menu.title}
                </p>

                <p
                  className={`mt-0.5 hidden text-xs md:block ${
                    active
                      ? "text-emerald-600/70"
                      : "text-slate-400"
                  }`}
                >
                  {menu.description}
                </p>
              </div>

              {active && (
                <span className="ml-auto hidden h-2 w-2 rounded-full bg-emerald-500 md:block" />
              )}
            </button>
          );
        })}
      </nav>

      {/* SIDEBAR FOOTER */}
      <div className="mt-auto hidden p-5 md:block">
    

        <p className="mt-5 text-center text-[11px] text-slate-400">
          EggVision Research Prototype
        </p>
      </div>
    </aside>
  );
}

function StatCard({
  title,
  value,
  subtitle,
  accent = "text-slate-900",
}: {
  title: string;
  value: number;
  subtitle: string;
  accent?: string;
}) {
  const isFresh = title === "Telur Segar";
  const isHalf = title === "Setengah Segar";
  const isRotten = title === "Telur Busuk";

  const iconBackground = isFresh
    ? "bg-emerald-50 text-emerald-600"
    : isHalf
      ? "bg-amber-50 text-amber-600"
      : isRotten
        ? "bg-rose-50 text-rose-600"
        : "bg-sky-50 text-sky-600";

  const indicator = isFresh
    ? "bg-emerald-500"
    : isHalf
      ? "bg-amber-500"
      : isRotten
        ? "bg-rose-500"
        : "bg-sky-500";

  return (
    <div className="group relative overflow-hidden rounded-2xl border border-slate-200/80 bg-white p-5 shadow-sm transition-all duration-300 hover:-translate-y-1 hover:border-emerald-200 hover:shadow-xl hover:shadow-slate-200/60 sm:p-6">
      <div className="flex items-start justify-between gap-3">
        <p className="text-sm font-medium text-slate-500">
          {title}
        </p>

        <div
          className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-xl ${iconBackground}`}
        >
          {isFresh ? (
            <span className="text-xl">✓</span>
          ) : isHalf ? (
            <span className="text-xl">◐</span>
          ) : isRotten ? (
            <span className="text-xl">!</span>
          ) : (
            <Icon name="scan" className="h-5 w-5" />
          )}
        </div>
      </div>

      <p
        className={`mt-5 text-4xl font-extrabold tracking-tight ${accent}`}
      >
        {value.toLocaleString("id-ID")}
      </p>

      <p className="mt-2 text-xs text-slate-400">
        {subtitle}
      </p>

      <div className="mt-6 h-1 overflow-hidden rounded-full bg-slate-100">
        <div className={`h-full w-2/5 rounded-full ${indicator}`} />
      </div>
    </div>
  );
}

function ProbabilityBars({
  probabilities,
}: {
  probabilities: Probabilities;
}) {
  return (
    <div className="space-y-5">
      {CLASSES.map((label) => {
        const score = probabilities[label] * 100;

        return (
          <div key={label}>
            <div className="mb-2 flex justify-between gap-3 text-sm">
              <span className="font-medium text-slate-700">
                {LABELS[label]}
              </span>

              <span className="font-semibold text-slate-700">
                {score.toFixed(2)}%
              </span>
            </div>

            <div className="h-3 overflow-hidden rounded-full bg-slate-100">
              <div
                className={`h-full rounded-full transition-all duration-500 ${BAR_COLORS[label]}`}
                style={{
                  width: `${Math.max(
                    0,
                    Math.min(100, score)
                  )}%`,
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function ExplanationCard({
  explanation,
  prediction,
  originalImage,
}: {
  explanation?: GradCamExplanation;
  prediction: FreshnessClass;
  originalImage: string | null;
}) {
  const descriptions: Record<FreshnessClass, string> = {
    Fresh: "Model memprediksi kategori Segar berdasarkan pola visual pada data pelatihan. Ini bukan bukti bahwa isi telur benar-benar segar.",
    "Half-Fresh": "Model memprediksi kategori Setengah Segar berdasarkan pola visual pada data pelatihan. Tingkat mutu internal telur tidak diukur secara langsung.",
    Rotten: "Model memprediksi kategori Busuk berdasarkan pola visual pada data pelatihan. Prediksi ini belum memastikan telur mengalami pembusukan internal.",
  };

  return (
    <section className="mt-7 overflow-hidden rounded-2xl border border-slate-200 bg-white">
      <div className="flex items-center gap-3 border-b border-slate-100 px-5 py-4">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-violet-50 text-violet-600">
          <Icon name="sparkles" className="h-5 w-5" />
        </div>
        <div>
          <h4 className="text-sm font-bold text-slate-900">Grad-CAM</h4>
          <p className="mt-1 text-xs text-slate-500">Visualisasi area yang berkontribusi pada prediksi CNN</p>
        </div>
      </div>
      <div className="p-5">
        {explanation?.available && explanation.image ? (
          <>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div>
                <p className="mb-2 text-xs font-semibold text-slate-600">Input Model (224 × 224)</p>
                <div className="overflow-hidden rounded-xl bg-slate-100">
                  {(explanation.source_image || originalImage) && (
                    <img src={explanation.source_image || originalImage || ""} alt="Gambar input klasifikasi" className="aspect-square w-full object-contain" />
                  )}
                </div>
              </div>
              <div>
                <p className="mb-2 text-xs font-semibold text-slate-600">Grad-CAM + TTA</p>
                <div className="overflow-hidden rounded-xl bg-slate-100">
                  <img src={explanation.image} alt="Heatmap Grad-CAM pada input model" className="aspect-square w-full object-contain" />
                </div>
              </div>
            </div>
            <div className="mt-4 flex items-center gap-2 text-xs text-slate-500">
              <span className="h-2.5 w-2.5 rounded-full bg-blue-500" />Kontribusi rendah
              <span className="ml-auto h-2.5 w-2.5 rounded-full bg-red-500" />Kontribusi tinggi
            </div>
            <p className="mt-4 text-xs leading-relaxed text-slate-600">
              Warna merah dan kuning menandai lokasi dengan aktivasi Grad-CAM relatif tinggi untuk kelas
              {" "}<strong>{LABELS[prediction]}</strong>. Ini menunjukkan perilaku model, bukan bukti penyebab biologis kesegaran.
            </p>
            {explanation.layer && (
              <p className="mt-3 break-all text-[11px] text-slate-400">Layer: {explanation.layer}</p>
            )}
          </>
        ) : (
          <div className="rounded-xl bg-amber-50 p-4 text-xs leading-relaxed text-amber-800">
            {explanation?.message || "Visualisasi Grad-CAM belum tersedia. Pastikan backend sudah memakai versi Grad-CAM."}
          </div>
        )}
        <div className="mt-5 border-t border-slate-100 pt-5">
          <h5 className="text-sm font-bold text-slate-800">Interpretasi hasil</h5>
          <p className="mt-2 text-sm leading-relaxed text-slate-600">{descriptions[prediction]}</p>
        </div>
      </div>
    </section>
  );
}

function HistoryTable({
  data,
  onClear,
}: {
  data: HistoryItem[];
  onClear?: () => void;
}) {
  return (
    <div className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 px-6 py-5">
        <div>
          <h3 className="font-bold text-slate-900">
            Riwayat Pengecekan
          </h3>

          <p className="mt-1 text-xs text-slate-500">
            {data.length} hasil ditampilkan
          </p>
        </div>

        {onClear && data.length > 0 && (
          <button
            type="button"
            onClick={onClear}
            className="inline-flex items-center gap-2 rounded-xl px-3 py-2 text-xs font-semibold text-rose-600 transition hover:bg-rose-50"
          >
            <Icon name="trash" className="h-4 w-4" />
            Hapus Histori
          </button>
        )}
      </div>

      {data.length === 0 ? (
        <div className="px-5 py-16 text-center">
          <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-slate-100 text-slate-400">
            <Icon name="history" className="h-7 w-7" />
          </div>

          <p className="mt-4 font-semibold text-slate-700">
            Belum Ada Histori
          </p>

          <p className="mx-auto mt-2 max-w-sm text-sm leading-relaxed text-slate-400">
            Hasil klasifikasi telur yang berhasil diproses
            akan otomatis ditampilkan di sini.
          </p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[680px] text-left text-sm">
            <thead className="bg-slate-50/80 text-xs uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-6 py-4 font-semibold">No.</th>
                <th className="px-6 py-4 font-semibold">Nama File</th>
                <th className="px-6 py-4 font-semibold">Tanggal</th>
                <th className="px-6 py-4 font-semibold">Hasil</th>
                <th className="px-6 py-4 font-semibold">Confidence</th>
              </tr>
            </thead>

            <tbody className="divide-y divide-slate-100">
              {data.map((item, index) => (
                <tr
                  key={item.id}
                  className="transition hover:bg-slate-50"
                >
                  <td className="px-6 py-4 text-slate-500">
                    {index + 1}
                  </td>

                  <td className="max-w-[220px] truncate px-6 py-4 font-semibold text-slate-800">
                    {item.filename}
                  </td>

                  <td className="whitespace-nowrap px-6 py-4 text-slate-500">
                    {formatDate(item.timestamp)}
                  </td>

                  <td className="px-6 py-4">
                    <ResultBadge value={item.prediction} />
                  </td>

                  <td className="px-6 py-4 font-bold text-slate-700">
                    {formatPercent(item.confidence)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ==========================================
// MAIN PAGE
// ==========================================

export default function Home() {
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Navigation
  const [page, setPage] = useState<Page>("dashboard");

  // Classification
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);

  // History
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [historyReady, setHistoryReady] = useState(false);
  const [storageWarning, setStorageWarning] = useState("");

  // ==========================================
  // LOAD HISTORY
  // ==========================================

  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);

      if (saved) {
        const parsed: unknown = JSON.parse(saved);

        if (Array.isArray(parsed)) {
          setHistory(
            parsed
              .filter(isHistoryItem)
              .slice(0, MAX_HISTORY)
          );
        }
      }
    } catch {
      setStorageWarning(
        "Histori yang tersimpan tidak dapat dibaca."
      );
    } finally {
      setHistoryReady(true);
    }
  }, []);

  // ==========================================
  // SAVE HISTORY
  // ==========================================

  useEffect(() => {
    if (!historyReady) return;

    try {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify(history)
      );

      setStorageWarning("");
    } catch {
      setStorageWarning(
        "Penyimpanan browser tidak tersedia. Histori hanya tersimpan sementara."
      );
    }
  }, [history, historyReady]);

  // ==========================================
  // CLEANUP PREVIEW
  // ==========================================

  useEffect(() => {
    return () => {
      if (preview) {
        URL.revokeObjectURL(preview);
      }
    };
  }, [preview]);

  // ==========================================
  // FILE HANDLING
  // ==========================================

  function handleFile(selected?: File) {
    if (!selected) return;

    const allowed = [
      "image/jpeg",
      "image/png",
      "image/webp",
    ];

    if (!allowed.includes(selected.type)) {
      setError("Format file harus JPG, PNG, atau WebP.");
      return;
    }

    if (selected.size > MAX_FILE_SIZE) {
      setError("Ukuran maksimal foto adalah 10 MB.");
      return;
    }

    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
    setError("");
  }

  function onFileChange(
    event: ChangeEvent<HTMLInputElement>
  ) {
    handleFile(event.target.files?.[0]);
    event.target.value = "";
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    handleFile(event.dataTransfer.files?.[0]);
  }

  function removeFile() {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError("");

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  // ==========================================
  // CLASSIFICATION
  // ==========================================

  async function classify() {
    if (!file || loading) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(
        `${API_URL}/predict`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data: unknown = await response.json();

      if (!response.ok) {
        const detail =
          data &&
          typeof data === "object" &&
          "detail" in data
            ? (data as { detail?: unknown }).detail
            : null;

        throw new Error(
          typeof detail === "string"
            ? detail
            : "Klasifikasi gagal."
        );
      }

      if (!isPrediction(data)) {
        throw new Error(
          "Response backend tidak sesuai dengan format Model G."
        );
      }

      setResult(data);

      const newItem: HistoryItem = {
        id: crypto.randomUUID(),
        filename: file.name,
        timestamp: new Date().toISOString(),
        prediction: data.prediction,
        confidence: data.confidence,
        probabilities: data.probabilities,
      };

      setHistory((previous) =>
        [newItem, ...previous].slice(0, MAX_HISTORY)
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Tidak dapat terhubung ke backend."
      );
    } finally {
      setLoading(false);
    }
  }

  // ==========================================
  // CLEAR HISTORY
  // ==========================================

  function clearHistory() {
    const confirmed = window.confirm(
      "Yakin ingin menghapus seluruh histori klasifikasi?"
    );

    if (confirmed) {
      setHistory([]);
    }
  }

  // ==========================================
  // DASHBOARD STATS
  // ==========================================

  const total = history.length;

  const countByClass = (label: FreshnessClass) =>
    history.filter(
      (item) => item.prediction === label
    ).length;

  // ==========================================
  // RENDER
  // ==========================================

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 md:flex">
      <Sidebar
        page={page}
        onNavigate={setPage}
      />

      <main className="min-w-0 flex-1">
        <div className="mx-auto max-w-7xl px-4 py-7 sm:px-7 md:py-10 lg:px-10">

          {/* ======================================
              PAGE HEADER
          ====================================== */}

          <header className="mb-8 flex flex-wrap items-center justify-between gap-4">
            <div>
              <p className="mb-2 text-xs font-bold uppercase tracking-[0.2em] text-emerald-600">
                EggVision /{" "}
                {page === "dashboard"
                  ? "Overview"
                  : page === "classification"
                    ? "Classification"
                    : "History"}
              </p>

              <h2 className="text-2xl font-extrabold tracking-tight text-slate-900 sm:text-3xl">
                {page === "dashboard"
                  ? "Dashboard"
                  : page === "classification"
                    ? "Pengecekan Klasifikasi"
                    : "Histori Klasifikasi"}
              </h2>

              <p className="mt-2 text-sm text-slate-500">
                {page === "dashboard"
                  ? "Pantau aktivitas dan hasil analisis kesegaran telur."
                  : page === "classification"
                    ? "Unggah foto telur untuk diklasifikasikan menggunakan Model G."
                    : "Lihat semua riwayat pengecekan kesegaran telur."}
              </p>
            </div>

            {page === "dashboard" && (
              <button
                type="button"
                onClick={() => setPage("classification")}
                className="inline-flex items-center gap-2 rounded-xl bg-emerald-600 px-5 py-3 text-sm font-bold text-white shadow-sm transition hover:bg-emerald-700"
              >
                <Icon name="scan" className="h-5 w-5" />
                Klasifikasi Baru
              </button>
            )}
          </header>

          {/* STORAGE WARNING */}
          {storageWarning && (
            <div className="mb-6 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
              {storageWarning}
            </div>
          )}

          {/* ======================================
              DASHBOARD PAGE
          ====================================== */}

          {page === "dashboard" && (
            <div className="space-y-7">

              {/* WELCOME BANNER */}
              <section className="relative overflow-hidden rounded-3xl bg-slate-900 p-6 text-white shadow-lg shadow-slate-200 sm:p-8 lg:p-10">
                <div className="pointer-events-none absolute -right-16 -top-24 h-64 w-64 rounded-full bg-emerald-500/20 blur-3xl" />
                <div className="pointer-events-none absolute -bottom-32 right-24 h-64 w-64 rounded-full bg-teal-400/10 blur-3xl" />

                <div className="relative flex flex-col gap-7 lg:flex-row lg:items-center lg:justify-between">
                  <div className="max-w-xl">
                    <div className="inline-flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1.5 text-[11px] font-semibold tracking-wide text-emerald-300">
                      <span className="h-2 w-2 rounded-full bg-emerald-400" />
                      EGG FRESHNESS INTELLIGENCE
                    </div>

                    <h3 className="mt-5 text-2xl font-extrabold leading-tight tracking-tight sm:text-3xl">
                      Selamat Datang di
                      <span className="block text-emerald-400">
                        EggVision Dashboard
                      </span>
                    </h3>

                    <p className="mt-4 max-w-lg text-sm leading-relaxed text-slate-300">
                      Pantau hasil klasifikasi telur ayam
                      melalui teknologi MobileNetV2 dan
                      Test-Time Augmentation.
                    </p>

                    <button
                      type="button"
                      onClick={() => setPage("classification")}
                      className="mt-6 inline-flex items-center gap-3 rounded-xl bg-emerald-500 px-5 py-3 text-sm font-bold text-white transition hover:bg-emerald-400"
                    >
                      <Icon name="scan" className="h-5 w-5" />
                      Mulai Klasifikasi
                      <Icon name="arrow" className="h-4 w-4" />
                    </button>
                  </div>

                  <div className="flex shrink-0 items-center gap-4 rounded-2xl border border-white/10 bg-white/5 p-5 backdrop-blur-sm lg:w-60">
                    <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-emerald-400/15 text-emerald-300">
                      <Icon name="egg" className="h-7 w-7" />
                    </div>

                    <div>
                      <p className="text-xs text-slate-400">
                        Total Analisis
                      </p>

                      <p className="mt-1 text-3xl font-extrabold">
                        {total}
                      </p>

                      <p className="mt-1 text-[11px] text-emerald-300">
                        Hasil tersimpan
                      </p>
                    </div>
                  </div>
                </div>
              </section>

              {/* SECTION HEADING */}
              <div className="flex flex-wrap items-end justify-between gap-3">
                <div>
                  <h3 className="text-lg font-bold tracking-tight text-slate-900">
                    Ringkasan Klasifikasi
                  </h3>

                  <p className="mt-1 text-sm text-slate-500">
                    Statistik berdasarkan histori pengecekan.
                  </p>
                </div>

                <span className="rounded-full border border-slate-200 bg-white px-4 py-2 text-xs font-medium text-slate-500">
                  Model G · 3 Kategori
                </span>
              </div>

              {/* STATS */}
              <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                <StatCard
                  title="Total Pengecekan"
                  value={total}
                  subtitle="Seluruh hasil klasifikasi"
                />

                <StatCard
                  title="Telur Segar"
                  value={countByClass("Fresh")}
                  subtitle="Prediksi Fresh"
                  accent="text-emerald-600"
                />

                <StatCard
                  title="Setengah Segar"
                  value={countByClass("Half-Fresh")}
                  subtitle="Prediksi Half-Fresh"
                  accent="text-amber-600"
                />

                <StatCard
                  title="Telur Busuk"
                  value={countByClass("Rotten")}
                  subtitle="Prediksi Rotten"
                  accent="text-rose-600"
                />
              </div>

              {/* ANALYTICS GRID */}
              <div className="grid gap-6 xl:grid-cols-5">

                {/* DISTRIBUTION */}
                <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm sm:p-7 xl:col-span-3">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <h3 className="text-lg font-bold text-slate-900">
                        Distribusi Klasifikasi
                      </h3>

                      <p className="mt-1 text-sm text-slate-500">
                        Perbandingan hasil prediksi setiap kategori.
                      </p>
                    </div>

                    <div className="rounded-xl bg-slate-50 p-3 text-slate-500">
                      <Icon name="chart" className="h-5 w-5" />
                    </div>
                  </div>

                  <div className="mt-8 space-y-7">
                    {CLASSES.map((label) => {
                      const count = countByClass(label);

                      const percentage =
                        total > 0
                          ? (count / total) * 100
                          : 0;

                      return (
                        <div key={label}>
                          <div className="mb-3 flex items-center justify-between gap-3">
                            <div className="flex items-center gap-3">
                              <span
                                className={`h-3 w-3 rounded-full ${DOT_COLORS[label]}`}
                              />

                              <span className="text-sm font-semibold text-slate-700">
                                {LABELS[label]}
                              </span>
                            </div>

                            <div className="flex items-center gap-3">
                              <span className="text-xs text-slate-400">
                                {count} hasil
                              </span>

                              <span className="min-w-14 text-right text-sm font-bold text-slate-800">
                                {percentage.toFixed(1)}%
                              </span>
                            </div>
                          </div>

                          <div className="h-3 overflow-hidden rounded-full bg-slate-100">
                            <div
                              className={`h-full rounded-full transition-all duration-500 ${BAR_COLORS[label]}`}
                              style={{
                                width: `${percentage}%`,
                              }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  <div className="mt-9 rounded-2xl bg-slate-50 p-4">
                    <div className="flex items-start gap-3">
                      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-white text-emerald-600 shadow-sm">
                        <Icon name="info" className="h-5 w-5" />
                      </span>

                      <div>
                        <p className="text-sm font-semibold text-slate-700">
                          Informasi Statistik
                        </p>

                        <p className="mt-1 text-xs leading-relaxed text-slate-500">
                          Grafik menunjukkan distribusi prediksi
                          yang tersimpan, bukan persentase akurasi
                          atau kondisi telur sebenarnya.
                        </p>
                      </div>
                    </div>
                  </div>
                </section>

                {/* MODEL INFO */}
                <section className="flex flex-col rounded-3xl border border-slate-200 bg-white p-6 shadow-sm sm:p-7 xl:col-span-2">
                  <div>
                    <h3 className="text-lg font-bold text-slate-900">
                      Model Information
                    </h3>

                    <p className="mt-1 text-sm text-slate-500">
                      Konfigurasi klasifikasi EggVision.
                    </p>
                  </div>

                  <div className="mt-7 flex items-center gap-4 rounded-2xl bg-emerald-50 p-4">
                    <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-emerald-600 text-white">
                      <Icon name="scan" className="h-6 w-6" />
                    </div>

                    <div>
                      <p className="font-bold text-emerald-900">
                        MobileNetV2
                      </p>

                      <p className="mt-1 text-xs text-emerald-700">
                        Rotation Robust · Model G
                      </p>
                    </div>
                  </div>

                  <div className="mt-6 divide-y divide-slate-100">
                    {[
                      ["Architecture", "CNN · MobileNetV2"],
                      ["Model", "rotation_robust.keras"],
                      ["Input Image", "224 × 224 RGB"],
                      ["Classification", "3 Categories"],
                      ["Inference", "TTA · 4 Rotations"],
                    ].map(([label, value]) => (
                      <div
                        key={label}
                        className="flex items-center justify-between gap-3 py-3"
                      >
                        <span className="text-xs text-slate-500">
                          {label}
                        </span>

                        <span className="break-all text-right text-xs font-semibold text-slate-700">
                          {value}
                        </span>
                      </div>
                    ))}
                  </div>

                  <button
                    type="button"
                    onClick={() => setPage("classification")}
                    className="mt-auto flex w-full items-center justify-between rounded-xl bg-slate-900 px-5 py-3.5 text-sm font-semibold text-white transition hover:bg-slate-800"
                  >
                    Coba Klasifikasi
                    <Icon name="arrow" className="h-4 w-4" />
                  </button>
                </section>
              </div>

              {/* RECENT ACTIVITY */}
              <section className="space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <h3 className="text-lg font-bold text-slate-900">
                      Aktivitas Terbaru
                    </h3>

                    <p className="mt-1 text-sm text-slate-500">
                      Lima hasil pengecekan terakhir.
                    </p>
                  </div>

                  <button
                    type="button"
                    onClick={() => setPage("history")}
                    className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-xs font-semibold text-slate-700 transition hover:bg-slate-50"
                  >
                    Lihat Semua Histori
                    <Icon name="arrow" className="h-4 w-4" />
                  </button>
                </div>

                <HistoryTable
                  data={history.slice(0, 5)}
                />
              </section>
            </div>
          )}

          {/* ======================================
              CLASSIFICATION PAGE
          ====================================== */}

          {page === "classification" && (
            <div className="grid gap-6 xl:grid-cols-5">

              {/* LEFT COLUMN */}
              <div className="space-y-6 xl:col-span-3">

                {/* UPLOAD CARD */}
                <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm sm:p-7">
                  <div className="mb-6 flex items-start gap-4">
                    <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-emerald-50 text-emerald-600">
                      <Icon name="upload" className="h-6 w-6" />
                    </div>

                    <div>
                      <h3 className="text-lg font-bold">
                        Upload Foto Telur
                      </h3>

                      <p className="mt-2 text-sm leading-relaxed text-slate-500">
                        Unggah foto telur untuk mendapatkan prediksi
                        kategori Segar, Setengah Segar, atau Busuk.
                      </p>
                    </div>
                  </div>

                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/jpeg,image/png,image/webp"
                    className="hidden"
                    onChange={onFileChange}
                  />

                  {/* DROP ZONE */}
                  <div
                    role="button"
                    tabIndex={0}
                    onClick={() => fileInputRef.current?.click()}
                    onKeyDown={(event) => {
                      if (
                        event.key === "Enter" ||
                        event.key === " "
                      ) {
                        event.preventDefault();
                        fileInputRef.current?.click();
                      }
                    }}
                    onDragOver={(event) => {
                      event.preventDefault();
                      setDragging(true);
                    }}
                    onDragLeave={() => setDragging(false)}
                    onDrop={onDrop}
                    className={`flex min-h-72 cursor-pointer flex-col items-center justify-center overflow-hidden rounded-2xl border-2 border-dashed p-6 text-center transition ${
                      dragging
                        ? "border-emerald-500 bg-emerald-50"
                        : "border-slate-300 bg-slate-50 hover:border-emerald-400 hover:bg-emerald-50/40"
                    }`}
                  >
                    {preview ? (
                      <img
                        src={preview}
                        alt="Preview telur yang diunggah"
                        className="max-h-72 max-w-full rounded-xl object-contain"
                      />
                    ) : (
                      <>
                        <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-white text-emerald-600 shadow-sm">
                          <Icon name="upload" className="h-8 w-8" />
                        </div>

                        <p className="mt-5 font-semibold text-slate-800">
                          Pilih atau Tarik Foto ke Sini
                        </p>

                        <p className="mt-2 text-sm text-slate-500">
                          Klik area ini atau drag & drop foto telur
                        </p>

                        <p className="mt-2 text-xs text-slate-400">
                          JPG, PNG, WebP · Maksimal 10 MB
                        </p>
                      </>
                    )}
                  </div>

                  {/* SELECTED FILE */}
                  {file && (
                    <div className="mt-4 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-100 bg-slate-50 px-4 py-3">
                      <div className="flex min-w-0 items-center gap-3">
                        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-white text-emerald-600">
                          <Icon name="image" className="h-5 w-5" />
                        </div>

                        <div className="min-w-0">
                          <p className="truncate text-sm font-semibold">
                            {file.name}
                          </p>

                          <p className="text-xs text-slate-400">
                            {(file.size / 1024 / 1024).toFixed(2)} MB
                          </p>
                        </div>
                      </div>

                      <button
                        type="button"
                        onClick={removeFile}
                        className="rounded-lg p-2 text-slate-400 transition hover:bg-rose-50 hover:text-rose-600"
                        aria-label="Hapus foto"
                      >
                        <Icon name="close" className="h-4 w-4" />
                      </button>
                    </div>
                  )}

                  {/* CLASSIFY BUTTON */}
                  <button
                    type="button"
                    onClick={classify}
                    disabled={!file || loading}
                    className="mt-6 flex w-full items-center justify-center gap-3 rounded-xl bg-emerald-600 px-5 py-3.5 text-sm font-bold text-white shadow-sm transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {loading ? (
                      <>
                        <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                        Sedang Memproses Klasifikasi...
                      </>
                    ) : (
                      <>
                        <Icon name="scan" className="h-5 w-5" />
                        Mulai Klasifikasi
                      </>
                    )}
                  </button>

                  {error && (
                    <div className="mt-4 rounded-xl border border-rose-100 bg-rose-50 p-4 text-sm text-rose-700">
                      {error}
                    </div>
                  )}
                </section>

                {/* MODEL INFORMATION */}
                <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm sm:p-7">
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-slate-100 text-slate-600">
                      <Icon name="info" className="h-5 w-5" />
                    </div>

                    <h3 className="text-base font-bold">
                      Informasi Model
                    </h3>
                  </div>

                  <div className="mt-6 divide-y divide-slate-100">
                    {[
                      ["Arsitektur", "MobileNetV2"],
                      ["Model", "Rotation Robust (G)"],
                      ["Input", "224 × 224 RGB"],
                      ["Kategori", "Fresh, Half-Fresh, Rotten"],
                      ["Metode", "TTA 4 Rotasi"],
                    ].map(([label, value]) => (
                      <div
                        key={label}
                        className="flex items-start justify-between gap-4 py-3 text-sm"
                      >
                        <span className="shrink-0 text-slate-500">
                          {label}
                        </span>

                        <span className="text-right font-semibold text-slate-700">
                          {value}
                        </span>
                      </div>
                    ))}
                  </div>
                </section>
              </div>

              {/* RIGHT COLUMN */}
              <div className="xl:col-span-2">
                <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm sm:p-7">
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-50 text-emerald-600">
                      <Icon name="chart" className="h-5 w-5" />
                    </div>

                    <h3 className="text-lg font-bold">
                      Hasil Klasifikasi
                    </h3>
                  </div>

                  {!result ? (
                    <div className="py-16 text-center">
                      <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-slate-100 text-slate-400">
                        <Icon name="egg" className="h-8 w-8" />
                      </div>

                      <p className="mt-5 font-semibold text-slate-700">
                        Belum Ada Hasil
                      </p>

                      <p className="mx-auto mt-2 max-w-xs text-sm leading-relaxed text-slate-500">
                        Upload foto dan tekan tombol
                        Mulai Klasifikasi untuk melihat
                        hasil analisis Model G.
                      </p>
                    </div>
                  ) : (
                    <div className="mt-6">
                      {/* RESULT HERO */}
                      <div className="rounded-2xl bg-slate-50 p-6 text-center">
                        <p className="text-xs font-bold uppercase tracking-widest text-slate-500">
                          Hasil Prediksi
                        </p>

                        <div className="mt-5">
                          <ResultBadge value={result.prediction} />
                        </div>

                        <p className="mt-5 text-4xl font-extrabold tracking-tight text-slate-900">
                          {formatPercent(result.confidence)}
                        </p>

                        <p className="mt-2 text-xs text-slate-500">
                          Confidence Model
                        </p>
                      </div>

                      {/* PROBABILITIES */}
                      <div className="mt-7">
                        <h4 className="mb-5 text-sm font-bold text-slate-800">
                          Distribusi Probabilitas
                        </h4>

                        <ProbabilityBars
                          probabilities={result.probabilities}
                        />
                      </div>

                      <ExplanationCard
                        explanation={result.explanation}
                        prediction={result.prediction}
                        originalImage={preview}
                      />

                      {/* IMAGE QUALITY */}
                      {result.image_quality && (
                        <div className="mt-7 border-t border-slate-100 pt-6">
                          <h4 className="text-sm font-bold text-slate-800">
                            Pemeriksaan Kualitas Foto
                          </h4>

                          <div className="mt-4 space-y-3 text-sm">
                            <div className="flex justify-between gap-3">
                              <span className="text-slate-500">
                                Resolusi
                              </span>

                              <span className="font-semibold">
                                {result.image_quality.width} ×{" "}
                                {result.image_quality.height}
                              </span>
                            </div>

                            <div className="flex justify-between gap-3">
                              <span className="text-slate-500">
                                Ketajaman
                              </span>

                              <span className="font-semibold">
                                {result.image_quality.blur_score.toFixed(2)}
                              </span>
                            </div>

                            <div className="flex justify-between gap-3">
                              <span className="text-slate-500">
                                Pencahayaan
                              </span>

                              <span className="font-semibold">
                                {result.image_quality.brightness.toFixed(2)}
                              </span>
                            </div>

                            <div className="flex justify-between gap-3">
                              <span className="text-slate-500">
                                Status
                              </span>

                              <span
                                className={`font-semibold ${
                                  result.image_quality.status === "passed"
                                    ? "text-emerald-600"
                                    : "text-amber-600"
                                }`}
                              >
                                {result.image_quality.status === "passed"
                                  ? "Memenuhi Pemeriksaan"
                                  : "Perlu Perhatian"}
                              </span>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* WARNINGS */}
                      {result.warnings?.length > 0 && (
                        <div className="mt-6 rounded-xl bg-amber-50 p-4 text-xs leading-relaxed text-amber-800">
                          <p className="mb-2 font-bold">
                            Catatan Pemeriksaan
                          </p>

                          {result.warnings.map((warning, index) => (
                            <p key={index} className="mt-1">
                              • {warning}
                            </p>
                          ))}
                        </div>
                      )}

                      {/* DISCLAIMER */}
                      <div className="mt-6 rounded-xl bg-slate-50 p-4">
                        <div className="flex items-start gap-3">
                          <Icon
                            name="info"
                            className="h-5 w-5 shrink-0 text-slate-500"
                          />

                          <p className="text-xs leading-relaxed text-slate-600">
                            Hasil berdasarkan foto cangkang telur. Kategori Segar,
                            Setengah Segar, atau Busuk tidak dapat
                            memastikan kondisi internal maupun
                            keamanan konsumsi telur.
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </section>
              </div>
            </div>
          )}

          {/* ======================================
              HISTORY PAGE
          ====================================== */}

          {page === "history" && (
            <div className="space-y-6">
              {/* HISTORY SUMMARY */}
              <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                <StatCard
                  title="Total Pengecekan"
                  value={total}
                  subtitle="Histori tersimpan"
                />

                <StatCard
                  title="Telur Segar"
                  value={countByClass("Fresh")}
                  subtitle="Prediksi Fresh"
                  accent="text-emerald-600"
                />

                <StatCard
                  title="Setengah Segar"
                  value={countByClass("Half-Fresh")}
                  subtitle="Prediksi Half-Fresh"
                  accent="text-amber-600"
                />

                <StatCard
                  title="Telur Busuk"
                  value={countByClass("Rotten")}
                  subtitle="Prediksi Rotten"
                  accent="text-rose-600"
                />
              </div>

              {/* HISTORY TABLE */}
              <HistoryTable
                data={history}
                onClear={clearHistory}
              />

              {/* HISTORY INFO */}
              <div className="flex items-start gap-3 rounded-2xl border border-slate-200 bg-white p-5">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-emerald-50 text-emerald-600">
                  <Icon name="info" className="h-5 w-5" />
                </div>

                <div>
                  <p className="text-sm font-semibold text-slate-800">
                    Informasi Penyimpanan
                  </p>

                  <p className="mt-2 text-xs leading-relaxed text-slate-500">
                    Histori disimpan menggunakan localStorage
                    pada browser ini, maksimal {MAX_HISTORY} hasil.
                    Data tidak tersinkronisasi antarperangkat dan
                    dapat hilang jika data browser dihapus.
                    Foto asli tidak disimpan dalam histori.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* ======================================
              FOOTER
          ====================================== */}

          <footer className="mt-10 border-t border-slate-200 pt-6 text-center text-xs text-slate-400">
            EggVision © {new Date().getFullYear()} —
            Egg Freshness Classification System
          </footer>
        </div>
      </main>
    </div>
  );
}
