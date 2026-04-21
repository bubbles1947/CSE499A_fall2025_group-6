// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ChatMessage {
  id?: number;
  role: "user" | "assistant";
  content: string;
  timestamp?: string;
}

export interface AnalysisResponse {
  is_diseased: boolean;
  disease_name: string;
  confidence: "High" | "Medium" | "Low";
  symptoms: string[];
  treatment: string[];
  prevention: string[];
  raw_response?: string;
}

export interface Crop {
  id: string;
  name: string;
  name_bn: string;
  icon: string;
}

export interface AdvisoryResponse {
  crop: string;
  topic: string;
  content: string;
}

export type WeatherRiskLevel = "HIGH" | "MEDIUM" | "LOW";

export interface WeatherRiskAlert {
  disease: string;
  crop: string;
  risk: WeatherRiskLevel;
  reason: string;
}

export interface WeatherRiskResponse {
  location: { lat: number; lon: number };
  current:  { temperature: number; humidity: number };
  forecast_summary: { avg_temp: number; avg_humidity: number };
  risks: WeatherRiskAlert[];
  updated_at: string;
}

export type CommunitySeverity = "Low" | "Medium" | "High";

export interface CommunityReport {
  id: number;
  disease_name: string;
  crop_name: string;
  location_name: string;
  division: string | null;
  severity: string;
  notes: string | null;
  created_at: string;
  report_date: string;
}

export interface CommunityReportCreate {
  disease_name: string;
  crop_name: string;
  location_name: string;
  division?: string;
  severity?: CommunitySeverity;
  notes?: string | null;
}

export interface DiseaseCount {
  disease_name: string;
  count: number;
}

export interface LocationCount {
  location_name: string;
  count: number;
}

export interface DivisionCount {
  division: string;
  count: number;
}

export interface CommunityStats {
  total_reports: number;
  top_diseases: DiseaseCount[];
  top_locations: LocationCount[];
  recent_reports: CommunityReport[];
  reports_by_division: DivisionCount[];
}

// ---------------------------------------------------------------------------
// Base URL
// ---------------------------------------------------------------------------

const API_BASE =
  (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function checkResponse(res: Response): Promise<Response> {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail ?? body.error ?? detail;
    } catch {
      // non-JSON error body — keep the status string
    }
    throw new ApiError(res.status, detail);
  }
  return res;
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

/**
 * Send a chat message and stream the assistant reply via SSE.
 *
 * The backend emits newline-delimited `data: <token>` lines.
 * `onToken` is called for each token; `onDone` is called on `data: [DONE]`.
 * Any `data: [ERROR] …` line is thrown as an Error.
 */
export async function sendChatMessage(
  sessionId: string,
  message: string,
  language: string,
  onToken: (token: string) => void,
  onDone: () => void,
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/chat/message`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message, language }),
  });

  await checkResponse(res);

  const reader = res.body?.getReader();
  if (!reader) throw new Error("Response body is not readable.");

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by double newlines; process complete frames only.
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? ""; // keep the incomplete tail for the next chunk

    for (const frame of frames) {
      for (const line of frame.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        const payload = line.slice(6); // strip "data: "

        if (payload === "[DONE]") {
          onDone();
          return;
        }

        if (payload.startsWith("[ERROR]")) {
          throw new Error(payload.slice(8).trim());
        }

        // The backend escapes embedded newlines as "\\n".
        onToken(payload.replace(/\\n/g, "\n"));
      }
    }
  }

  // Stream ended without [DONE] — treat as done anyway.
  onDone();
}

export async function getChatHistory(sessionId: string): Promise<ChatMessage[]> {
  const res = await fetch(`${API_BASE}/api/chat/history/${encodeURIComponent(sessionId)}`);
  if (res.status === 404) return []; // new session — no history yet
  await checkResponse(res);
  const data = await res.json();
  return (data.messages ?? []) as ChatMessage[];
}

export async function clearChatSession(sessionId: string): Promise<void> {
  const res = await fetch(
    `${API_BASE}/api/chat/session/${encodeURIComponent(sessionId)}`,
    { method: "DELETE" },
  );
  // 404 means it was already gone — that's fine
  if (res.status !== 404) await checkResponse(res);
}

// ---------------------------------------------------------------------------
// Image analysis
// ---------------------------------------------------------------------------

export async function analyzeImage(file: File): Promise<AnalysisResponse> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API_BASE}/api/analyze/image`, {
    method: "POST",
    body: form,
    // Do NOT set Content-Type — the browser must set the multipart boundary.
  });

  await checkResponse(res);
  return res.json() as Promise<AnalysisResponse>;
}

// ---------------------------------------------------------------------------
// Advisory
// ---------------------------------------------------------------------------

export async function getCrops(): Promise<Crop[]> {
  const res = await fetch(`${API_BASE}/api/advisory/crops`);
  await checkResponse(res);
  return res.json() as Promise<Crop[]>;
}

export async function getTopics(): Promise<{ id: string; name: string; name_bn: string }[]> {
  const res = await fetch(`${API_BASE}/api/advisory/topics`);
  await checkResponse(res);
  return res.json();
}

export async function getAdvisory(
  crop: string,
  topic: string,
  question?: string,
): Promise<AdvisoryResponse> {
  const res = await fetch(`${API_BASE}/api/advisory/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ crop, topic, question: question ?? null }),
  });
  await checkResponse(res);
  return res.json() as Promise<AdvisoryResponse>;
}

// ---------------------------------------------------------------------------
// Weather disease-risk forecast
// ---------------------------------------------------------------------------

export async function getWeatherRisk(
  lat?: number,
  lon?: number,
): Promise<WeatherRiskResponse> {
  const params = new URLSearchParams();
  if (lat !== undefined) params.set("lat", String(lat));
  if (lon !== undefined) params.set("lon", String(lon));

  const query = params.toString();
  const url = `${API_BASE}/api/weather/risk${query ? `?${query}` : ""}`;

  const res = await fetch(url);
  await checkResponse(res);
  return res.json() as Promise<WeatherRiskResponse>;
}

// ---------------------------------------------------------------------------
// Community disease board
// ---------------------------------------------------------------------------

export async function getDivisions(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/api/community/divisions`);
  await checkResponse(res);
  return res.json() as Promise<string[]>;
}

export async function getCommunityReports(opts?: {
  division?: string;
  crop?: string;
  limit?: number;
}): Promise<CommunityReport[]> {
  const params = new URLSearchParams();
  if (opts?.division) params.set("division", opts.division);
  if (opts?.crop)     params.set("crop", opts.crop);
  if (opts?.limit)    params.set("limit", String(opts.limit));

  const query = params.toString();
  const url = `${API_BASE}/api/community/reports${query ? `?${query}` : ""}`;

  const res = await fetch(url);
  await checkResponse(res);
  return res.json() as Promise<CommunityReport[]>;
}

export async function submitCommunityReport(
  body: CommunityReportCreate,
): Promise<CommunityReport> {
  const res = await fetch(`${API_BASE}/api/community/report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  await checkResponse(res);
  return res.json() as Promise<CommunityReport>;
}

export async function getCommunityStats(): Promise<CommunityStats> {
  const res = await fetch(`${API_BASE}/api/community/stats`);
  await checkResponse(res);
  return res.json() as Promise<CommunityStats>;
}

export async function seedCommunityReports(): Promise<{
  status: string;
  message: string;
  inserted: number;
}> {
  const res = await fetch(`${API_BASE}/api/community/seed`, { method: "POST" });
  await checkResponse(res);
  return res.json();
}
