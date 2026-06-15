import { Client } from "@gradio/client";

// Same-origin in the Space; set NEXT_PUBLIC_GRADIO_API_URL for local dev
// (e.g. http://localhost:7860) when running `next dev` on a different port.
const API_BASE = (process.env.NEXT_PUBLIC_GRADIO_API_URL || "").replace(/\/$/, "");

function base(): string {
  if (API_BASE) return API_BASE;
  return typeof window !== "undefined" ? window.location.origin : "";
}

export type CallResult = {
  ticker: string;
  signal: string;
  summary: string;
  evidence_quote: string;
  why_current: string;
  company_name: string | null;
  verdict: string;
  alpha_pct: number | null;
  return_pct: number | null;
  score: number | null;
  horizon_days: number;
};

export type VideoResult = {
  video_id: string;
  title: string;
  url: string;
  published_at: string | null;
  has_call: boolean;
  summary: string;
  calls: CallResult[];
  error: string | null;
};

export type ChannelResult = {
  channel_id: string | null;
  title: string;
  model: string;
  reputation: number;
  measured_calls: number;
  win_rate: number;
  avg_weighted_score: number;
  videos: VideoResult[];
};

export type LeaderboardEntry = {
  channel_id: string;
  title: string;
  reputation: number;
  measured_calls: number;
  win_rate: number;
  avg_weighted_score: number;
  last_updated: string;
};

// A channel's detail page renders either a fresh analysis (ChannelResult) or the
// stored channel doc from the dataset — same fields, plus an optional timestamp.
export type ChannelDetail = ChannelResult & { last_updated?: string };

let clientPromise: Promise<Client> | null = null;
function gradio(): Promise<Client> {
  if (!clientPromise) clientPromise = Client.connect(`${base()}/gradio`);
  return clientPromise;
}

export async function analyzeChannel(target: string): Promise<ChannelResult> {
  const client = await gradio();
  const res = await client.predict("/analyze_channel", [target]);
  return (res.data as ChannelResult[])[0];
}

// --- Streaming analyze (live progress) --------------------------------------

export type StreamStatus = {
  type: "status";
  fraction: number;
  label: string;
  current?: string;
  videos_done?: number;
  videos_total?: number;
};
export type StreamVideo = {
  type: "video";
  video: VideoResult;
  videos_done: number;
  videos_total: number;
  fraction: number;
};
export type StreamResult = { type: "result"; result: ChannelResult };
export type StreamError = { type: "error"; error: string };
export type StreamEvent = StreamStatus | StreamVideo | StreamResult | StreamError;

/**
 * Run an analysis and stream progress. `onEvent` fires for every status/video
 * event so the UI can show a progress bar + the current video and fill the
 * table incrementally. Resolves to the final ChannelResult.
 */
export async function analyzeChannelStream(
  target: string,
  onEvent: (e: StreamEvent) => void,
): Promise<ChannelResult> {
  const client = await gradio();
  const job = client.submit("/analyze_channel_stream", [target]);
  let final: ChannelResult | null = null;
  for await (const msg of job) {
    if (msg.type !== "data") continue;
    const ev = (msg.data as StreamEvent[])[0];
    if (!ev) continue;
    if (ev.type === "error") throw new Error(ev.error);
    if (ev.type === "result") final = ev.result;
    onEvent(ev);
  }
  if (!final) throw new Error("Analysis ended without a result");
  return final;
}

// --- Live activity (everyone's in-flight runs) ------------------------------

export type ActiveRun = {
  run_id: string;
  target: string;
  status: string; // queued | running
  fraction: number;
  label: string;
  channel_id: string | null;
  title: string;
  videos_done: number;
  videos_total: number;
  error: string | null;
  started_at: number;
  updated_at: number;
};

export async function getActiveRuns(): Promise<ActiveRun[]> {
  const res = await fetch(`${base()}/api/runs`, { cache: "no-store" });
  if (!res.ok) return [];
  return res.json();
}

export async function getLeaderboard(): Promise<LeaderboardEntry[]> {
  const res = await fetch(`${base()}/api/leaderboard`, { cache: "no-store" });
  if (!res.ok) return [];
  return res.json();
}

// Key under which the home page stashes a freshly-analyzed result in
// sessionStorage so the channel page can render it without a dataset refetch
// (and so single-video results, which aren't persisted, still display).
export function stashKey(id: string): string {
  return `ss-channel:${id}`;
}

export async function getChannel(channelId: string): Promise<ChannelDetail | null> {
  const res = await fetch(`${base()}/api/channel/${encodeURIComponent(channelId)}`, {
    cache: "no-store",
  });
  if (!res.ok) return null;
  return res.json();
}

export async function getHealth(): Promise<{ status: string; model: string; storage: string }> {
  const res = await fetch(`${base()}/api/health`, { cache: "no-store" });
  return res.json();
}
