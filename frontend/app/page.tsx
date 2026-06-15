"use client";

import { useEffect, useRef, useState } from "react";
import {
  analyzeChannelStream,
  getActiveRuns,
  getLeaderboard,
  getHealth,
  stashKey,
  type ActiveRun,
  type ChannelResult,
  type LeaderboardEntry,
} from "./lib/api";

// localStorage key holding the target of the run this tab kicked off, so a
// refresh can re-attach to the still-running server-side run instead of losing
// it. Cleared when the run finishes or errors.
const ACTIVE_TARGET_KEY = "ss-active-target";

// Signed percentage with the sole surviving color cue: green for gains,
// red for losses. Everything else in the UI is black/white.
function Pct({ v }: { v: number | null }) {
  if (v === null || v === undefined) return <span className="muted">—</span>;
  const cls = v > 0 ? "pos" : v < 0 ? "neg" : "";
  return <span className={cls}>{`${v >= 0 ? "+" : ""}${v.toFixed(1)}%`}</span>;
}

// ETA from observed throughput: elapsed/done already reflects the parallelism,
// so remaining ≈ elapsed × (total − done) / done. Self-corrects as it runs.
function formatEta(seconds: number): string {
  if (!isFinite(seconds) || seconds <= 0) return "";
  if (seconds < 60) return `~${Math.round(seconds)}s left`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `~${m}m ${s}s left`;
}

function resultId(r: ChannelResult): string {
  return r.channel_id ?? `vid:${r.videos[0]?.video_id ?? "unknown"}`;
}

export default function Home() {
  const [target, setTarget] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<{ fraction: number; label: string } | null>(null);
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([]);
  const [model, setModel] = useState<string>("");
  const [activeRuns, setActiveRuns] = useState<ActiveRun[]>([]);
  const resumedRef = useRef(false);

  useEffect(() => {
    getLeaderboard().then(setLeaderboard).catch(() => {});
    getHealth().then((h) => setModel(h.model)).catch(() => {});
  }, []);

  // Poll the global live feed, and on first load re-attach to a run this tab
  // started if it's still in flight (survives a refresh).
  useEffect(() => {
    let cancelled = false;
    async function poll() {
      const runs = await getActiveRuns().catch(() => [] as ActiveRun[]);
      if (cancelled) return;
      setActiveRuns(runs);
      if (!resumedRef.current) {
        resumedRef.current = true;
        let stashed: string | null = null;
        try {
          stashed = localStorage.getItem(ACTIVE_TARGET_KEY);
        } catch {
          /* localStorage may be unavailable */
        }
        if (stashed) {
          const match = runs.find(
            (r) => r.target.trim().toLowerCase() === stashed!.trim().toLowerCase(),
          );
          if (match) {
            setTarget(stashed);
            runAnalysis(stashed);
          } else {
            try {
              localStorage.removeItem(ACTIVE_TARGET_KEY);
            } catch {
              /* ignore */
            }
          }
        }
      }
    }
    poll();
    const id = setInterval(poll, 3000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function clearActiveTarget() {
    try {
      localStorage.removeItem(ACTIVE_TARGET_KEY);
    } catch {
      /* ignore */
    }
  }

  async function runAnalysis(t: string) {
    setLoading(true);
    setError(null);
    setProgress({ fraction: 0, label: "Starting…" });
    try {
      localStorage.setItem(ACTIVE_TARGET_KEY, t);
    } catch {
      /* ignore */
    }
    const startedAt = Date.now();
    try {
      await analyzeChannelStream(t, (ev) => {
        if (ev.type === "status") {
          setProgress({ fraction: ev.fraction, label: ev.label });
        } else if (ev.type === "video") {
          const { videos_done: done, videos_total: total } = ev;
          if (done >= total) {
            // Last video in: the backend now computes reputation + commits.
            setProgress({ fraction: ev.fraction, label: "All videos scored — finalizing…" });
          } else {
            const elapsed = (Date.now() - startedAt) / 1000;
            const eta = done > 0 ? formatEta((elapsed * (total - done)) / done) : "";
            setProgress({
              fraction: ev.fraction,
              label: `Scored ${done}/${total} videos${eta ? ` · ${eta}` : ""}`,
            });
          }
        } else if (ev.type === "result") {
          // The result event carries the full ChannelResult, and the backend has
          // already kicked off the dataset commit in a background thread. So we
          // navigate right here instead of awaiting the stream's close — waiting
          // for that trailing finalize is what left the UI stuck on "Saving to
          // leaderboard…". Stash the result so the channel page renders instantly.
          setProgress({ fraction: 1, label: "Done — opening channel…" });
          clearActiveTarget();
          const id = resultId(ev.result);
          try {
            sessionStorage.setItem(stashKey(id), JSON.stringify(ev.result));
          } catch {
            /* sessionStorage may be unavailable; channel page falls back to the API */
          }
          // Hard navigation, not router.push: this is a static export served by a
          // plain file server, and Next's client-side RSC navigation 404s on the
          // nested /channel segment file (the flat dotted name the client requests
          // doesn't exist on disk), which stalls the redirect. A full-page load hits
          // /channel/index.html directly; the page reads id from the query string.
          window.location.assign(`/channel/?id=${encodeURIComponent(id)}`);
        }
      });
    } catch (e) {
      clearActiveTarget();
      setError(e instanceof Error ? e.message : String(e));
      setLoading(false);
      setProgress(null);
    }
    // On success we navigate away, so we intentionally leave `loading` set
    // (the button stays disabled until the route changes).
  }

  function onAnalyze() {
    if (!target.trim() || loading) return;
    runAnalysis(target.trim());
  }

  return (
    <div className="wrap">
      <header className="hero">
        <h1>Smol Signals</h1>
        <p>Calls scored against SPY after 30 days.</p>
      </header>

      <section className="panel">
        <div className="controls">
          <div className="field">
            <label htmlFor="target">Analyze a channel or video</label>
            <input
              id="target"
              type="text"
              placeholder="@HandleName · youtube.com/@channel · a single video URL"
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && onAnalyze()}
              disabled={loading}
            />
          </div>
          <button className="primary" onClick={onAnalyze} disabled={loading}>
            {loading ? (
              <>
                <span className="spin" />
                Analyzing…
              </>
            ) : (
              "Analyze"
            )}
          </button>
        </div>
        {loading && progress && (
          <div className="progress-wrap">
            <div className="progress">
              <div
                className="progress-bar"
                style={{ width: `${Math.max(4, Math.round(progress.fraction * 100))}%` }}
              />
            </div>
            <p className="progress-label">{progress.label}</p>
          </div>
        )}
        {error && <div className="error">Error: {error}</div>}
      </section>

      {activeRuns.length > 0 && (
        <section className="panel">
          <h2 className="section-title">Live now</h2>
          <p className="section-sub">
            Analyses being processed on this Space right now — by anyone. These keep
            running even if you close the tab.
          </p>
          <table>
            <thead>
              <tr>
                <th>Channel / target</th>
                <th>Status</th>
                <th>Progress</th>
              </tr>
            </thead>
            <tbody>
              {activeRuns.map((r) => (
                <tr key={r.run_id}>
                  <td>{r.title || r.target}</td>
                  <td>{r.status}</td>
                  <td>
                    {r.videos_total
                      ? `${r.videos_done}/${r.videos_total} · ${Math.round(r.fraction * 100)}%`
                      : r.label}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      <section className="panel">
        <h2 className="section-title">Leaderboard</h2>
        <p className="section-sub">Channels ranked by reputation.</p>
        {leaderboard.length === 0 ? (
          <p className="muted" style={{ fontSize: 13 }}>
            No channels scored yet. Analyze one above to seed the board.
          </p>
        ) : (
          <table>
            <thead>
              <tr>
                <th className="lb-rank">#</th>
                <th>Channel</th>
                <th>Reputation</th>
                <th>Calls</th>
                <th>Win rate</th>
                <th>Avg α</th>
              </tr>
            </thead>
            <tbody>
              {leaderboard.map((e, i) => (
                <tr key={e.channel_id} className="lb-row">
                  <td className="lb-rank">{i + 1}</td>
                  <td>
                    <a href={`/channel/?id=${encodeURIComponent(e.channel_id)}`}>
                      {e.title}
                    </a>
                  </td>
                  <td>
                    <b>{e.reputation.toFixed(0)}</b>
                  </td>
                  <td>{e.measured_calls}</td>
                  <td>{(e.win_rate * 100).toFixed(0)}%</td>
                  <td><Pct v={e.avg_weighted_score} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <footer>
        {model && <><span className="modeltag">{model}</span> · </>}
        Calls scored against SPY via Yahoo Finance. Not investment advice.
      </footer>
    </div>
  );
}
