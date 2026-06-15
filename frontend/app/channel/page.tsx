"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import ChannelView from "../components/ChannelView";
import {
  analyzeChannelStream,
  getChannel,
  stashKey,
  type ChannelDetail,
} from "../lib/api";

function ChannelDetailInner() {
  const params = useSearchParams();
  const id = params.get("id") || "";
  const [channel, setChannel] = useState<ChannelDetail | null>(null);
  const [state, setState] = useState<"loading" | "ready" | "missing">("loading");
  const [refreshing, setRefreshing] = useState(false);
  const [progress, setProgress] = useState<{ fraction: number; label: string } | null>(null);
  const [refreshError, setRefreshError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) {
      setState("missing");
      return;
    }
    // Prefer the freshly-analyzed result stashed on the home page (instant, and
    // the only source for single videos that aren't in the dataset).
    try {
      const cached = sessionStorage.getItem(stashKey(id));
      if (cached) {
        setChannel(JSON.parse(cached));
        setState("ready");
        return;
      }
    } catch {
      /* fall through to the API */
    }
    if (id.startsWith("vid:")) {
      setState("missing"); // single-video result not persisted; nothing to fetch
      return;
    }
    getChannel(id)
      .then((c) => {
        setChannel(c);
        setState(c ? "ready" : "missing");
      })
      .catch(() => setState("missing"));
  }, [id]);

  // Re-run analysis for this channel. The backend is incremental: already-scored
  // videos are reused and only new uploads (plus pending calls that have matured)
  // do real work, so a refresh is cheap. Updates the view + stash in place.
  async function onRefresh() {
    const cid = channel?.channel_id;
    if (!cid || refreshing) return;
    setRefreshing(true);
    setRefreshError(null);
    setProgress({ fraction: 0, label: "Starting…" });
    try {
      await analyzeChannelStream(cid, (ev) => {
        if (ev.type === "status") {
          setProgress({ fraction: ev.fraction, label: ev.label });
        } else if (ev.type === "video") {
          setProgress({
            fraction: ev.fraction,
            label: `Scored ${ev.videos_done}/${ev.videos_total} videos`,
          });
        } else if (ev.type === "result") {
          // Apply the fresh result the instant it arrives (the commit runs in the
          // background server-side) rather than waiting for the stream to close.
          setChannel(ev.result);
          try {
            sessionStorage.setItem(stashKey(id), JSON.stringify(ev.result));
          } catch {
            /* sessionStorage may be unavailable; the API holds the fresh data anyway */
          }
        }
      });
    } catch (e) {
      setRefreshError(e instanceof Error ? e.message : String(e));
    } finally {
      setRefreshing(false);
      setProgress(null);
    }
  }

  return (
    <div className="wrap">
      <p style={{ marginBottom: 18 }}>
        <Link href="/">Leaderboard</Link>
      </p>

      {state === "loading" && (
        <section className="panel">
          <p className="muted" style={{ fontSize: 13 }}>
            <span className="spin" style={{ borderColor: "var(--muted)", borderTopColor: "transparent" }} />
            Loading channel…
          </p>
        </section>
      )}

      {state === "missing" && (
        <section className="panel">
          <h2 className="section-title">Channel not found</h2>
          <p className="muted" style={{ fontSize: 13 }}>
            No stored history for this channel. Run an analysis from the{" "}
            <Link href="/">leaderboard page</Link> to generate it.
          </p>
        </section>
      )}

      {state === "ready" && channel && (
        <section className="panel">
          {channel.channel_id && (
            <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 14 }}>
              <button className="primary" onClick={onRefresh} disabled={refreshing}>
                {refreshing ? (
                  <>
                    <span className="spin" />
                    Refreshing…
                  </>
                ) : (
                  "Refresh"
                )}
              </button>
            </div>
          )}
          {refreshing && progress && (
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
          {refreshError && <div className="error">Error: {refreshError}</div>}
          <ChannelView channel={channel} />
        </section>
      )}
    </div>
  );
}

export default function ChannelPage() {
  // useSearchParams requires a Suspense boundary for the static export build.
  return (
    <Suspense fallback={<div className="wrap" />}>
      <ChannelDetailInner />
    </Suspense>
  );
}
