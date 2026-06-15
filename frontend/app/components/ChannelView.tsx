import type { ChannelDetail } from "../lib/api";

function signalClass(signal: string): string {
  const s = signal.toLowerCase();
  if (s === "buy") return "buy";
  if (s === "sell") return "sell";
  if (s === "none") return "none";
  return "hold";
}

// Signed percentage with the sole surviving color cue: green for gains,
// red for losses. Everything else in the UI is black/white.
function Pct({ v }: { v: number | null }) {
  if (v === null || v === undefined) return <span className="muted">—</span>;
  const cls = v > 0 ? "pos" : v < 0 ? "neg" : "";
  return <span className={cls}>{`${v >= 0 ? "+" : ""}${v.toFixed(1)}%`}</span>;
}

/** Reputation card + per-video calls table for a single channel. */
export default function ChannelView({ channel }: { channel: ChannelDetail }) {
  const videos = channel.videos ?? [];
  const scannedWithCalls = videos.filter((v) => v.has_call).length;

  return (
    <>
      <div className="repcard">
        <div className="score">
          {channel.reputation.toFixed(0)}
          <small> / 100</small>
        </div>
        <div>
          <div style={{ fontSize: 17, fontWeight: 600 }}>{channel.title}</div>
          <div className="stat">
            Reputation (neutral = 50) · <b>{channel.measured_calls}</b> measured calls ·
            win rate <b>{(channel.win_rate * 100).toFixed(0)}%</b> · avg weighted alpha{" "}
            <b><Pct v={channel.avg_weighted_score} /></b>
          </div>
          <div className="modeltag">model {channel.model}</div>
        </div>
      </div>

      <p className="section-sub" style={{ marginTop: 18 }}>
        {videos.length} videos · {scannedWithCalls} with calls. Calls only score once 30
        days have elapsed since publish; recent ones show as{" "}
        <span className="verdict pending">pending</span>.
      </p>

      <table>
        <thead>
          <tr>
            <th>Video</th>
            <th>Ticker</th>
            <th>Call</th>
            <th>30d α</th>
            <th>Verdict</th>
            <th>Summary</th>
          </tr>
        </thead>
        <tbody>
          {videos.flatMap((v) => {
            if (v.error)
              return [
                <tr key={v.video_id}>
                  <td>
                    <a href={v.url} target="_blank" rel="noreferrer">
                      {v.title}
                    </a>
                  </td>
                  <td className="ticker">—</td>
                  <td colSpan={4} className="muted">
                    error: {v.error}
                  </td>
                </tr>,
              ];
            if (!v.has_call)
              return [
                <tr key={v.video_id}>
                  <td>
                    <a href={v.url} target="_blank" rel="noreferrer">
                      {v.title}
                    </a>
                  </td>
                  <td className="ticker">—</td>
                  <td>
                    <span className="pill none">none</span>
                  </td>
                  <td className="muted">—</td>
                  <td className="muted">—</td>
                  <td className="muted">{v.summary || "no call"}</td>
                </tr>,
              ];
            return v.calls.map((c, i) => (
              <tr key={`${v.video_id}-${i}`}>
                <td>
                  {i === 0 ? (
                    <a href={v.url} target="_blank" rel="noreferrer">
                      {v.title}
                    </a>
                  ) : (
                    ""
                  )}
                </td>
                <td className="ticker">{c.ticker}</td>
                <td>
                  <span className={`pill ${signalClass(c.signal)}`}>{c.signal}</span>
                </td>
                <td><Pct v={c.alpha_pct} /></td>
                <td>
                  <span className="verdict">
                    {c.verdict}
                    {c.score !== null && (
                      <>
                        {" ("}
                        <span className={c.score > 0 ? "pos" : c.score < 0 ? "neg" : ""}>
                          {`${c.score >= 0 ? "+" : ""}${c.score.toFixed(1)}`}
                        </span>
                        {")"}
                      </>
                    )}
                  </span>
                </td>
                <td>{c.summary}</td>
              </tr>
            ));
          })}
        </tbody>
      </table>
    </>
  );
}
