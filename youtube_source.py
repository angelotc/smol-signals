"""Resolve channels/videos and pull transcripts.

Ported from src/lib/youtube/discovery.ts and transcripts.ts. Uses the YouTube
Data API when YOUTUBE_API_KEY is set; otherwise falls back to the public RSS
feed (most recent ~15 uploads) and handle-page scraping so the app still works
with no YouTube API key. YouTube page/feed/transcript network access goes
through Zyte; set ZYTE_API_KEY in production.
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import requests
from requests.structures import CaseInsensitiveDict
from youtube_transcript_api import (
    IpBlocked,
    InvalidVideoId,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

from tickers import extract_channel_id, extract_handle, extract_video_id

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; small-signals/1.0)"}
_API = "https://www.googleapis.com/youtube/v3"
_ZYTE_API = "https://api.zyte.com/v1/extract"


def _zyte_api_key():
    return (os.environ.get("ZYTE_API_KEY") or "").strip() or None


def _zyte_timeout() -> int:
    try:
        return max(10, int(os.environ.get("ZYTE_API_TIMEOUT", "45")))
    except ValueError:
        return 45


def _transcript_retries() -> int:
    try:
        return max(0, int(os.environ.get("TRANSCRIPT_FETCH_RETRIES", "2")))
    except ValueError:
        return 2


def _transcript_retry_backoff() -> float:
    try:
        return max(0.0, float(os.environ.get("TRANSCRIPT_FETCH_RETRY_BACKOFF", "1.5")))
    except ValueError:
        return 1.5


def _url_with_params(url: str, params: dict | None) -> str:
    if not params:
        return url
    prepared = requests.Request("GET", url, params=params).prepare()
    return prepared.url or url


def _decode_zyte_body(data: dict) -> bytes:
    body = data.get("httpResponseBody")
    if not body:
        return b""
    return base64.b64decode(body)


def _zyte_extract(payload: dict, timeout: int) -> dict:
    zyte_key = _zyte_api_key()
    if not zyte_key:
        raise RuntimeError("Set ZYTE_API_KEY to fetch YouTube pages, feeds, and transcripts.")
    resp = requests.post(_ZYTE_API, auth=(zyte_key, ""), json=payload,
                         timeout=max(timeout, _zyte_timeout()))
    if not resp.ok:
        raise RuntimeError(f"Zyte API HTTP {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    status = data.get("status")
    if isinstance(status, int) and status >= 400:
        raise RuntimeError(f"Zyte target HTTP {status}")
    return data


def _get_text(url: str, params: dict | None = None, *, browser: bool = False,
              timeout: int = 20) -> str:
    """Fetch text through Zyte API.

    Zyte's structured extraction catalog does not have a YouTube-specific type
    we can consume here. This app still needs the raw YouTube page/feed content,
    so we use browserHtml for YouTube pages and httpResponseBody for RSS/XML.
    """
    final_url = _url_with_params(url, params)
    payload = {"url": final_url, "browserHtml" if browser else "httpResponseBody": True}
    data = _zyte_extract(payload, timeout)
    if browser:
        return data.get("browserHtml") or ""
    return _decode_zyte_body(data).decode("utf-8", errors="replace")


@dataclass
class Channel:
    channel_id: str | None
    handle: str | None
    title: str
    uploads_playlist_id: str | None = None


@dataclass
class Video:
    video_id: str
    title: str
    published_at: datetime | None
    url: str


def _api_key():
    return (os.environ.get("YOUTUBE_API_KEY") or "").strip() or None


def _get_json(url, params):
    resp = requests.get(url, params=params, headers=_HEADERS, timeout=20)
    if not resp.ok:
        raise RuntimeError(f"YouTube API HTTP {resp.status_code}: {resp.text[:200]}")
    return resp.json()


def _parse_dt(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def resolve_channel(value: str) -> Channel:
    key = _api_key()
    channel_id = extract_channel_id(value)
    handle = extract_handle(value)

    if key and channel_id:
        return _channel_from_api(_get_channel(id=channel_id), value)
    if key and handle:
        return _channel_from_api(_get_channel(forHandle=f"@{handle}"), value)
    if key:
        data = _get_json(f"{_API}/search", {"part": "snippet", "type": "channel",
                                            "maxResults": 1, "q": value, "key": key})
        items = data.get("items") or []
        if not items:
            raise RuntimeError(f'No YouTube channel found for "{value}".')
        cid = items[0]["snippet"]["channelId"]
        return _channel_from_api(_get_channel(id=cid), value)

    if channel_id:
        return Channel(channel_id, handle, handle and f"@{handle}" or channel_id)
    if handle:
        return _scrape_handle(handle)
    raise RuntimeError("Set YOUTUBE_API_KEY to search by name, or paste a channel URL / @handle.")


def _get_channel(**filt):
    key = _api_key()
    params = {"part": "snippet,contentDetails", "key": key, **filt}
    data = _get_json(f"{_API}/channels", params)
    items = data.get("items") or []
    if not items:
        raise RuntimeError("No YouTube channel found.")
    return items[0]


def _channel_from_api(item: dict, source: str) -> Channel:
    snippet = item.get("snippet", {})
    uploads = item.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
    return Channel(
        channel_id=item.get("id"),
        handle=(snippet.get("customUrl") or "").lstrip("@") or extract_handle(source),
        title=snippet.get("title") or item.get("id"),
        uploads_playlist_id=uploads,
    )


def _scrape_handle(handle: str) -> Channel:
    try:
        html = _get_text(f"https://www.youtube.com/@{handle}", browser=True, timeout=20)
    except RuntimeError as e:
        raise RuntimeError(f"Could not fetch @{handle}: {e}") from e
    m = re.search(r'"channelId":"(UC[\w-]+)"', html) or re.search(r'"externalId":"(UC[\w-]+)"', html)
    if not m:
        raise RuntimeError(f"Could not resolve channel ID for @{handle}.")
    title = re.search(r'<meta property="og:title" content="([^"]+)"', html)
    return Channel(m.group(1), handle, title.group(1) if title else f"@{handle}")


def list_channel_videos(channel: Channel, limit: int | None = None) -> list[Video]:
    """Every upload for the channel (or at most `limit`, if given).

    With YOUTUBE_API_KEY we page through the full uploads playlist. Without it we
    fall back to the public RSS feed, which only exposes the ~15 most recent
    uploads — a limitation of the feed, not this code.
    """
    key = _api_key()
    if key and channel.uploads_playlist_id:
        return _list_playlist_videos(channel.uploads_playlist_id, limit)
    if channel.channel_id:
        return _list_rss_videos(channel.channel_id, limit)
    return []


def _list_playlist_videos(playlist_id: str, limit: int | None) -> list[Video]:
    key = _api_key()
    videos: list[Video] = []
    page_token = None
    while limit is None or len(videos) < limit:
        remaining = 50 if limit is None else min(50, limit - len(videos))
        params = {"part": "snippet,contentDetails", "playlistId": playlist_id,
                  "maxResults": remaining, "key": key}
        if page_token:
            params["pageToken"] = page_token
        data = _get_json(f"{_API}/playlistItems", params)
        for item in data.get("items", []):
            vid = item.get("contentDetails", {}).get("videoId") or \
                item.get("snippet", {}).get("resourceId", {}).get("videoId")
            if not vid:
                continue
            snippet = item.get("snippet", {})
            videos.append(Video(vid, snippet.get("title") or vid,
                                _parse_dt(snippet.get("publishedAt")),
                                f"https://www.youtube.com/watch?v={vid}"))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return videos if limit is None else videos[:limit]


def _list_rss_videos(channel_id: str, limit: int | None) -> list[Video]:
    try:
        rss_text = _get_text("https://www.youtube.com/feeds/videos.xml",
                             params={"channel_id": channel_id}, timeout=20)
    except RuntimeError as e:
        raise RuntimeError(f"Could not fetch RSS feed: {e}") from e
    videos: list[Video] = []
    for entry in re.findall(r"<entry>([\s\S]*?)</entry>", rss_text):
        vid = re.search(r"<yt:videoId>([\w-]+)</yt:videoId>", entry)
        if not vid:
            continue
        title = re.search(r"<title>([\s\S]*?)</title>", entry)
        published = re.search(r"<published>([\s\S]*?)</published>", entry)
        videos.append(Video(
            vid.group(1),
            _decode_html(title.group(1)) if title else vid.group(1),
            _parse_dt(published.group(1)) if published else None,
            f"https://www.youtube.com/watch?v={vid.group(1)}",
        ))
        if limit is not None and len(videos) >= limit:
            break
    return videos


def single_video(value: str) -> Video | None:
    vid = extract_video_id(value)
    if not vid:
        return None
    published = None
    try:
        html = _get_text(f"https://www.youtube.com/watch?v={vid}", browser=True, timeout=20)
        m = re.search(r'"publishDate":"([^"]+)"', html) or \
            re.search(r'<meta itemprop="datePublished" content="([^"]+)"', html)
        if m:
            published = _parse_dt(m.group(1))
        tm = re.search(r'<meta property="og:title" content="([^"]+)"', html)
        title = _decode_html(tm.group(1)) if tm else vid
    except RuntimeError as e:
        if "ZYTE_API_KEY" in str(e):
            raise
        title = vid
    except Exception:
        title = vid
    return Video(vid, title, published, f"https://www.youtube.com/watch?v={vid}")


# --- Transcript fetching through Zyte ----------------------------------------


class _TimeoutSession(requests.Session):
    """Session that injects a default timeout (youtube-transcript-api omits one)."""

    def __init__(self, timeout: int):
        super().__init__()
        self._timeout = timeout

    def request(self, *args, **kwargs):
        kwargs.setdefault("timeout", self._timeout)
        return super().request(*args, **kwargs)


class _ZyteExtractSession(_TimeoutSession):
    """Requests-compatible session backed by Zyte extract API.

    This avoids certificate handling while preserving the `requests.Session`
    interface expected by youtube-transcript-api.
    """

    def request(self, method, url, **kwargs):
        timeout = kwargs.pop("timeout", self._timeout)
        params = kwargs.pop("params", None)
        headers = CaseInsensitiveDict(kwargs.pop("headers", {}) or {})

        final_url = _url_with_params(url, params)
        payload = {
            "url": final_url,
            "httpResponseBody": True,
            "httpRequestMethod": method.upper(),
        }
        custom_headers = []
        for name, value in headers.items():
            if name.lower() in {"cookie", "user-agent", "accept-encoding", "connection"}:
                continue
            custom_headers.append({"name": name, "value": str(value)})

        if "json" in kwargs and kwargs["json"] is not None:
            payload["httpRequestText"] = json.dumps(kwargs["json"])
            if "content-type" not in {h["name"].lower() for h in custom_headers}:
                custom_headers.append({"name": "Content-Type", "value": "application/json"})
        elif "data" in kwargs and kwargs["data"] is not None:
            data = kwargs["data"]
            if isinstance(data, bytes):
                payload["httpRequestBody"] = base64.b64encode(data).decode("ascii")
            else:
                payload["httpRequestText"] = str(data)

        if custom_headers:
            payload["customHttpRequestHeaders"] = custom_headers

        data = _zyte_extract(payload, timeout)
        response = requests.Response()
        response.status_code = int(data.get("status") or 200)
        response.url = data.get("url") or final_url
        response.headers = CaseInsensitiveDict()
        response._content = _decode_zyte_body(data)
        response.encoding = "utf-8"
        return response


def _fetch_via_zyte(video_id: str, langs: list[str], timeout: int) -> str:
    session = _ZyteExtractSession(timeout)
    session.headers.update(_HEADERS)
    fetched = YouTubeTranscriptApi(http_client=session).fetch(video_id, languages=langs)
    return "\n".join(snippet.text for snippet in fetched)


_PERMANENT_TRANSCRIPT_ERRORS = (
    InvalidVideoId,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

_RETRYABLE_TRANSCRIPT_ERRORS = (
    IpBlocked,
    RequestBlocked,
    requests.RequestException,
)


def _is_retryable_transcript_error(exc: Exception) -> bool:
    if isinstance(exc, _PERMANENT_TRANSCRIPT_ERRORS):
        return False
    if isinstance(exc, _RETRYABLE_TRANSCRIPT_ERRORS):
        return True

    message = str(exc).lower()
    permanent_markers = (
        "content isn't available",
        "content is not available",
        "could not retrieve a transcript",
        "no transcript",
        "transcripts disabled",
        "transcripts are disabled",
        "unplayable",
        "video unavailable",
    )
    if any(marker in message for marker in permanent_markers):
        return False

    retryable_markers = (
        "connection",
        "connect timeout",
        "read timed out",
        "timeout",
        "temporarily unavailable",
        "too many requests",
        "zyte api http 429",
        "zyte api http 5",
        "zyte target http 429",
        "zyte target http 5",
    )
    return any(marker in message for marker in retryable_markers)


def fetch_transcript(video_id: str) -> str:
    """Return transcript text through Zyte extract API."""
    lang = (os.environ.get("YOUTUBE_TRANSCRIPT_LANG") or "en").split("-")[0]
    langs = [lang, "en"] if lang != "en" else ["en"]
    attempts = _transcript_retries() + 1
    backoff = _transcript_retry_backoff()
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            return _fetch_via_zyte(video_id, langs, timeout=25)
        except Exception as e:  # noqa: BLE001 - classify before retrying
            last_error = e
            if attempt >= attempts or not _is_retryable_transcript_error(e):
                raise
            sleep_for = backoff * (2 ** (attempt - 1))
            if sleep_for > 0:
                time.sleep(sleep_for)

    assert last_error is not None
    raise last_error


def _decode_html(value: str) -> str:
    return (value.replace("&amp;", "&").replace("&quot;", '"')
            .replace("&#39;", "'").replace("&lt;", "<").replace("&gt;", ">"))
