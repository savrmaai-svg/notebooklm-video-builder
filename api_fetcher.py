from pathlib import Path
from urllib.parse import quote_plus

import requests


DOWNLOAD_TIMEOUT = 45
SEARCH_TIMEOUT = 20
MIN_RESULTS_BEFORE_FALLBACK = 5
DEFAULT_CLIP_LIMIT = 8


class VideoFetchError(RuntimeError):
    pass


def _request_json(url, headers=None):
    response = requests.get(url, headers=headers or {}, timeout=SEARCH_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _best_pexels_link(video):
    files = video.get("video_files", [])
    mp4_files = [item for item in files if item.get("file_type") == "video/mp4" and item.get("link")]
    if not mp4_files:
        return None

    preferred = sorted(
        mp4_files,
        key=lambda item: (
            item.get("quality") != "hd",
            abs((item.get("width") or 1280) - 1920),
            -(item.get("height") or 0),
        ),
    )
    return preferred[0]["link"]


def fetch_pexels_videos(topic, api_key, per_page=DEFAULT_CLIP_LIMIT):
    if not api_key:
        return []

    url = f"https://api.pexels.com/videos/search?query={quote_plus(topic)}&per_page={per_page}"
    data = _request_json(url, headers={"Authorization": api_key})
    videos = []

    for video in data.get("videos", []):
        link = _best_pexels_link(video)
        if link:
            videos.append({"source": "pexels", "url": link})

    return videos


def _best_pixabay_link(video):
    variants = video.get("videos", {})
    for key in ("large", "medium", "small", "tiny"):
        link = variants.get(key, {}).get("url")
        if link:
            return link
    return None


def fetch_pixabay_videos(topic, api_key, per_page=DEFAULT_CLIP_LIMIT):
    if not api_key:
        return []

    url = f"https://pixabay.com/api/videos/?key={api_key}&q={quote_plus(topic)}&per_page={per_page}"
    data = _request_json(url)
    videos = []

    for video in data.get("hits", []):
        link = _best_pixabay_link(video)
        if link:
            videos.append({"source": "pixabay", "url": link})

    return videos


def _coverr_items(data):
    if isinstance(data, list):
        return data
    for key in ("videos", "hits", "data", "results"):
        value = data.get(key)
        if isinstance(value, list):
            return value
    return []


def _best_coverr_link(video):
    for key in ("download_url", "video_url", "url"):
        value = video.get(key)
        if isinstance(value, str) and value.startswith("http"):
            return value

    video_files = video.get("video_files") or video.get("files") or []
    if isinstance(video_files, dict):
        video_files = video_files.values()

    for item in video_files:
        if isinstance(item, str) and item.startswith("http"):
            return item
        if isinstance(item, dict):
            for key in ("url", "link", "download_url"):
                value = item.get(key)
                if isinstance(value, str) and value.startswith("http"):
                    return value
    return None


def fetch_coverr_videos(topic, per_page=DEFAULT_CLIP_LIMIT):
    url = f"https://api.coverr.co/videos?keywords={quote_plus(topic)}&page=1"
    data = _request_json(url)
    videos = []

    for video in _coverr_items(data):
        link = _best_coverr_link(video)
        if link:
            videos.append({"source": "coverr", "url": link})
        if len(videos) >= per_page:
            break

    return videos


def search_stock_videos(topic, pexels_api_key=None, pixabay_api_key=None, limit=DEFAULT_CLIP_LIMIT):
    if not topic.strip():
        raise VideoFetchError("Topic required hai.")

    errors = []
    found = []

    try:
        found.extend(fetch_pexels_videos(topic, pexels_api_key, per_page=limit))
    except Exception as exc:
        errors.append(f"Pexels failed: {exc}")

    if len(found) < MIN_RESULTS_BEFORE_FALLBACK:
        try:
            found.extend(fetch_pixabay_videos(topic, pixabay_api_key, per_page=limit))
        except Exception as exc:
            errors.append(f"Pixabay failed: {exc}")

    if not found:
        try:
            found.extend(fetch_coverr_videos(topic, per_page=limit))
        except Exception as exc:
            errors.append(f"Coverr failed: {exc}")

    unique = []
    seen = set()
    for video in found:
        url = video["url"]
        if url not in seen:
            seen.add(url)
            unique.append(video)
        if len(unique) >= limit:
            break

    if not unique:
        detail = " | ".join(errors) if errors else "No stock videos found."
        raise VideoFetchError(detail)

    return unique


def download_videos(videos, destination_dir):
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for index, video in enumerate(videos, start=1):
        source = video.get("source", "stock")
        output_path = destination / f"{source}_{index:03}.mp4"
        with requests.get(video["url"], stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
            response.raise_for_status()
            with output_path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
        saved_paths.append(output_path)

    return saved_paths


def fetch_and_download_videos(topic, destination_dir, pexels_api_key=None, pixabay_api_key=None, limit=DEFAULT_CLIP_LIMIT):
    videos = search_stock_videos(
        topic=topic,
        pexels_api_key=pexels_api_key,
        pixabay_api_key=pixabay_api_key,
        limit=limit,
    )
    return download_videos(videos, destination_dir)

