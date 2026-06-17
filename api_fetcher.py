from pathlib import Path
import re
from urllib.parse import quote_plus, urlparse

import requests


DOWNLOAD_TIMEOUT = 45
SEARCH_TIMEOUT = 20
MIN_RESULTS_BEFORE_FALLBACK = 5
MIN_RELEVANT_RESULTS_BEFORE_PIXABAY = 3
DEFAULT_CLIP_LIMIT = 60
MAX_SEARCH_RESULTS_TO_COLLECT = 120
MAX_QUERY_COUNT = 80
MAX_TOPIC_WORDS = 8
MIN_VALID_CLIP_SECONDS = 8.0
KEYWORD_MAP = {
    "kohinoor": ["diamond", "crown jewels", "gold treasure", "india palace", "royal crown"],
    "jallianwala": ["india crowd", "memorial", "historical", "india history"],
    "tipu sultan": ["sword warrior", "india fort", "mughal", "royal palace"],
    "cricket": ["cricket india", "cricket stadium", "cricket bat", "sports crowd"],
    "mughal": ["india palace", "taj mahal", "mughal architecture", "royal fort"],
}
STOP_WORDS = {
    "ki",
    "ka",
    "ke",
    "ek",
    "aur",
    "hai",
    "hain",
    "tha",
    "thi",
    "mein",
    "me",
    "ko",
    "se",
    "bahut",
    "sachi",
    "kahani",
    "hindi",
    "podcast",
}


class VideoFetchError(RuntimeError):
    pass


def _request_json(url, headers=None):
    response = requests.get(url, headers=headers or {}, timeout=SEARCH_TIMEOUT)
    response.raise_for_status()
    return response.json()


def clean_topic(topic):
    topic = re.sub(r"[^\w\s-]", " ", topic, flags=re.UNICODE)
    words = [word for word in topic.split() if word.lower() not in STOP_WORDS]
    return " ".join(words[:MAX_TOPIC_WORDS])


def topic_lines(topic):
    lines = []
    for line in topic.splitlines():
        line = re.sub(r"^\s*\d+[\).\-\s]+", "", line).strip()
        cleaned = clean_topic(line)
        if cleaned:
            lines.append(cleaned)
    if lines:
        return lines[:MAX_QUERY_COUNT]

    cleaned = clean_topic(topic)
    return [cleaned] if cleaned else []


def search_queries(topic):
    raw_topic = topic.lower()
    cleaned_topics = topic_lines(topic)
    if not cleaned_topics:
        return []

    queries = []

    for trigger, mapped_queries in KEYWORD_MAP.items():
        if trigger in raw_topic:
            queries.extend(mapped_queries)

    for cleaned_topic in cleaned_topics:
        words = cleaned_topic.split()
        queries.append(cleaned_topic)

        if len(words) > 3:
            queries.append(" ".join(words[:3]))
        if len(words) > 1:
            queries.append(" ".join(words[:2]))
        queries.extend(
            [
                f"{cleaned_topic} cinematic video",
                f"{cleaned_topic} documentary",
                f"{cleaned_topic} history",
                f"{cleaned_topic} travel",
                f"{cleaned_topic} monument",
                f"{cleaned_topic} landscape",
            ]
        )

    unique = []
    for query in queries:
        if not query:
            continue
        enhanced_query = f"{query} cinematic 4k"
        if enhanced_query not in unique:
            unique.append(enhanced_query)
        if len(unique) >= MAX_QUERY_COUNT:
            break
    return unique


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


def is_landscape(width, height):
    return bool(width and height and width >= height)


def relevance_score(video, query, source):
    width = video.get("width") or 0
    height = video.get("height") or 0
    duration = video.get("duration") or 0
    score = 0

    if is_landscape(width, height):
        score += 100
    if width >= 1920:
        score += 50
    elif width >= 1280:
        score += 30
    if duration and 8 <= duration <= 30:
        score += 20
    if source == "pexels":
        score += 10
    if query:
        score += max(0, 10 - len(query.split()))

    return score


def fetch_pexels_videos(topic, api_key, per_page=DEFAULT_CLIP_LIMIT):
    if not api_key:
        return []

    url = (
        f"https://api.pexels.com/videos/search?query={quote_plus(topic)}"
        f"&per_page={per_page}&orientation=landscape&min_width=1280"
    )
    data = _request_json(url, headers={"Authorization": api_key})
    videos = []

    for video in data.get("videos", []):
        link = _best_pexels_link(video)
        if link:
            videos.append(
                {
                    "source": "pexels",
                    "url": link,
                    "query": topic,
                    "score": relevance_score(video, topic, "pexels"),
                }
            )

    return sorted(videos, key=lambda item: item.get("score", 0), reverse=True)


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

    url = (
        f"https://pixabay.com/api/videos/?key={api_key}&q={quote_plus(topic)}"
        f"&per_page={per_page}&min_width=1280&video_type=film"
    )
    data = _request_json(url)
    videos = []

    for video in data.get("hits", []):
        link = _best_pixabay_link(video)
        if link:
            videos.append(
                {
                    "source": "pixabay",
                    "url": link,
                    "query": topic,
                    "score": relevance_score(video, topic, "pixabay"),
                }
            )

    return sorted(videos, key=lambda item: item.get("score", 0), reverse=True)


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


def fetch_coverr_videos(topic, api_key=None, per_page=DEFAULT_CLIP_LIMIT):
    if not api_key:
        return []

    url = f"https://api.coverr.co/videos?keywords={quote_plus(topic)}&page=1"
    data = _request_json(url, headers={"Authorization": f"Bearer {api_key}"})
    videos = []

    for video in _coverr_items(data):
        link = _best_coverr_link(video)
        if link:
            videos.append({"source": "coverr", "url": link})
        if len(videos) >= per_page:
            break

    return videos


def search_stock_videos(topic, pexels_api_key=None, pixabay_api_key=None, coverr_api_key=None, limit=DEFAULT_CLIP_LIMIT):
    queries = search_queries(topic)
    if not queries:
        raise VideoFetchError("Topic required hai.")

    errors = []
    found = []
    target_pool_size = max(limit, min(MAX_SEARCH_RESULTS_TO_COLLECT, limit * 2))
    per_query = 3 if len(queries) > 20 else min(15, max(8, limit // 2))

    if pexels_api_key:
        for query in queries:
            try:
                found.extend(fetch_pexels_videos(query, pexels_api_key, per_page=per_query))
            except Exception as exc:
                errors.append(f"Pexels failed: {exc}")
                break
            if len(found) >= target_pool_size:
                break

    if pixabay_api_key and len(found) < limit:
        for query in queries:
            try:
                found.extend(fetch_pixabay_videos(query, pixabay_api_key, per_page=per_query))
            except Exception as exc:
                errors.append(f"Pixabay failed: {exc}")
                break
            if len(found) >= target_pool_size:
                break

    if coverr_api_key and not found:
        for query in queries:
            try:
                found.extend(fetch_coverr_videos(query, api_key=coverr_api_key, per_page=per_query))
            except Exception as exc:
                errors.append(f"Coverr failed: {exc}")
                break
            if found:
                break

    unique = []
    seen = set()
    for video in sorted(found, key=lambda item: item.get("score", 0), reverse=True):
        url = video["url"]
        if url not in seen:
            seen.add(url)
            unique.append(video)
        if len(unique) >= limit:
            break

    if not unique:
        if not any([pexels_api_key, pixabay_api_key, coverr_api_key]):
            detail = "Stock video API key secrets empty hain. PEXELS_API_KEY ya PIXABAY_API_KEY mein se ek add karo."
        else:
            detail = "Is topic par free stock clips nahi mile. Topic ko short English keywords mein try karo."
        raise VideoFetchError(detail)

    return unique


def looks_like_video_response(response, url):
    content_type = response.headers.get("Content-Type", "").lower()
    suffix = Path(urlparse(url).path).suffix.lower()
    if content_type.startswith("image/"):
        return False
    if content_type.startswith("video/"):
        return True
    return suffix in {".mp4", ".mov", ".mkv", ".webm"}


def validate_video_file(path):
    import subprocess
    import imageio_ffmpeg

    try:
        result = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(path)],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:
        return False

    output = f"{result.stdout}\n{result.stderr}"
    duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", output)
    has_video = "Video:" in output
    if not duration_match or not has_video:
        return False

    hours, minutes, seconds = duration_match.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return duration >= MIN_VALID_CLIP_SECONDS


def download_videos(videos, destination_dir):
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for index, video in enumerate(videos, start=1):
        source = video.get("source", "stock")
        suffix = Path(urlparse(video["url"]).path).suffix.lower()
        if suffix not in {".mp4", ".mov", ".mkv", ".webm"}:
            suffix = ".mp4"
        output_path = destination / f"{source}_{index:03}{suffix}"
        with requests.get(video["url"], stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
            response.raise_for_status()
            if not looks_like_video_response(response, video["url"]):
                continue
            with output_path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)

        if not validate_video_file(output_path):
            output_path.unlink(missing_ok=True)
            continue

        saved_paths.append(output_path)

    return saved_paths


def fetch_and_download_videos(topic, destination_dir, pexels_api_key=None, pixabay_api_key=None, coverr_api_key=None, limit=DEFAULT_CLIP_LIMIT):
    videos = search_stock_videos(
        topic=topic,
        pexels_api_key=pexels_api_key,
        pixabay_api_key=pixabay_api_key,
        coverr_api_key=coverr_api_key,
        limit=limit,
    )
    saved_paths = download_videos(videos, destination_dir)
    if not saved_paths:
        raise VideoFetchError("Real video clips download nahi hue. Image/static files skip kar diye gaye.")
    return saved_paths
