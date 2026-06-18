from pathlib import Path
import hashlib
import json
import re
from urllib.parse import quote_plus, urlparse

import requests


DOWNLOAD_TIMEOUT = 45
SEARCH_TIMEOUT = 20
MIN_RESULTS_BEFORE_FALLBACK = 5
MIN_RELEVANT_RESULTS_BEFORE_PIXABAY = 3
DEFAULT_CLIP_LIMIT = 100
MAX_SEARCH_RESULTS_TO_COLLECT = 220
MAX_QUERY_COUNT = 140
MAX_TOPIC_WORDS = 8
MIN_VALID_CLIP_SECONDS = 3.0
DEFAULT_IMAGE_LIMIT = 70
MIN_IMAGE_WIDTH = 1000
MIN_IMAGE_HEIGHT = 560
CINEMATIC_FALLBACK_QUERIES = [
    "cinematic movie scene dark dramatic",
    "film director movie set cinematic",
    "dramatic lighting close up face",
    "film noir dark atmosphere",
    "mysterious silhouette night",
    "detective investigation night",
    "secret files classified documents",
    "conspiracy board investigation",
    "dark alley night cinematic",
    "neon city night cinematic",
    "rain window night city",
    "old film projector vintage",
    "space moon astronaut cinematic",
    "nasa control room cinematic",
    "storm lightning dramatic sky",
    "fog mysterious forest night",
    "dark ocean waves cinematic",
    "abandoned building interior",
    "smoke fog mysterious",
    "typewriter vintage dramatic",
]
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


class ImageFetchError(RuntimeError):
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

    priority_queries = []
    extra_queries = []

    for trigger, mapped_queries in KEYWORD_MAP.items():
        if trigger in raw_topic:
            extra_queries.extend(mapped_queries)

    for cleaned_topic in cleaned_topics:
        words = cleaned_topic.split()
        priority_queries.append(cleaned_topic)

        if len(words) > 3:
            extra_queries.append(" ".join(words[:3]))
        if len(words) > 1:
            extra_queries.append(" ".join(words[:2]))
        extra_queries.extend(
            [
                f"{cleaned_topic} cinematic video",
                f"{cleaned_topic} documentary",
                f"{cleaned_topic} history",
                f"{cleaned_topic} travel",
                f"{cleaned_topic} monument",
                f"{cleaned_topic} landscape",
            ]
        )

    extra_queries.extend(CINEMATIC_FALLBACK_QUERIES)

    unique = []
    for query in priority_queries + extra_queries:
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


def fetch_pexels_videos(topic, api_key, per_page=DEFAULT_CLIP_LIMIT, query_index=0):
    if not api_key:
        return []

    url = (
        f"https://api.pexels.com/v1/videos/search?query={quote_plus(topic)}"
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
                    "query_index": query_index,
                }
            )

    return sorted(videos, key=lambda item: item.get("score", 0), reverse=True)


def fetch_pexels_popular_videos(api_key, per_page=DEFAULT_CLIP_LIMIT):
    if not api_key:
        return []

    url = (
        "https://api.pexels.com/v1/videos/popular"
        f"?per_page={per_page}&min_width=1280&min_duration={int(MIN_VALID_CLIP_SECONDS)}"
    )
    data = _request_json(url, headers={"Authorization": api_key})
    videos = []

    for video in data.get("videos", []):
        link = _best_pexels_link(video)
        if link:
            videos.append(
                {
                    "source": "pexels_popular",
                    "url": link,
                    "query": "popular cinematic",
                    "score": relevance_score(video, "popular cinematic", "pexels") + 5,
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


def fetch_pixabay_videos(topic, api_key, per_page=DEFAULT_CLIP_LIMIT, query_index=0):
    if not api_key:
        return []

    url = (
        f"https://pixabay.com/api/videos/?key={api_key}&q={quote_plus(topic)}"
        f"&per_page={per_page}&min_width=1280&video_type=film&order=popular&safesearch=true"
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
                    "query_index": query_index,
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
    per_query = 2 if len(queries) > 40 else 3 if len(queries) > 20 else min(15, max(8, limit // 2))

    if pexels_api_key:
        for query_index, query in enumerate(queries):
            try:
                found.extend(fetch_pexels_videos(query, pexels_api_key, per_page=per_query, query_index=query_index))
            except Exception as exc:
                errors.append(f"Pexels failed: {exc}")
                break
            if len(found) >= target_pool_size:
                break
        if len(found) < limit:
            try:
                found.extend(fetch_pexels_popular_videos(pexels_api_key, per_page=min(80, limit)))
            except Exception as exc:
                errors.append(f"Pexels popular failed: {exc}")

    if pixabay_api_key and len(found) < limit:
        for query_index, query in enumerate(queries):
            try:
                found.extend(fetch_pixabay_videos(query, pixabay_api_key, per_page=per_query, query_index=query_index))
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
    for video in sorted(found, key=lambda item: (item.get("query_index", 9999), -item.get("score", 0))):
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


def image_search_queries(topic):
    direct_queries = topic_lines(topic)
    if not direct_queries:
        return []

    queries = []
    for query in direct_queries:
        for candidate in (query, f"{query} cinematic", f"{query} dramatic background"):
            if candidate not in queries:
                queries.append(candidate)
            if len(queries) >= MAX_QUERY_COUNT:
                return queries
    return queries


def _pexels_image_url(photo):
    sources = photo.get("src") or {}
    return sources.get("large2x") or sources.get("landscape") or sources.get("large")


def fetch_pexels_images(query, api_key, per_page=5, query_index=0):
    if not api_key:
        return []

    url = (
        f"https://api.pexels.com/v1/search?query={quote_plus(query)}"
        f"&per_page={per_page}&orientation=landscape"
    )
    data = _request_json(url, headers={"Authorization": api_key})
    results = []
    for photo in data.get("photos", []):
        image_url = _pexels_image_url(photo)
        width = photo.get("width") or 0
        height = photo.get("height") or 0
        if image_url and is_landscape(width, height):
            photographer = photo.get("photographer") or "Pexels contributor"
            results.append(
                {
                    "source": "Pexels",
                    "url": image_url,
                    "page_url": photo.get("url") or image_url,
                    "creator": photographer,
                    "license": "Pexels License",
                    "query": query,
                    "query_index": query_index,
                    "score": 30 + (20 if width >= 1920 else 10 if width >= 1280 else 0),
                }
            )
    return results


def fetch_pixabay_images(query, api_key, per_page=5, query_index=0):
    if not api_key:
        return []

    url = (
        f"https://pixabay.com/api/?key={api_key}&q={quote_plus(query)}"
        f"&image_type=photo&orientation=horizontal&per_page={max(3, per_page)}"
        "&order=popular&safesearch=true&min_width=1280"
    )
    data = _request_json(url)
    results = []
    for image in data.get("hits", []):
        image_url = image.get("fullHDURL") or image.get("largeImageURL") or image.get("webformatURL")
        width = image.get("imageWidth") or image.get("webformatWidth") or 0
        height = image.get("imageHeight") or image.get("webformatHeight") or 0
        if image_url and is_landscape(width, height):
            results.append(
                {
                    "source": "Pixabay",
                    "url": image_url,
                    "page_url": image.get("pageURL") or image_url,
                    "creator": image.get("user") or "Pixabay contributor",
                    "license": "Pixabay Content License",
                    "query": query,
                    "query_index": query_index,
                    "score": 25 + (20 if width >= 1920 else 10),
                }
            )
    return results


def fetch_wikimedia_images(query, per_page=5, query_index=0):
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": 6,
        "gsrlimit": max(3, min(10, per_page)),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|mime|size",
        "iiurlwidth": 1920,
        "format": "json",
        "origin": "*",
    }
    response = requests.get(
        "https://commons.wikimedia.org/w/api.php",
        params=params,
        headers={"User-Agent": "NotebookLMVideoBuilder/1.0"},
        timeout=SEARCH_TIMEOUT,
    )
    response.raise_for_status()
    pages = (response.json().get("query") or {}).get("pages") or {}
    results = []
    for page in pages.values():
        info_items = page.get("imageinfo") or []
        if not info_items:
            continue
        info = info_items[0]
        mime = str(info.get("mime") or "")
        width = info.get("width") or 0
        height = info.get("height") or 0
        image_url = info.get("thumburl") or info.get("url")
        if not image_url or not mime.startswith("image/") or not is_landscape(width, height):
            continue
        metadata = info.get("extmetadata") or {}
        creator = (metadata.get("Artist") or {}).get("value") or "Wikimedia Commons contributor"
        creator = re.sub(r"<[^>]+>", "", creator).strip()
        license_name = (metadata.get("LicenseShortName") or {}).get("value") or "Wikimedia Commons license"
        results.append(
            {
                "source": "Wikimedia Commons",
                "url": image_url,
                "page_url": info.get("descriptionurl") or image_url,
                "creator": creator,
                "license": license_name,
                "query": query,
                "query_index": query_index,
                "score": 20 + (20 if width >= 1920 else 5),
            }
        )
    return results


def fetch_openverse_images(query, per_page=5, query_index=0):
    response = requests.get(
        "https://api.openverse.org/v1/images/",
        params={
            "q": query,
            "page_size": max(3, min(20, per_page)),
            "license_type": "commercial",
            "mature": "false",
        },
        headers={"User-Agent": "NotebookLMVideoBuilder/1.0"},
        timeout=SEARCH_TIMEOUT,
    )
    response.raise_for_status()
    results = []
    for image in response.json().get("results", []):
        image_url = image.get("url") or image.get("thumbnail")
        width = image.get("width") or 0
        height = image.get("height") or 0
        if image_url and (not width or not height or is_landscape(width, height)):
            results.append(
                {
                    "source": "Openverse",
                    "url": image_url,
                    "page_url": image.get("foreign_landing_url") or image_url,
                    "creator": image.get("creator") or "Openverse contributor",
                    "license": str(image.get("license") or "CC").upper(),
                    "query": query,
                    "query_index": query_index,
                    "score": 15 + (20 if width >= 1920 else 5 if width >= 1280 else 0),
                }
            )
    return results


def search_stock_images(topic, pexels_api_key=None, pixabay_api_key=None, limit=DEFAULT_IMAGE_LIMIT):
    queries = image_search_queries(topic)
    if not queries:
        raise ImageFetchError("Topic required hai.")

    found = []
    errors = []
    direct_query_count = max(1, len(topic_lines(topic)))
    per_query = 3 if direct_query_count <= 20 else 2

    for query_index, query in enumerate(queries):
        providers = []
        if pexels_api_key:
            providers.append(("Pexels", fetch_pexels_images, pexels_api_key))
        if pixabay_api_key:
            providers.append(("Pixabay", fetch_pixabay_images, pixabay_api_key))

        for provider_name, provider, api_key in providers:
            try:
                found.extend(provider(query, api_key, per_page=per_query, query_index=query_index))
            except Exception as exc:
                errors.append(f"{provider_name}: {exc}")

        if len(found) < limit:
            try:
                found.extend(fetch_wikimedia_images(query, per_page=per_query, query_index=query_index))
            except Exception as exc:
                errors.append(f"Wikimedia: {exc}")

        if len(found) < limit:
            try:
                found.extend(fetch_openverse_images(query, per_page=per_query, query_index=query_index))
            except Exception as exc:
                errors.append(f"Openverse: {exc}")

        if len(found) >= limit * 2:
            break

    unique = []
    seen_urls = set()
    for image in sorted(found, key=lambda item: (item.get("query_index", 9999), -item.get("score", 0))):
        url = image.get("url")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        unique.append(image)
        if len(unique) >= limit:
            break

    if not unique:
        detail = "Relevant licensed images nahi mili."
        if errors:
            detail += " " + " | ".join(errors[:3])
        raise ImageFetchError(detail)
    return unique


def download_images(images, destination_dir):
    from PIL import Image

    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    credits = []
    seen_hashes = set()

    for index, image in enumerate(images, start=1):
        output_path = destination / f"scene_{index:03}.jpg"
        try:
            with requests.get(
                image["url"],
                stream=True,
                headers={"User-Agent": "NotebookLMVideoBuilder/1.0"},
                timeout=DOWNLOAD_TIMEOUT,
            ) as response:
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").lower()
                if content_type and not content_type.startswith("image/"):
                    continue
                digest = hashlib.sha256()
                with output_path.open("wb") as file:
                    for chunk in response.iter_content(chunk_size=512 * 1024):
                        if chunk:
                            digest.update(chunk)
                            file.write(chunk)

            file_hash = digest.hexdigest()
            if file_hash in seen_hashes:
                output_path.unlink(missing_ok=True)
                continue

            with Image.open(output_path) as loaded:
                loaded.load()
                width, height = loaded.size
                if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT or width < height:
                    output_path.unlink(missing_ok=True)
                    continue
                loaded.convert("RGB").save(output_path, "JPEG", quality=92, optimize=True)

            seen_hashes.add(file_hash)
            saved_paths.append(output_path)
            credits.append(
                {
                    "file": output_path.name,
                    "source": image.get("source", "Unknown"),
                    "creator": image.get("creator", "Unknown"),
                    "license": image.get("license", "Check source"),
                    "source_url": image.get("page_url") or image.get("url"),
                    "query": image.get("query", ""),
                }
            )
        except Exception:
            output_path.unlink(missing_ok=True)

    if credits:
        (destination / "credits.json").write_text(json.dumps(credits, indent=2, ensure_ascii=False), encoding="utf-8")
        lines = ["Cinematic Story image credits", ""]
        for item in credits:
            lines.append(
                f"{item['file']} | {item['source']} | {item['creator']} | "
                f"{item['license']} | {item['source_url']}"
            )
        (destination / "credits.txt").write_text("\n".join(lines), encoding="utf-8")
    return saved_paths


def fetch_and_download_images(topic, destination_dir, pexels_api_key=None, pixabay_api_key=None, limit=DEFAULT_IMAGE_LIMIT):
    images = search_stock_images(
        topic=topic,
        pexels_api_key=pexels_api_key,
        pixabay_api_key=pixabay_api_key,
        limit=limit,
    )
    saved_paths = download_images(images, destination_dir)
    if not saved_paths:
        raise ImageFetchError("Licensed scene images download nahi ho paayi.")
    return saved_paths
