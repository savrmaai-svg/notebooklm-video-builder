from pathlib import Path
import traceback

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
VIDEO_DIR = INPUT_DIR / "videos"
IMAGE_DIR = INPUT_DIR / "images"
AUDIO_FILE = INPUT_DIR / "audio.m4a"
AUDIO_SOURCE_FILE = INPUT_DIR / "audio_source"
TRANSCRIPT_FILE = INPUT_DIR / "transcript.txt"
OUTPUT_DIR = BASE_DIR / "output"

FINAL_VIDEO = OUTPUT_DIR / "final_video.mp4"
YOUTUBE_SHORTS = OUTPUT_DIR / "youtube_shorts.mp4"
SUBTITLE_FILE = OUTPUT_DIR / "subtitles.srt"
CREDITS_FILE = IMAGE_DIR / "credits.txt"

TARGET_SECONDS = 10 * 60
VIDEO_SIZE = (1920, 1080)
FPS = 30
CROSSFADE_SECONDS = 0.8
SUBTITLE_MAX_CHARS = 46

FONT_SIZE = 58
FONT = "Arial-Bold"
FONT_PATH = r"C:\Windows\Fonts\Nirmala.ttf"
SUBTITLE_COLOR = "white"
SUBTITLE_STROKE_COLOR = "black"
SUBTITLE_STROKE_WIDTH = 3
SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}
SUPPORTED_AUDIO_UPLOADS = ["mp3", "m4a", "aac", "wav", "ogg", "flac", "mp4", "mov", "mkv", "webm"]
AUTO_CLIP_LIMIT = 100
MAX_CUT_SECONDS = 5
MAX_ADAPTIVE_CUT_SECONDS = 16
MAX_IMAGE_SCENE_SECONDS = 12
IMAGE_SCENE_SECONDS = 7
CREATION_MODES = [
    "Stock Video Documentary",
    "Cinematic Image Story",
    "2D Cartoon Episode (Lip Sync)",
    "Storyboard Grid → Video",
    "Cinematic Slow-Mo (clip → long)",
    "Concat + Voiceover (clips + narration)",
    "Short / Reel (30s + Subscribe CTA)",
    "Faceless Video (script → narrated video)",
]
OUTPUT_DURATION_OPTIONS = {
    "1-2 minutes demo": 120,
    "2-3 minutes": 180,
    "4-5 minutes": 300,
    "5-6 minutes": 360,
}


st.set_page_config(
    page_title="NotebookLM Video Builder",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: radial-gradient(1100px 560px at 18% -12%, #1b2440 0%, #0e1117 58%); }
    .block-container { padding-top: 2.0rem; max-width: 1180px; }
    h1 {
        font-weight: 800; letter-spacing: -0.6px; font-size: 2.5rem;
        background: linear-gradient(90deg, #a78bfa 0%, #38bdf8 60%, #22d3ee 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    h2, h3 { font-weight: 700; }
    div[data-testid="stRadio"] > div { gap: 6px; flex-wrap: wrap; }
    div[data-testid="stRadio"] label {
        background: #171c2e; border: 1px solid #2a3350; border-radius: 12px;
        padding: 7px 14px; transition: all .15s ease;
    }
    div[data-testid="stRadio"] label:hover { border-color: #6d5cff; }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #7c5cff 0%, #22d3ee 100%);
        border: none; border-radius: 12px; font-weight: 700; padding: 0.72rem;
        box-shadow: 0 6px 22px rgba(124, 92, 255, .35);
    }
    div.stButton > button[kind="primary"]:hover { filter: brightness(1.08); transform: translateY(-1px); }
    .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
        border-radius: 10px; border: 1px solid #2a3350; background: #131829;
    }
    [data-testid="stFileUploader"] section {
        border: 1.5px dashed #3a4570; border-radius: 12px; background: #141a2c;
    }
    .nlm-badge {
        display:inline-block; padding:3px 12px; border-radius:999px; font-size:0.8rem; font-weight:600;
        background:linear-gradient(90deg,#7c5cff33,#22d3ee33); border:1px solid #3a4570; color:#cbd5ff;
    }
    div[data-testid="stFileUploader"] div[data-testid="stFileUploaderFile"] { width: 100%; max-width: 100%; margin-bottom: 0.55rem; }
    div[data-testid="stFileUploader"] div[data-testid="stFileUploaderFile"] > div { width: 100%; max-width: 100%; }
    div[data-testid="stFileUploader"] div[data-testid="stFileUploaderFile"] div { white-space: normal; overflow-wrap: anywhere; }
    </style>
    """,
    unsafe_allow_html=True,
)


def save_uploaded_file(uploaded_file, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as file:
        file.write(uploaded_file.getbuffer())


def ensure_folders():
    INPUT_DIR.mkdir(exist_ok=True)
    VIDEO_DIR.mkdir(exist_ok=True)
    IMAGE_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)


def validate_inputs():
    if not AUDIO_FILE.exists():
        raise FileNotFoundError("NotebookLM audio upload karo. App usme se sirf audio track use karega.")


def media_duration(ffmpeg, path, timeout=30):
    import re
    import subprocess

    result = subprocess.run(
        [ffmpeg, "-i", str(path)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = f"{result.stdout}\n{result.stderr}"
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", output)
    if not match:
        return 0.0
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def create_video(target_seconds):
    import math
    import subprocess
    import imageio_ffmpeg

    ensure_folders()
    validate_inputs()

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    video_paths = sorted(
        path for path in VIDEO_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not video_paths:
        raise FileNotFoundError("input/videos folder mein MP4 videos nahi mile.")

    def srt_time(seconds):
        seconds = max(0, float(seconds))
        millis = int(round((seconds - math.floor(seconds)) * 1000))
        whole = int(seconds)
        hours = whole // 3600
        minutes = (whole % 3600) // 60
        secs = whole % 60
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    video_durations = []
    for path in video_paths:
        duration = media_duration(ffmpeg, path) or MAX_CUT_SECONDS
        if duration > 0:
            video_durations.append((path, duration))

    requested_duration = min(float(target_seconds), TARGET_SECONDS)
    adaptive_cut_seconds = min(
        MAX_ADAPTIVE_CUT_SECONDS,
        max(MAX_CUT_SECONDS, requested_duration / max(1, len(video_durations))),
    )
    available_unique_duration = sum(min(duration, adaptive_cut_seconds) for _, duration in video_durations)
    target_duration = min(requested_duration, available_unique_duration)
    if target_duration <= 0:
        raise RuntimeError("Audio/video duration read nahi ho paayi.")
    duration_was_shortened = target_duration < requested_duration - 1

    transcript = ""
    if TRANSCRIPT_FILE.exists():
        transcript = TRANSCRIPT_FILE.read_text(encoding="utf-8").strip()
    if transcript:
        words = transcript.split()
        chunk_count = max(1, min(16, math.ceil(len(words) / 9)))
        chunk_size = max(1, math.ceil(len(words) / chunk_count))
        segment_duration = target_duration / chunk_count
        lines = []
        for index in range(chunk_count):
            text = " ".join(words[index * chunk_size:(index + 1) * chunk_size])
            if not text:
                continue
            start = index * segment_duration
            end = min(target_duration, (index + 1) * segment_duration)
            lines.extend([str(index + 1), f"{srt_time(start)} --> {srt_time(end)}", text, ""])
        SUBTITLE_FILE.write_text("\n".join(lines), encoding="utf-8")

    temp_dir = OUTPUT_DIR / "segments"
    temp_dir.mkdir(parents=True, exist_ok=True)
    for old_file in temp_dir.glob("*.mp4"):
        old_file.unlink()

    segments = []
    remaining = target_duration
    for index, (source, source_duration) in enumerate(video_durations):
        if remaining <= 0.2:
            break
        segment_seconds = min(source_duration, remaining, adaptive_cut_seconds)
        output_segment = temp_dir / f"segment_{index:03}.mp4"
        max_offset = max(0, source_duration - segment_seconds - 0.1)
        start_offset = (index * 3.7) % max_offset if max_offset > 0 else 0
        vf = "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,fps=24"
        command = [
            ffmpeg,
            "-y",
        ]
        if start_offset > 0:
            command.extend(["-ss", f"{start_offset:.2f}"])
        command.extend(
            [
                "-i",
                str(source),
                "-t",
                f"{segment_seconds:.2f}",
                "-vf",
                vf,
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                str(output_segment),
            ]
        )
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        segments.append(output_segment)
        remaining -= segment_seconds

    if not segments:
        raise RuntimeError("Unique video clips render ke liye available nahi hain.")

    concat_file = temp_dir / "concat.txt"
    concat_file.write_text(
        "\n".join(f"file '{segment.as_posix()}'" for segment in segments),
        encoding="utf-8",
    )

    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-i",
            str(AUDIO_FILE),
            "-t",
            f"{target_duration:.2f}",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(FINAL_VIDEO),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=420,
    )
    return target_duration, duration_was_shortened


def prepare_cartoon_scene(source, destination):
    from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps

    with Image.open(source) as loaded:
        canvas = ImageOps.fit(
            loaded.convert("RGB"),
            (1280, 720),
            method=Image.Resampling.LANCZOS,
        )

    softened = canvas.filter(ImageFilter.MedianFilter(size=5))
    posterized = ImageOps.posterize(softened, 5)
    posterized = ImageEnhance.Color(posterized).enhance(1.18)
    posterized = ImageEnhance.Contrast(posterized).enhance(1.08)

    edges = ImageOps.grayscale(canvas).filter(ImageFilter.FIND_EDGES)
    edges = edges.filter(ImageFilter.GaussianBlur(radius=0.7))
    ink = ImageOps.invert(edges).point(lambda value: 255 if value > 205 else 105)
    ink_rgb = Image.merge("RGB", (ink, ink, ink))
    cartoon = ImageChops.multiply(posterized, ink_rgb)
    cartoon.save(destination, "JPEG", quality=94, optimize=True)


def create_cinematic_story(target_seconds):
    import subprocess

    import imageio_ffmpeg

    ensure_folders()
    validate_inputs()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    image_paths = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not image_paths:
        raise FileNotFoundError("Cinematic story ke liye scene images nahi mili.")

    requested_duration = min(float(target_seconds), TARGET_SECONDS)
    audio_duration = media_duration(ffmpeg, AUDIO_FILE)
    if audio_duration > 0:
        requested_duration = min(requested_duration, audio_duration)

    available_duration = len(image_paths) * MAX_IMAGE_SCENE_SECONDS
    target_duration = min(requested_duration, available_duration)
    if target_duration <= 0:
        raise RuntimeError("Cinematic story duration prepare nahi ho paayi.")
    duration_was_shortened = target_duration < requested_duration - 1
    scene_seconds = min(MAX_IMAGE_SCENE_SECONDS, target_duration / len(image_paths))

    temp_dir = OUTPUT_DIR / "cartoon_segments"
    prepared_dir = temp_dir / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    for old_file in temp_dir.glob("*.mp4"):
        old_file.unlink()
    for old_file in prepared_dir.glob("*.jpg"):
        old_file.unlink()

    segments = []
    remaining = target_duration
    for index, source in enumerate(image_paths):
        if remaining <= 0.1:
            break
        duration = min(scene_seconds, remaining)
        prepared = prepared_dir / f"scene_{index:03}.jpg"
        prepare_cartoon_scene(source, prepared)

        frame_count = max(1, int(round(duration * 24)))
        motion = index % 4
        if motion == 0:
            zoom = "min(zoom+0.0010,1.14)"
            x = "iw/2-(iw/zoom/2)"
            y = "ih/2-(ih/zoom/2)"
        elif motion == 1:
            zoom = "1.10"
            x = f"(iw-iw/zoom)*on/{max(1, frame_count - 1)}"
            y = "ih/2-(ih/zoom/2)"
        elif motion == 2:
            zoom = "1.10"
            x = f"(iw-iw/zoom)*(1-on/{max(1, frame_count - 1)})"
            y = "ih/2-(ih/zoom/2)"
        else:
            zoom = "min(zoom+0.0008,1.12)"
            x = "iw/2-(iw/zoom/2)"
            y = f"(ih-ih/zoom)*on/{max(1, frame_count - 1)}"

        vf = (
            "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,"
            f"zoompan=z='{zoom}':x='{x}':y='{y}':d={frame_count}:s=1280x720:fps=24,"
            "eq=contrast=1.06:saturation=1.10:brightness=-0.01,vignette=PI/5,format=yuv420p"
        )
        output_segment = temp_dir / f"scene_{index:03}.mp4"
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loop",
                "1",
                "-i",
                str(prepared),
                "-t",
                f"{duration:.2f}",
                "-vf",
                vf,
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                str(output_segment),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        segments.append(output_segment)
        remaining -= duration

    if not segments:
        raise RuntimeError("Cinematic scenes render nahi ho paaye.")

    concat_file = temp_dir / "concat.txt"
    concat_file.write_text(
        "\n".join(f"file '{segment.as_posix()}'" for segment in segments),
        encoding="utf-8",
    )
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-i",
            str(AUDIO_FILE),
            "-t",
            f"{target_duration:.2f}",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(FINAL_VIDEO),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=420,
    )
    return target_duration, duration_was_shortened


def create_cartoon_episode(target_seconds, topic="", transcript="", progress_cb=None):
    # New self-contained local pipeline: transcribe -> scene-aware backgrounds + recurring host
    # narrators with face-warp viseme lip-sync + scene ambient sound. Generates its own art
    # (no stock photos). Requires the local cartoon dependencies (see requirements-cartoon-local.txt).
    import imageio_ffmpeg
    import importlib, cartoon_local
    cartoon_local = importlib.reload(cartoon_local)   # always pick up the latest cartoon_local.py

    ensure_folders()
    validate_inputs()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    requested_duration = min(float(target_seconds), TARGET_SECONDS)
    audio_duration = media_duration(ffmpeg, AUDIO_FILE)
    if audio_duration > 0:
        requested_duration = min(requested_duration, audio_duration)
    if requested_duration <= 0:
        raise RuntimeError("Cartoon episode duration prepare nahi ho paayi.")

    final_duration = cartoon_local.render(
        audio_path=AUDIO_FILE,
        output_path=FINAL_VIDEO,
        target_seconds=requested_duration,
        topic=topic,
        transcript=transcript,
        progress=progress_cb,
    )
    return final_duration, final_duration < requested_duration - 1


def extract_youtube_shorts():
    import subprocess
    import imageio_ffmpeg

    if not FINAL_VIDEO.exists():
        raise FileNotFoundError("Final video nahi mila, Shorts extract nahi ho sakta.")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    full_duration = media_duration(ffmpeg, FINAL_VIDEO)
    if full_duration <= 0:
        raise RuntimeError("Final video duration read nahi ho paayi.")

    shorts_duration = min(55, full_duration)
    start_time = max(0, full_duration * 0.30 - shorts_duration / 2)
    if start_time + shorts_duration > full_duration:
        start_time = max(0, full_duration - shorts_duration)

    vf = "crop=ih*9/16:ih,scale=1080:1920"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-ss",
            f"{start_time:.2f}",
            "-i",
            str(FINAL_VIDEO),
            "-t",
            f"{shorts_duration:.2f}",
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(YOUTUBE_SHORTS),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )


def robust_unlink(path, retries=6, delay=0.4):
    """Delete a file even if Windows has it transiently locked.

    Retries a few times (handles antivirus scans / a subprocess handle that is
    about to close), then as a last resort renames the file out of the way so a
    fresh file can be written to the original name. Raises only if even the
    rename fails (file is held with an exclusive, non-shareable lock).
    """
    import os
    import time
    import uuid

    for attempt in range(retries):
        try:
            path.unlink()
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt < retries - 1:
                time.sleep(delay)
                continue
            # Last resort: move it aside so the name is free for the new file.
            stale = path.with_name(f"_stale_{uuid.uuid4().hex[:8]}_{path.name}")
            try:
                os.replace(path, stale)
                return
            except OSError:
                raise PermissionError(
                    f"'{path.name}' abhi bhi locked hai. Koi render/transcription chal "
                    f"raha ho to use khatam hone do, ya app restart karke dobara upload karo."
                )


def clear_old_videos():
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    for path in VIDEO_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            robust_unlink(path)


def clear_old_images():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for path in IMAGE_DIR.iterdir():
        if path.is_file():
            robust_unlink(path)


def clear_old_audio():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Best-effort sweep of files parked aside by a previous locked delete.
    for stale in INPUT_DIR.glob("_stale_*"):
        try:
            stale.unlink()
        except OSError:
            pass  # still locked; leave it, try again next run
    for path in INPUT_DIR.glob("audio*"):
        if path.is_file():
            robust_unlink(path)


def save_and_extract_audio(uploaded_file):
    import subprocess
    import imageio_ffmpeg

    clear_old_audio()
    suffix = Path(uploaded_file.name).suffix.lower() or ".mp3"
    source_path = AUDIO_SOURCE_FILE.with_suffix(suffix)
    save_uploaded_file(uploaded_file, source_path)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(source_path),
            "-vn",
            "-acodec",
            "aac",
            "-b:a",
            "160k",
            str(AUDIO_FILE),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if media_duration(ffmpeg, AUDIO_FILE) <= 0:
        raise RuntimeError("Uploaded file se audio track extract nahi ho paaya.")
    return AUDIO_FILE


def save_video_uploads(uploaded_videos):
    clear_old_videos()
    saved_paths = []

    for index, uploaded_video in enumerate(uploaded_videos, start=1):
        suffix = Path(uploaded_video.name).suffix.lower() or ".mp4"
        safe_path = VIDEO_DIR / f"pexels_{index:03}{suffix}"
        save_uploaded_file(uploaded_video, safe_path)
        saved_paths.append(safe_path)

    return saved_paths


def build_auto_topic(topic, transcript_text):
    if topic.strip():
        return topic.strip()

    clean_transcript = " ".join(transcript_text.strip().replace('"', "").split())
    if clean_transcript:
        return " ".join(clean_transcript.split()[:10])

    return ""


def count_topic_lines(topic):
    lines = [line.strip() for line in topic.splitlines() if line.strip()]
    return len(lines) if lines else 1


def get_secret(name):
    try:
        value = st.secrets.get(name, "")
    except Exception:
        return ""
    return str(value).strip()


def show_existing_output():
    if not FINAL_VIDEO.exists() and not YOUTUBE_SHORTS.exists():
        return

    try:
        st.success("Final video ready hai.")
        if FINAL_VIDEO.exists():
            with FINAL_VIDEO.open("rb") as file:
                st.download_button(
                    "Download Full Video (10 min)",
                    data=file,
                    file_name="final_video.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )
        if YOUTUBE_SHORTS.exists():
            with YOUTUBE_SHORTS.open("rb") as file:
                st.download_button(
                    "Download YouTube Shorts (55 sec)",
                    data=file,
                    file_name="youtube_shorts.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )
        if CREDITS_FILE.exists():
            with CREDITS_FILE.open("rb") as file:
                st.download_button(
                    "Download Image Credits",
                    data=file,
                    file_name="image_credits.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
    except Exception:
        st.warning("Previous output file load nahi ho paayi. Naya video generate karo.")


def render_app():
    ensure_folders()

    st.title("🎬 NotebookLM Video Builder")
    st.markdown(
        '<span class="nlm-badge">Stock Documentary · Cinematic Story · 2D Cartoon Lip-Sync</span>',
        unsafe_allow_html=True,
    )
    st.caption("Audio upload karo — app use samajh kar matching visuals, Hindi captions aur scene-aware sound ke saath video bana deta hai.")

    left, right = st.columns([0.52, 0.48], gap="large")

    with left:
        creation_mode = st.radio(
            "Creation mode",
            CREATION_MODES,
            horizontal=True,
        )
        if creation_mode == "Storyboard Grid → Video":
            try:
                import importlib, storyboard_studio as _sbs
                importlib.reload(_sbs)
                _sbs.render_mode()
            except Exception as _e:
                st.error("Storyboard mode error: " + repr(_e))
                st.exception(_e)
            st.stop()
        if creation_mode == "Cinematic Slow-Mo (clip → long)":
            try:
                import importlib, cinematic_studio as _cine
                importlib.reload(_cine)
                _cine.render_mode()
            except Exception as _e:
                st.error("Cinematic mode error: " + repr(_e))
                st.exception(_e)
            st.stop()
        if creation_mode == "Concat + Voiceover (clips + narration)":
            try:
                import importlib, concat_studio as _cc
                importlib.reload(_cc)
                _cc.render_mode()
            except Exception as _e:
                st.error("Concat mode error: " + repr(_e))
                st.exception(_e)
            st.stop()
        if creation_mode == "Short / Reel (30s + Subscribe CTA)":
            try:
                import importlib, shorts_studio as _sh
                importlib.reload(_sh)
                _sh.render_mode()
            except Exception as _e:
                st.error("Shorts mode error: " + repr(_e))
                st.exception(_e)
            st.stop()
        if creation_mode == "Faceless Video (script → narrated video)":
            try:
                import importlib, faceless_studio as _fl
                importlib.reload(_fl)
                _fl.render_mode()
            except Exception as _e:
                st.error("Faceless mode error: " + repr(_e))
                st.exception(_e)
            st.stop()
        topic = st.text_area(
            "Topic",
            height=96,
            placeholder=(
                "1. film director cinematic dark\n"
                "2. movie set dramatic lighting\n"
                "3. camera cinematography closeup\n"
                "4. hollywood studio"
            ),
        )
        video_source = "Auto fetch stock videos"
        if creation_mode == "Stock Video Documentary":
            video_source = st.radio(
                "Video source",
                ["Auto fetch stock videos", "Manual upload"],
                horizontal=True,
            )
        duration_label = st.selectbox(
            "Output duration",
            list(OUTPUT_DURATION_OPTIONS.keys()),
            index=1,
            help="Video length select karo. Audio speed change nahi hogi.",
        )
        selected_duration = OUTPUT_DURATION_OPTIONS[duration_label]
        min_estimated_clips = max(1, int((selected_duration + MAX_ADAPTIVE_CUT_SECONDS - 1) // MAX_ADAPTIVE_CUT_SECONDS))
        ideal_estimated_clips = max(1, int((selected_duration + MAX_CUT_SECONDS - 1) // MAX_CUT_SECONDS))
        ideal_estimated_images = max(1, int((selected_duration + IMAGE_SCENE_SECONDS - 1) // IMAGE_SCENE_SECONDS))
        if creation_mode == "2D Cartoon Episode (Lip Sync)":
            st.caption(
                f"{duration_label} tak ka cartoon episode banega. Har scene ka background, character, "
                "lip-sync aur ambient sound app khud generate karega. Audio original speed par rahega."
            )
        elif creation_mode == "Cinematic Image Story":
            st.caption(
                f"{duration_label} output ke liye approx {ideal_estimated_images} scene images fetch hongi. "
                "Background story ke saath change hoga aur audio original speed par rahega."
            )
        else:
            st.caption(
                f"{duration_label} output ke liye approx {min_estimated_clips}-{ideal_estimated_clips} unique clips lagenge. "
                "Har line alag search query ki tarah use hogi. Audio original speed par rahega."
            )
        audio_upload = st.file_uploader(
            "NotebookLM Audio / Video Audio",
            type=SUPPORTED_AUDIO_UPLOADS,
            accept_multiple_files=False,
        )
        video_uploads = None
        if creation_mode == "Stock Video Documentary" and video_source == "Manual upload":
            video_uploads = st.file_uploader(
                "Pexels Videos (MP4 files)",
                type=["mp4", "mov", "mkv", "webm"],
                accept_multiple_files=True,
            )
        elif creation_mode == "Stock Video Documentary":
            st.info("Auto mode free stock-video sources se clips fetch karega: Pexels, Pixabay, aur Coverr.")
            st.caption("App zyada clips download karega aur ek clip ko same video mein repeat nahi karega.")
            st.markdown(
                "[Pexels](https://www.pexels.com/videos/) | "
                "[Pixabay](https://pixabay.com/videos/) | "
                "[Coverr](https://coverr.co/) free stock-video sources use honge."
            )
        elif creation_mode == "Cinematic Image Story":
            st.info(
                "Cinematic Image Story mode transcript/topic ke scene ke hisaab se licensed images fetch karke "
                "cartoon color treatment aur changing camera motion add karega."
            )
            st.markdown(
                "[Pexels](https://www.pexels.com/) | "
                "[Pixabay](https://pixabay.com/) | "
                "[Wikimedia Commons](https://commons.wikimedia.org/) | "
                "[Openverse](https://openverse.org/)"
            )
        else:
            st.info(
                "2D Cartoon Episode mode: app audio ko transcribe karke har scene ka background khud generate karta hai "
                "(story ke hisaab se — gaon, market, ghar, workshop...), recurring cartoon host narrator ka face-warp lip-sync "
                "karta hai, aur har scene ka apna ambient sound + mood music lagata hai."
            )
            st.caption(
                "⚙️ Local GPU mode — ye features aapke PC par chalte hain (public cloud par nahi). "
                "Pehli baar whisper model download hoga."
            )
        transcript_text = st.text_area(
            "Story Script / Hindi Text (optional — drives the visuals)",
            height=180,
            placeholder=("Yahan apni poori kahani ka TEXT paste karo. Agar diya, to har scene ka "
                         "background / character / weather / location ISI text se banega (saaf samajh — "
                         "ASR galtiyan nahi). Audio phir bhi voice + timing + lip-sync + captions deta hai. "
                         "Khaali chhodo to app khud audio transcribe karke samajhta hai."),
        )

        if creation_mode == "2D Cartoon Episode (Lip Sync)":
            button_label = "Generate 2D Cartoon Episode"
        elif creation_mode == "Cinematic Image Story":
            button_label = "Generate Cinematic Image Story"
        else:
            button_label = "Generate Professional Video"
        generate = st.button(button_label, type="primary", use_container_width=True)

    with right:
        st.subheader("Output")
        output_placeholder = st.empty()
        if st.session_state.get("video_generated"):
            show_existing_output()

    if not generate:
        return

    if audio_upload is None:
        st.error("Pehle NotebookLM audio upload karo.")
        return

    if creation_mode == "Stock Video Documentary" and video_source == "Manual upload" and not video_uploads:
        st.error("Manual mode mein kam se kam 1 video file upload karo.")
        return

    search_topic = build_auto_topic(topic, transcript_text)
    topic_line_count = count_topic_lines(search_topic)

    if creation_mode == "Stock Video Documentary" and video_source == "Auto fetch stock videos" and not search_topic:
        st.error("Auto mode ke liye topic enter karo ya transcript paste karo.")
        return

    try:
        save_and_extract_audio(audio_upload)
    except Exception as exc:
        st.error("Audio extract nahi ho paaya. MP3, M4A, WAV, MP4, MOV ya WEBM file try karo.")
        st.exception(exc)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if transcript_text.strip():
        TRANSCRIPT_FILE.write_text(transcript_text.strip(), encoding="utf-8")
    elif TRANSCRIPT_FILE.exists():
        TRANSCRIPT_FILE.write_text("", encoding="utf-8")

    with output_placeholder.container():
        progress = st.progress(0, text="Video build ho raha hai...")

        try:
            if creation_mode == "2D Cartoon Episode (Lip Sync)":
                def _cartoon_progress(pct, text):
                    progress.progress(min(99, max(1, int(pct))), text=text)
                final_duration, duration_was_shortened = create_cartoon_episode(
                    selected_duration,
                    topic=search_topic,
                    transcript=transcript_text,
                    progress_cb=_cartoon_progress,
                )
            elif creation_mode == "Cinematic Image Story":
                import importlib
                import api_fetcher

                api_fetcher = importlib.reload(api_fetcher)

                clear_old_images()
                progress.progress(10, text="Licensed cinematic scene images search ho rahi hain...")
                saved_images = api_fetcher.fetch_and_download_images(
                    topic=search_topic,
                    destination_dir=IMAGE_DIR,
                    pexels_api_key=get_secret("PEXELS_API_KEY"),
                    pixabay_api_key=get_secret("PIXABAY_API_KEY"),
                    limit=min(90, max(topic_line_count, ideal_estimated_images) + 10),
                )
                st.info(f"{len(saved_images)} licensed scene images download ho gayi.")
                progress.progress(38, text="Image treatment aur cinematic camera motion render ho raha hai...")
                final_duration, duration_was_shortened = create_cinematic_story(selected_duration)
            elif video_source == "Manual upload":
                saved_videos = save_video_uploads(video_uploads)
                st.info(f"{len(saved_videos)} manual video files save ho gaye.")
            else:
                from api_fetcher import VideoFetchError, fetch_and_download_videos

                clear_old_videos()
                progress.progress(10, text="Stock videos search aur download ho rahe hain...")
                saved_videos = fetch_and_download_videos(
                    topic=search_topic,
                    destination_dir=VIDEO_DIR,
                    pexels_api_key=get_secret("PEXELS_API_KEY"),
                    pixabay_api_key=get_secret("PIXABAY_API_KEY"),
                    coverr_api_key=get_secret("COVERR_API_KEY"),
                    limit=max(AUTO_CLIP_LIMIT, min(140, max(topic_line_count, ideal_estimated_clips) + 20)),
                )
                st.info(f"{len(saved_videos)} stock video clips download ho gaye.")

            if creation_mode == "Stock Video Documentary":
                progress.progress(35, text="Audio duration aur clips prepare ho rahe hain...")
                final_duration, duration_was_shortened = create_video(selected_duration)
            if duration_was_shortened:
                st.warning(
                    "Selected duration se kam video bana kyunki unique usable scenes kam mile ya audio chhota tha. "
                    "App ne repeat kiye bina maximum possible duration banaya."
                )
            st.info(
                f"Output duration: {final_duration / 60:.1f} minutes. "
                "Audio speed change nahi hui."
            )
            progress.progress(82, text="YouTube Shorts clip extract ho raha hai...")
            extract_youtube_shorts()
            st.session_state.video_generated = True
            progress.progress(100, text="Done")
        except Exception as exc:
            if exc.__class__.__name__ in {"VideoFetchError", "ImageFetchError"}:
                st.error(str(exc))
                return
            st.exception(exc)
            return

    show_existing_output()


try:
    render_app()
except BaseException:
    st.title("NotebookLM Video Builder")
    st.error("App startup error mila. Neeche exact error hai.")
    st.code(traceback.format_exc(), language="python")
