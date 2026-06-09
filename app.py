from pathlib import Path

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
VIDEO_DIR = INPUT_DIR / "videos"
AUDIO_FILE = INPUT_DIR / "audio.mp3"
TRANSCRIPT_FILE = INPUT_DIR / "transcript.txt"
OUTPUT_DIR = BASE_DIR / "output"

FINAL_VIDEO = OUTPUT_DIR / "final_video.mp4"
SUBTITLE_FILE = OUTPUT_DIR / "subtitles.srt"

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


st.set_page_config(
    page_title="NotebookLM Video Builder",
    layout="wide",
)


def save_uploaded_file(uploaded_file, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as file:
        file.write(uploaded_file.getbuffer())


def ensure_folders():
    INPUT_DIR.mkdir(exist_ok=True)
    VIDEO_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)


def validate_inputs():
    if not AUDIO_FILE.exists():
        raise FileNotFoundError("NotebookLM audio ko input/audio.mp3 naam se rakho.")


def create_video():
    from moviepy.editor import AudioFileClip
    from audio_utils import get_target_duration
    from subtitle_utils import add_subtitles, load_or_create_subtitle_segments, write_srt
    from video_utils import build_synced_video, collect_video_files

    ensure_folders()
    validate_inputs()

    target_duration = get_target_duration(AUDIO_FILE, TARGET_SECONDS)
    video_paths = collect_video_files(VIDEO_DIR)

    base_video = build_synced_video(
        video_paths=video_paths,
        target_duration=target_duration,
        size=VIDEO_SIZE,
        fps=FPS,
        crossfade_seconds=CROSSFADE_SECONDS,
    )

    audio = AudioFileClip(str(AUDIO_FILE)).subclip(0, target_duration)
    synced = base_video.set_audio(audio)

    subtitle_segments = load_or_create_subtitle_segments(
        transcript_path=TRANSCRIPT_FILE,
        audio_path=AUDIO_FILE,
        target_duration=target_duration,
        max_chars=SUBTITLE_MAX_CHARS,
    )
    write_srt(subtitle_segments, SUBTITLE_FILE)

    final = add_subtitles(
        video_clip=synced,
        segments=subtitle_segments,
        font=FONT,
        font_path=FONT_PATH,
        font_size=FONT_SIZE,
        color=SUBTITLE_COLOR,
        stroke_color=SUBTITLE_STROKE_COLOR,
        stroke_width=SUBTITLE_STROKE_WIDTH,
        max_chars=SUBTITLE_MAX_CHARS,
    )

    final.write_videofile(
        str(FINAL_VIDEO),
        codec="libx264",
        audio_codec="aac",
        fps=FPS,
        preset="medium",
        threads=4,
    )

    final.close()
    synced.close()
    base_video.close()
    audio.close()


def clear_old_videos():
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    for path in VIDEO_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            path.unlink()


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


def get_secret(name):
    try:
        value = st.secrets.get(name, "")
    except Exception:
        return ""
    return str(value).strip()


def show_existing_output():
    if not FINAL_VIDEO.exists():
        return

    try:
        st.success("Final video ready hai.")
        with FINAL_VIDEO.open("rb") as file:
            st.download_button(
                "Download final_video.mp4",
                data=file,
                file_name="final_video.mp4",
                mime="video/mp4",
                use_container_width=True,
            )
    except Exception:
        st.warning("Previous output file load nahi ho paayi. Naya video generate karo.")


def render_app():
    ensure_folders()

    st.title("NotebookLM Video Builder")
    st.caption("Topic aur MP3 audio do. App relevant video clips automatically fetch karke synced video generate karega.")

    left, right = st.columns([0.52, 0.48], gap="large")

    with left:
        topic = st.text_input(
            "Topic",
            placeholder="Kohinoor diamond history",
        )
        video_source = st.radio(
            "Video source",
            ["Auto fetch stock videos", "Manual upload"],
            horizontal=True,
        )
        audio_upload = st.file_uploader(
            "NotebookLM Audio (MP3)",
            type=["mp3"],
            accept_multiple_files=False,
        )
        video_uploads = None
        if video_source == "Manual upload":
            video_uploads = st.file_uploader(
                "Pexels Videos (MP4 files)",
                type=["mp4", "mov", "mkv", "webm"],
                accept_multiple_files=True,
            )
        else:
            st.info("Auto mode free stock-video sources se clips fetch karega: Pexels, Pixabay, aur Coverr.")
        transcript_text = st.text_area(
            "Hindi Transcript / Subtitles (optional)",
            height=180,
            placeholder="Yahan Hindi transcript paste karo.",
        )

        generate = st.button("Generate Professional Video", type="primary", use_container_width=True)

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

    if video_source == "Manual upload" and not video_uploads:
        st.error("Manual mode mein kam se kam 1 video file upload karo.")
        return

    search_topic = build_auto_topic(topic, transcript_text)

    if video_source == "Auto fetch stock videos" and not search_topic:
        st.error("Auto mode ke liye topic enter karo ya transcript paste karo.")
        return

    save_uploaded_file(audio_upload, AUDIO_FILE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if transcript_text.strip():
        TRANSCRIPT_FILE.write_text(transcript_text.strip(), encoding="utf-8")
    elif TRANSCRIPT_FILE.exists():
        TRANSCRIPT_FILE.write_text("", encoding="utf-8")

    with output_placeholder.container():
        progress = st.progress(0, text="Video build ho raha hai...")

        try:
            if video_source == "Manual upload":
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
                    limit=8,
                )
                st.info(f"{len(saved_videos)} stock video clips download ho gaye.")

            progress.progress(35, text="Audio duration aur clips prepare ho rahe hain...")
            create_video()
            st.session_state.video_generated = True
            progress.progress(100, text="Done")
        except Exception as exc:
            if exc.__class__.__name__ == "VideoFetchError":
                st.error(str(exc))
                return
            st.exception(exc)
            return

    show_existing_output()


if __name__ == "__main__":
    render_app()
