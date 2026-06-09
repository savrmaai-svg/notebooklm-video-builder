from pathlib import Path

import streamlit as st

from api_fetcher import VideoFetchError, fetch_and_download_videos
from config import AUDIO_FILE, FINAL_VIDEO, INPUT_DIR, OUTPUT_DIR, TRANSCRIPT_FILE, VIDEO_DIR
from main import create_video, ensure_folders
from video_utils import SUPPORTED_EXTENSIONS


st.set_page_config(
    page_title="NotebookLM Video Builder",
    layout="wide",
)


def save_uploaded_file(uploaded_file, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as file:
        file.write(uploaded_file.getbuffer())


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


def render_api_key_sidebar():
    st.sidebar.header("API Keys")
    st.session_state.setdefault("pexels_api_key", "")
    st.session_state.setdefault("pixabay_api_key", "")
    st.session_state.setdefault("coverr_api_key", "")

    st.session_state.pexels_api_key = st.sidebar.text_input(
        "Pexels API Key",
        value=st.session_state.pexels_api_key,
        type="password",
    )
    st.session_state.pixabay_api_key = st.sidebar.text_input(
        "Pixabay API Key",
        value=st.session_state.pixabay_api_key,
        type="password",
    )
    st.session_state.coverr_api_key = st.sidebar.text_input(
        "Coverr API Key (optional)",
        value=st.session_state.coverr_api_key,
        type="password",
    )


def show_existing_output():
    if not FINAL_VIDEO.exists():
        return

    st.success("Final video ready hai.")
    st.video(str(FINAL_VIDEO))
    with FINAL_VIDEO.open("rb") as file:
        st.download_button(
            "Download final_video.mp4",
            data=file,
            file_name="final_video.mp4",
            mime="video/mp4",
            use_container_width=True,
        )


def render_app():
    ensure_folders()
    render_api_key_sidebar()

    st.title("NotebookLM Audio + Pexels Video Builder")
    st.caption("MP3 audio upload karo, topic se stock clips auto fetch karo ya manual MP4 clips upload karo, phir synced video generate karo.")

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
            st.info("Auto mode Pexels se videos fetch karega. Pexels mein 5 se kam results mile to Pixabay fallback use hoga, phir Coverr try hoga.")
        transcript_text = st.text_area(
            "Hindi Transcript / Subtitles (optional)",
            height=180,
            placeholder="Yahan Hindi transcript paste karo.",
        )

        generate = st.button("Generate Professional Video", type="primary", use_container_width=True)

    with right:
        st.subheader("Output")
        output_placeholder = st.empty()
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
                clear_old_videos()
                progress.progress(10, text="Stock videos search aur download ho rahe hain...")
                saved_videos = fetch_and_download_videos(
                    topic=search_topic,
                    destination_dir=VIDEO_DIR,
                    pexels_api_key=st.session_state.pexels_api_key,
                    pixabay_api_key=st.session_state.pixabay_api_key,
                    coverr_api_key=st.session_state.coverr_api_key,
                    limit=8,
                )
                st.info(f"{len(saved_videos)} stock video clips download ho gaye.")

            progress.progress(35, text="Audio duration aur clips prepare ho rahe hain...")
            create_video()
            progress.progress(100, text="Done")
        except VideoFetchError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.exception(exc)
            return

    show_existing_output()


if __name__ == "__main__":
    render_app()
