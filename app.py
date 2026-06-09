from pathlib import Path

import streamlit as st

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

    st.title("NotebookLM Audio + Pexels Video Builder")
    st.caption("MP3 audio aur Pexels MP4 clips upload karo, app auto trim, sync, subtitles aur transitions ke saath final video export karega.")

    left, right = st.columns([0.52, 0.48], gap="large")

    with left:
        audio_upload = st.file_uploader(
            "NotebookLM Audio (MP3)",
            type=["mp3"],
            accept_multiple_files=False,
        )
        video_uploads = st.file_uploader(
            "Pexels Videos (MP4 files)",
            type=["mp4", "mov", "mkv", "webm"],
            accept_multiple_files=True,
        )
        transcript_text = st.text_area(
            "Hindi Transcript / Subtitles (optional)",
            height=180,
            placeholder="Yahan Hindi transcript paste karo. Empty chhodne par faster-whisper se auto subtitle try hoga.",
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

    if not video_uploads:
        st.error("Pexels ke kam se kam 1 video file upload karo.")
        return

    save_uploaded_file(audio_upload, AUDIO_FILE)
    saved_videos = save_video_uploads(video_uploads)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if transcript_text.strip():
        TRANSCRIPT_FILE.write_text(transcript_text.strip(), encoding="utf-8")
    elif TRANSCRIPT_FILE.exists():
        TRANSCRIPT_FILE.write_text("", encoding="utf-8")

    with output_placeholder.container():
        st.info(f"{len(saved_videos)} video files save ho gaye. Render start ho raha hai.")
        progress = st.progress(0, text="Video build ho raha hai...")

        try:
            progress.progress(20, text="Audio duration aur clips prepare ho rahe hain...")
            create_video()
            progress.progress(100, text="Done")
        except Exception as exc:
            st.exception(exc)
            return

    show_existing_output()


if __name__ == "__main__":
    render_app()
