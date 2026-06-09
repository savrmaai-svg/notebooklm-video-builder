from moviepy.editor import AudioFileClip

from audio_utils import get_target_duration
from config import (
    AUDIO_FILE,
    CROSSFADE_SECONDS,
    FINAL_VIDEO,
    FONT,
    FONT_PATH,
    FONT_SIZE,
    FPS,
    INPUT_DIR,
    OUTPUT_DIR,
    SUBTITLE_COLOR,
    SUBTITLE_FILE,
    SUBTITLE_MAX_CHARS,
    SUBTITLE_STROKE_COLOR,
    SUBTITLE_STROKE_WIDTH,
    TARGET_SECONDS,
    TRANSCRIPT_FILE,
    VIDEO_DIR,
    VIDEO_SIZE,
)
from subtitle_utils import add_subtitles, load_or_create_subtitle_segments, write_srt
from video_utils import build_synced_video, collect_video_files


def ensure_folders():
    INPUT_DIR.mkdir(exist_ok=True)
    VIDEO_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)


def validate_inputs():
    if not AUDIO_FILE.exists():
        raise FileNotFoundError("NotebookLM audio ko input/audio.mp3 naam se rakho.")


def create_video():
    ensure_folders()
    validate_inputs()

    target_duration = get_target_duration(AUDIO_FILE, TARGET_SECONDS)
    video_paths = collect_video_files(VIDEO_DIR)

    print(f"Audio duration: {target_duration:.2f} seconds")
    print(f"Pexels videos found: {len(video_paths)}")

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

    print(f"Done: {FINAL_VIDEO}")
    print(f"Subtitles: {SUBTITLE_FILE}")


if __name__ == "__main__":
    create_video()
