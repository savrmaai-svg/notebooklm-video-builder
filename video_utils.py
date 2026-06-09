from pathlib import Path

from PIL import Image

# MoviePy 1.0.3 expects Image.ANTIALIAS, which Pillow 10 removed.
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.video.fx.all import crop, resize


SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}


def collect_video_files(video_dir):
    videos = [
        path
        for path in Path(video_dir).iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(videos)


def fit_to_canvas(clip, size):
    target_w, target_h = size
    clip = resize(clip, height=target_h)

    if clip.w < target_w:
        clip = resize(clip, width=target_w)

    x_center = clip.w / 2
    y_center = clip.h / 2
    return crop(
        clip,
        width=target_w,
        height=target_h,
        x_center=x_center,
        y_center=y_center,
    )


def build_synced_video(video_paths, target_duration, size, fps, crossfade_seconds):
    if not video_paths:
        raise FileNotFoundError("input/videos folder mein MP4 videos nahi mile.")

    clips = []
    elapsed = 0.0
    index = 0

    while elapsed < target_duration:
        video_path = video_paths[index % len(video_paths)]
        source = VideoFileClip(str(video_path)).without_audio()
        remaining = target_duration - elapsed
        use_seconds = min(source.duration, remaining + crossfade_seconds)

        clip = source.subclip(0, use_seconds)
        clip = fit_to_canvas(clip, size).set_fps(fps)

        if clips and crossfade_seconds > 0:
            clip = clip.crossfadein(min(crossfade_seconds, clip.duration / 3))

        clips.append(clip)
        elapsed += max(0.1, use_seconds - (crossfade_seconds if clips[:-1] else 0))
        index += 1

    final = concatenate_videoclips(
        clips,
        method="compose",
        padding=-crossfade_seconds if crossfade_seconds > 0 else 0,
    )
    return final.subclip(0, target_duration)
