import math
import textwrap
from pathlib import Path

import numpy as np
from moviepy.editor import CompositeVideoClip, ImageClip
from PIL import Image, ImageDraw, ImageFont


def format_srt_time(seconds):
    seconds = max(0, float(seconds))
    millis = int(round((seconds - math.floor(seconds)) * 1000))
    whole = int(seconds)
    hours = whole // 3600
    minutes = (whole % 3600) // 60
    secs = whole % 60
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def split_text_into_segments(text, target_duration, max_chars):
    words = text.split()
    if not words:
        return []

    chunks = []
    current = []

    for word in words:
        candidate = " ".join([*current, word])
        if len(candidate) > max_chars and current:
            chunks.append(" ".join(current))
            current = [word]
        else:
            current.append(word)

    if current:
        chunks.append(" ".join(current))

    segment_duration = target_duration / len(chunks)
    return [
        {
            "start": i * segment_duration,
            "end": min(target_duration, (i + 1) * segment_duration),
            "text": chunk,
        }
        for i, chunk in enumerate(chunks)
    ]


def transcribe_audio_with_whisper(audio_path):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Auto subtitles ke liye faster-whisper install hona zaroori hai. "
            "Ya phir input/transcript.txt mein Hindi text daal do."
        ) from exc

    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, _info = model.transcribe(str(audio_path), language="hi", vad_filter=True)
    return [
        {"start": float(segment.start), "end": float(segment.end), "text": segment.text.strip()}
        for segment in segments
        if segment.text.strip()
    ]


def load_or_create_subtitle_segments(transcript_path, audio_path, target_duration, max_chars):
    if transcript_path.exists() and transcript_path.read_text(encoding="utf-8").strip():
        text = transcript_path.read_text(encoding="utf-8").strip()
        return split_text_into_segments(text, target_duration, max_chars)

    return [
        segment
        for segment in transcribe_audio_with_whisper(audio_path)
        if segment["start"] < target_duration
    ]


def write_srt(segments, output_path):
    lines = []
    for index, segment in enumerate(segments, start=1):
        lines.extend(
            [
                str(index),
                f"{format_srt_time(segment['start'])} --> {format_srt_time(segment['end'])}",
                segment["text"],
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def load_subtitle_font(font_path, font_size):
    candidates = [
        font_path,
        r"C:\Windows\Fonts\Nirmala.ttf",
        r"C:\Windows\Fonts\NirmalaB.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, font_size)

    return ImageFont.load_default()


def text_bbox(draw, position, text, font, stroke_width):
    try:
        return draw.multiline_textbbox(
            position,
            text,
            font=font,
            spacing=10,
            stroke_width=stroke_width,
            align="center",
        )
    except AttributeError:
        width, height = draw.multiline_textsize(text, font=font, spacing=10)
        x, y = position
        return x, y, x + width, y + height


def make_subtitle_image(
    text,
    clip_width,
    font,
    color,
    stroke_color,
    stroke_width,
    max_chars,
):
    wrapped = "\n".join(textwrap.wrap(text, width=max_chars))
    padding_x = 44
    padding_y = 28
    max_width = int(clip_width * 0.86)

    measure_image = Image.new("RGBA", (max_width, 400), (0, 0, 0, 0))
    draw = ImageDraw.Draw(measure_image)
    bbox = text_bbox(draw, (0, 0), wrapped, font, stroke_width)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    image_width = min(max_width, text_width + padding_x * 2)
    image_height = text_height + padding_y * 2
    image = Image.new("RGBA", (image_width, image_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    x = image_width / 2
    y = padding_y

    draw.multiline_text(
        (x, y),
        wrapped,
        font=font,
        fill=color,
        spacing=10,
        align="center",
        anchor="ma",
        stroke_width=stroke_width,
        stroke_fill=stroke_color,
    )
    return np.array(image)


def add_subtitles(
    video_clip,
    segments,
    font,
    font_path,
    font_size,
    color,
    stroke_color,
    stroke_width,
    max_chars,
):
    subtitle_clips = []
    subtitle_font = load_subtitle_font(font_path, font_size)

    for segment in segments:
        duration = max(0.1, segment["end"] - segment["start"])
        subtitle_image = make_subtitle_image(
            text=segment["text"],
            clip_width=video_clip.w,
            font=subtitle_font,
            color=color,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            max_chars=max_chars,
        )
        subtitle = (
            ImageClip(subtitle_image, transparent=True)
            .set_start(segment["start"])
            .set_duration(duration)
            .set_position(("center", int(video_clip.h * 0.78)))
        )
        subtitle_clips.append(subtitle)

    return CompositeVideoClip([video_clip, *subtitle_clips], size=video_clip.size)
