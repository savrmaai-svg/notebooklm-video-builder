import math
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
import imageio_ffmpeg

from cartoon_engine import _audio_envelope


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "prototype_assets"
OUTPUT_DIR = ROOT / "output"
SOURCE_VIDEO = Path(r"C:\Users\Sameer\Downloads\final_video (21).mp4")
OUTPUT = OUTPUT_DIR / "surya_mandir_cartoon_prototype_20s.mp4"
AUDIO = OUTPUT_DIR / "surya_mandir_prototype_audio.m4a"
FPS = 12
DURATION = 20
SIZE = (1280, 720)

SCENES = [
    (ASSETS / "scene_01_jungle_temple.png", (322, 157), "mist"),
    (ASSETS / "scene_02_secret_door.png", (571, 221), "glow"),
    (ASSETS / "scene_03_blue_artifact.png", (297, 177), "dust"),
    (ASSETS / "scene_04_escape.png", (281, 148), "collapse"),
]


def draw_mouth(image, center, state):
    if state == 0:
        return
    draw = ImageDraw.Draw(image, "RGBA")
    x, y = center
    width = 19
    height = 7 if state == 1 else 14
    draw.ellipse(
        (x - width // 2, y - height // 2, x + width // 2, y + height // 2),
        fill=(62, 20, 25, 245),
        outline=(30, 13, 18, 255),
        width=2,
    )
    if state == 2:
        draw.arc((x - 6, y, x + 6, y + 7), 180, 360, fill=(231, 97, 105, 255), width=2)


def camera_frame(source, progress, scene_index):
    source = source.convert("RGB")
    base = ImageOps.fit(source, SIZE, method=Image.Resampling.LANCZOS)
    zoom = 1.0 + 0.07 * progress
    enlarged = base.resize((int(SIZE[0] * zoom), int(SIZE[1] * zoom)), Image.Resampling.LANCZOS)
    max_x = enlarged.width - SIZE[0]
    max_y = enlarged.height - SIZE[1]
    if scene_index % 2:
        x = int(max_x * (1.0 - progress) * 0.75)
    else:
        x = int(max_x * progress * 0.75)
    y = int(max_y * (0.35 + 0.2 * math.sin(progress * math.pi)))
    return enlarged.crop((x, y, x + SIZE[0], y + SIZE[1])).convert("RGBA")


def add_effects(frame, effect, frame_index, local_progress):
    overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    if effect == "mist":
        for index in range(12):
            x = int((index * 137 + frame_index * 2.2) % 1500) - 100
            y = 410 + (index % 4) * 55
            draw.ellipse((x, y, x + 260, y + 90), fill=(220, 238, 219, 18))
        overlay = overlay.filter(ImageFilter.GaussianBlur(28))
    elif effect == "glow":
        for index in range(22):
            angle = index * 1.7 + frame_index * 0.06
            radius = 70 + (index * 23) % 260
            x = 640 + math.cos(angle) * radius
            y = 370 + math.sin(angle) * radius * 0.55
            size = 3 + index % 5
            draw.ellipse((x - size, y - size, x + size, y + size), fill=(70, 255, 224, 90))
        overlay = overlay.filter(ImageFilter.GaussianBlur(2))
    elif effect == "dust":
        for index in range(45):
            x = (index * 91 + frame_index * (1 + index % 3)) % 1280
            y = (index * 53 + frame_index * 0.6) % 680
            size = 1 + index % 3
            draw.ellipse((x, y, x + size, y + size), fill=(255, 225, 150, 80))
    else:
        for index in range(15):
            x = (index * 103 + frame_index * (2 + index % 2)) % 1280
            y = int((index * 71 + frame_index * (4 + index % 3)) % 650)
            rock = 5 + index % 9
            draw.polygon(
                [(x, y), (x + rock, y + 3), (x + rock - 2, y + rock), (x - 2, y + rock - 1)],
                fill=(86, 67, 54, 125),
            )
        haze = int(45 * local_progress)
        draw.rectangle((0, 520, 1280, 720), fill=(180, 145, 105, haze))
    return Image.alpha_composite(frame, overlay)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg, "-y", "-i", str(SOURCE_VIDEO), "-t", str(DURATION), "-vn", "-c:a", "aac", str(AUDIO)],
        check=True,
        capture_output=True,
    )
    levels = _audio_envelope(ffmpeg, AUDIO, DURATION, fps=FPS)
    source_images = [Image.open(path).convert("RGBA") for path, _, _ in SCENES]
    silent = OUTPUT.with_name("surya_mandir_cartoon_prototype_silent.mp4")
    encoder = subprocess.Popen(
        [
            ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720", "-r", str(FPS),
            "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
            "-pix_fmt", "yuv420p", str(silent),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    frames_per_scene = FPS * 5
    for frame_index in range(FPS * DURATION):
        scene_index = min(3, frame_index // frames_per_scene)
        local_frame = frame_index % frames_per_scene
        progress = local_frame / max(1, frames_per_scene - 1)
        image = source_images[scene_index].copy()
        level = levels[min(frame_index, len(levels) - 1)]
        mouth_state = 0 if level < 0.14 else 1 if level < 0.48 else 2
        draw_mouth(image, SCENES[scene_index][1], mouth_state)
        frame = camera_frame(image, progress, scene_index)
        frame = add_effects(frame, SCENES[scene_index][2], frame_index, progress)

        # Soft crossfade from the previous illustrated story scene.
        if scene_index > 0 and local_frame < 8:
            previous = camera_frame(source_images[scene_index - 1], 1.0, scene_index - 1)
            frame = Image.blend(previous, frame, local_frame / 8)

        frame = ImageEnhance.Color(frame.convert("RGB")).enhance(1.05)
        encoder.stdin.write(frame.tobytes())

    encoder.stdin.close()
    stderr = encoder.stderr.read().decode("utf-8", errors="replace")
    code = encoder.wait()
    if code:
        raise RuntimeError(stderr[-1500:])

    subprocess.run(
        [
            ffmpeg, "-y", "-i", str(silent), "-i", str(AUDIO), "-t", str(DURATION),
            "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", str(OUTPUT),
        ],
        check=True,
        capture_output=True,
    )
    silent.unlink(missing_ok=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
