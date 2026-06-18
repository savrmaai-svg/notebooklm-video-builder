from array import array
import math
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


CANVAS_SIZE = (1280, 720)
CARTOON_FPS = 12
BACKGROUND_SECONDS = 7


def _audio_envelope(ffmpeg, audio_path, duration, fps=CARTOON_FPS):
    sample_rate = 16000
    result = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(audio_path),
            "-t",
            f"{duration:.3f}",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "s16le",
            "-",
        ],
        check=True,
        capture_output=True,
        timeout=max(120, int(duration * 2)),
    )
    samples = array("h")
    samples.frombytes(result.stdout)
    samples_per_frame = max(1, sample_rate // fps)
    levels = []
    expected_frames = max(1, int(math.ceil(duration * fps)))
    for frame_index in range(expected_frames):
        start = frame_index * samples_per_frame
        chunk = samples[start:start + samples_per_frame]
        if not chunk:
            levels.append(0.0)
            continue
        mean_square = sum(sample * sample for sample in chunk) / len(chunk)
        levels.append(math.sqrt(mean_square))

    active_levels = sorted(level for level in levels if level > 30)
    reference = active_levels[int(len(active_levels) * 0.88)] if active_levels else 1.0
    return [min(1.0, level / max(1.0, reference)) for level in levels]


def _cartoon_background(path):
    with Image.open(path) as loaded:
        image = ImageOps.fit(loaded.convert("RGB"), CANVAS_SIZE, method=Image.Resampling.LANCZOS)
    image = image.filter(ImageFilter.MedianFilter(size=3))
    image = ImageOps.posterize(image, 6)
    image = ImageEnhance.Color(image).enhance(1.18)
    image = ImageEnhance.Contrast(image).enhance(1.05)

    shade = Image.new("RGBA", CANVAS_SIZE, (0, 0, 0, 0))
    shade_draw = ImageDraw.Draw(shade)
    for y in range(430, 720):
        alpha = int(70 * ((y - 430) / 290))
        shade_draw.line((0, y, 1280, y), fill=(8, 13, 24, alpha))
    return Image.alpha_composite(image.convert("RGBA"), shade)


def _mouth(draw, cx, cy, state, scale, speaking=True):
    if not speaking or state == 0:
        draw.arc(
            (cx - 25 * scale, cy - 7 * scale, cx + 25 * scale, cy + 12 * scale),
            10,
            170,
            fill=(65, 24, 23, 255),
            width=max(2, int(4 * scale)),
        )
        return

    height = 15 if state == 1 else 31
    box = (
        cx - 25 * scale,
        cy - height * scale / 2,
        cx + 25 * scale,
        cy + height * scale / 2,
    )
    draw.ellipse(box, fill=(67, 20, 28, 255), outline=(35, 12, 18, 255), width=max(2, int(3 * scale)))
    if state == 2:
        draw.arc(
            (cx - 17 * scale, cy, cx + 17 * scale, cy + 13 * scale),
            180,
            360,
            fill=(224, 88, 92, 255),
            width=max(2, int(4 * scale)),
        )


def _hero_character(mouth_state=0, blink=False, speaking=True, gesture=0.0):
    scale = 2
    width, height = 430 * scale, 650 * scale
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    skin = (181, 105, 62, 255)
    skin_shadow = (139, 72, 48, 255)
    ink = (35, 24, 29, 255)
    red = (169, 43, 39, 255)
    red_shadow = (112, 26, 34, 255)
    gold = (236, 178, 66, 255)

    def p(value):
        return value * scale

    # Back arm and legs
    draw.rounded_rectangle((p(92), p(390), p(174), p(620)), radius=p(25), fill=(69, 68, 58, 255), outline=ink, width=p(5))
    draw.rounded_rectangle((p(245), p(390), p(327), p(620)), radius=p(25), fill=(69, 68, 58, 255), outline=ink, width=p(5))
    draw.rounded_rectangle((p(70), p(575), p(180), p(641)), radius=p(20), fill=(72, 43, 35, 255), outline=ink, width=p(5))
    draw.rounded_rectangle((p(239), p(575), p(350), p(641)), radius=p(20), fill=(72, 43, 35, 255), outline=ink, width=p(5))

    arm_raise = int(22 * max(0.0, gesture))
    draw.line((p(105), p(305), p(50), p(420 - arm_raise)), fill=ink, width=p(50))
    draw.line((p(105), p(305), p(50), p(420 - arm_raise)), fill=skin, width=p(39))
    draw.ellipse((p(25), p(392 - arm_raise), p(78), p(446 - arm_raise)), fill=skin, outline=ink, width=p(4))

    # Torso and costume
    draw.polygon(
        [(p(113), p(270)), (p(307), p(270)), (p(345), p(470)), (p(75), p(470))],
        fill=red,
        outline=ink,
    )
    draw.line((p(82), p(463), p(338), p(463)), fill=gold, width=p(12))
    draw.rectangle((p(80), p(430), p(340), p(468)), fill=(76, 45, 34, 255), outline=ink, width=p(4))
    draw.rounded_rectangle((p(187), p(425), p(235), p(474)), radius=p(8), fill=gold, outline=ink, width=p(4))
    draw.polygon([(p(115), p(275)), (p(165), p(275)), (p(260), p(466)), (p(220), p(466))], fill=red_shadow)
    draw.line((p(133), p(281), p(248), p(462)), fill=gold, width=p(9))
    draw.line((p(306), p(306), p(372), p(420 + arm_raise)), fill=ink, width=p(51))
    draw.line((p(306), p(306), p(372), p(420 + arm_raise)), fill=skin, width=p(39))
    draw.ellipse((p(345), p(396 + arm_raise), p(400), p(452 + arm_raise)), fill=skin, outline=ink, width=p(4))

    # Neck, ears, head
    draw.rounded_rectangle((p(170), p(225), p(250), p(298)), radius=p(24), fill=skin_shadow, outline=ink, width=p(4))
    draw.ellipse((p(67), p(83), p(137), p(180)), fill=skin, outline=ink, width=p(5))
    draw.ellipse((p(283), p(83), p(353), p(180)), fill=skin, outline=ink, width=p(5))
    draw.ellipse((p(88), p(20), p(332), p(265)), fill=skin, outline=ink, width=p(6))

    # Hair silhouette
    hair = [
        (p(95), p(105)), (p(70), p(52)), (p(126), p(64)), (p(136), p(12)),
        (p(174), p(45)), (p(209), p(0)), (p(228), p(45)), (p(281), p(8)),
        (p(276), p(57)), (p(342), p(48)), (p(310), p(102)), (p(326), p(132)),
        (p(285), p(115)), (p(267), p(75)), (p(228), p(91)), (p(188), p(64)),
        (p(155), p(96)), (p(119), p(80)),
    ]
    draw.polygon(hair, fill=(31, 27, 31, 255), outline=ink)

    # Brows and eyes
    draw.line((p(133), p(126), p(184), p(116)), fill=ink, width=p(7))
    draw.line((p(237), p(116), p(287), p(126)), fill=ink, width=p(7))
    if blink:
        draw.arc((p(132), p(130), p(188), p(164)), 0, 180, fill=ink, width=p(5))
        draw.arc((p(233), p(130), p(289), p(164)), 0, 180, fill=ink, width=p(5))
    else:
        draw.ellipse((p(132), p(126), p(188), p(180)), fill=(249, 246, 226, 255), outline=ink, width=p(4))
        draw.ellipse((p(233), p(126), p(289), p(180)), fill=(249, 246, 226, 255), outline=ink, width=p(4))
        draw.ellipse((p(153), p(140), p(177), p(171)), fill=(62, 39, 28, 255))
        draw.ellipse((p(244), p(140), p(268), p(171)), fill=(62, 39, 28, 255))
        draw.ellipse((p(160), p(145), p(169), p(155)), fill=(255, 255, 255, 255))
        draw.ellipse((p(251), p(145), p(260), p(155)), fill=(255, 255, 255, 255))
    draw.line((p(210), p(158), p(201), p(189), p(219), p(192)), fill=skin_shadow, width=p(4))
    _mouth(draw, p(210), p(218), mouth_state, scale, speaking=speaking)

    # Pendant and costume trim
    draw.ellipse((p(190), p(275), p(230), p(315)), fill=(31, 190, 190, 255), outline=gold, width=p(5))
    draw.line((p(170), p(266), p(204), p(285), p(250), p(266)), fill=gold, width=p(5))
    image = image.resize((430, 650), Image.Resampling.LANCZOS)
    return image


def _inventor_character(mouth_state=0, blink=False, speaking=True, gesture=0.0):
    scale = 2
    width, height = 430 * scale, 650 * scale
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    ink = (34, 25, 31, 255)
    skin = (171, 98, 63, 255)
    green = (43, 92, 67, 255)
    green_dark = (29, 57, 49, 255)
    cream = (236, 218, 172, 255)
    brass = (207, 142, 46, 255)

    def p(value):
        return value * scale

    draw.rounded_rectangle((p(101), p(400), p(182), p(618)), radius=p(24), fill=green_dark, outline=ink, width=p(5))
    draw.rounded_rectangle((p(243), p(400), p(324), p(618)), radius=p(24), fill=green_dark, outline=ink, width=p(5))
    draw.rounded_rectangle((p(75), p(578), p(186), p(641)), radius=p(20), fill=(91, 57, 35, 255), outline=ink, width=p(5))
    draw.rounded_rectangle((p(238), p(578), p(350), p(641)), radius=p(20), fill=(91, 57, 35, 255), outline=ink, width=p(5))

    arm_raise = int(25 * max(0.0, gesture))
    draw.line((p(110), p(315), p(53), p(425 - arm_raise)), fill=ink, width=p(48))
    draw.line((p(110), p(315), p(53), p(425 - arm_raise)), fill=cream, width=p(37))
    draw.ellipse((p(28), p(398 - arm_raise), p(78), p(448 - arm_raise)), fill=skin, outline=ink, width=p(4))
    draw.line((p(310), p(315), p(375), p(420 + arm_raise)), fill=ink, width=p(48))
    draw.line((p(310), p(315), p(375), p(420 + arm_raise)), fill=cream, width=p(37))
    draw.ellipse((p(349), p(395 + arm_raise), p(401), p(449 + arm_raise)), fill=skin, outline=ink, width=p(4))

    draw.rounded_rectangle((p(93), p(278), p(327), p(482)), radius=p(45), fill=cream, outline=ink, width=p(6))
    draw.polygon([(p(130), p(300)), (p(290), p(300)), (p(330), p(500)), (p(90), p(500))], fill=green, outline=ink)
    draw.rectangle((p(115), p(425), p(310), p(465)), fill=(83, 55, 36, 255), outline=ink, width=p(4))
    draw.rounded_rectangle((p(190), p(420), p(236), p(468)), radius=p(8), fill=brass, outline=ink, width=p(4))
    draw.line((p(145), p(298), p(155), p(430)), fill=brass, width=p(10))
    draw.line((p(277), p(298), p(266), p(430)), fill=brass, width=p(10))
    draw.rectangle((p(160), p(350), p(270), p(414)), fill=green_dark, outline=ink, width=p(4))

    draw.rounded_rectangle((p(171), p(225), p(250), p(298)), radius=p(24), fill=skin, outline=ink, width=p(4))
    draw.ellipse((p(75), p(85), p(138), p(177)), fill=skin, outline=ink, width=p(5))
    draw.ellipse((p(281), p(85), p(344), p(177)), fill=skin, outline=ink, width=p(5))
    draw.ellipse((p(92), p(28), p(328), p(267)), fill=skin, outline=ink, width=p(6))

    # Hair, bun, and goggles
    draw.ellipse((p(272), p(8), p(365), p(100)), fill=(29, 28, 31, 255), outline=ink, width=p(5))
    draw.polygon(
        [(p(95), p(112)), (p(87), p(55)), (p(143), p(70)), (p(168), p(20)),
         (p(208), p(60)), (p(252), p(18)), (p(270), p(67)), (p(331), p(55)),
         (p(310), p(120)), (p(275), p(88)), (p(233), p(106)), (p(190), p(72)),
         (p(150), p(104))],
        fill=(29, 28, 31, 255),
        outline=ink,
    )
    draw.ellipse((p(142), p(48), p(212), p(102)), fill=(43, 62, 59, 255), outline=brass, width=p(7))
    draw.ellipse((p(218), p(48), p(288), p(102)), fill=(43, 62, 59, 255), outline=brass, width=p(7))
    draw.line((p(210), p(75), p(220), p(75)), fill=brass, width=p(7))

    # Glasses and eyes
    draw.line((p(135), p(144), p(288), p(144)), fill=ink, width=p(5))
    if blink:
        draw.arc((p(132), p(139), p(190), p(172)), 0, 180, fill=ink, width=p(5))
        draw.arc((p(231), p(139), p(289), p(172)), 0, 180, fill=ink, width=p(5))
    else:
        draw.ellipse((p(132), p(127), p(192), p(185)), fill=(248, 244, 220, 255), outline=ink, width=p(4))
        draw.ellipse((p(229), p(127), p(289), p(185)), fill=(248, 244, 220, 255), outline=ink, width=p(4))
        draw.ellipse((p(154), p(142), p(178), p(174)), fill=(54, 38, 30, 255))
        draw.ellipse((p(241), p(142), p(265), p(174)), fill=(54, 38, 30, 255))
        draw.ellipse((p(160), p(146), p(169), p(156)), fill=(255, 255, 255, 255))
        draw.ellipse((p(247), p(146), p(256), p(156)), fill=(255, 255, 255, 255))
    draw.ellipse((p(122), p(116), p(201), p(195)), outline=(56, 48, 43, 255), width=p(6))
    draw.ellipse((p(220), p(116), p(299), p(195)), outline=(56, 48, 43, 255), width=p(6))
    draw.line((p(200), p(153), p(221), p(153)), fill=(56, 48, 43, 255), width=p(5))
    _mouth(draw, p(211), p(220), mouth_state, scale, speaking=speaking)

    image = image.resize((430, 650), Image.Resampling.LANCZOS)
    return image


def render_cartoon_episode(
    ffmpeg,
    audio_path,
    background_paths,
    output_path,
    duration,
    fps=CARTOON_FPS,
    preview_path=None,
):
    background_paths = [Path(path) for path in background_paths]
    if not background_paths:
        raise ValueError("Cartoon episode ke liye backgrounds required hain.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    silent_video = output_path.with_name(f"{output_path.stem}_silent.mp4")
    envelope = _audio_envelope(ffmpeg, audio_path, duration, fps=fps)
    backgrounds = [_cartoon_background(path) for path in background_paths]
    frame_total = max(1, int(math.ceil(duration * fps)))

    encoder = subprocess.Popen(
        [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{CANVAS_SIZE[0]}x{CANVAS_SIZE[1]}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            str(silent_video),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    try:
        for frame_index in range(frame_total):
            second = frame_index / fps
            scene_index = min(len(backgrounds) - 1, int(second // BACKGROUND_SECONDS) % len(backgrounds))
            base = backgrounds[scene_index].copy()
            level = envelope[min(frame_index, len(envelope) - 1)]
            mouth_state = 0 if level < 0.12 else 1 if level < 0.48 else 2
            speaking_hero = scene_index % 2 == 0
            blink = frame_index % (fps * 4) in {0, 1}
            gesture = max(0.0, math.sin(second * 2.3)) * (0.75 if level > 0.25 else 0.18)
            bob = int(math.sin(second * 3.0) * (3 if level > 0.12 else 1))

            hero = _hero_character(
                mouth_state=mouth_state if speaking_hero else 0,
                blink=blink,
                speaking=speaking_hero,
                gesture=gesture,
            )
            inventor = _inventor_character(
                mouth_state=mouth_state if not speaking_hero else 0,
                blink=blink,
                speaking=not speaking_hero,
                gesture=gesture,
            )
            base.alpha_composite(hero, (72, 84 + bob))
            base.alpha_composite(inventor, (778, 86 - bob))

            # Ground characters and keep faces readable against busy stock backgrounds.
            draw = ImageDraw.Draw(base)
            draw.ellipse((65, 670, 500, 710), fill=(8, 10, 18, 90))
            draw.ellipse((770, 670, 1215, 710), fill=(8, 10, 18, 90))

            if preview_path and frame_index == min(frame_total - 1, fps * 2):
                Path(preview_path).parent.mkdir(parents=True, exist_ok=True)
                base.convert("RGB").save(preview_path, "JPEG", quality=94)

            encoder.stdin.write(base.convert("RGB").tobytes())
    finally:
        if encoder.stdin:
            encoder.stdin.close()

    stderr = encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
    return_code = encoder.wait(timeout=max(180, int(duration * 5)))
    if return_code != 0:
        raise RuntimeError(f"Cartoon video encode failed: {stderr[-1200:]}")

    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(silent_video),
            "-i",
            str(audio_path),
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-shortest",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=max(180, int(duration * 4)),
    )
    silent_video.unlink(missing_ok=True)
    return output_path
