from pathlib import Path
import time

import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "prototype_assets"
REFERENCE_SOURCE = ASSETS / "character_references" / "characters_1907.png"
REFERENCE = ASSETS / "locked_character_pair.png"
MODEL_ID = "stabilityai/sdxl-turbo"

SCENES = [
    (
        "scene_01_jungle_temple.png",
        "Hand-painted 2D TV cartoon. Same boy and girl walking through dense jungle toward a huge ancient "
        "stone sun temple at sunset, vines, golden mist, cautious expressions, wide cinematic shot.",
    ),
    (
        "scene_02_secret_door.png",
        "Hand-painted 2D TV cartoon. Same girl studies glowing symbols while same boy presses a round stone "
        "switch at an ancient temple door, falling dust, turquoise light, surprised faces, medium shot.",
    ),
    (
        "scene_03_blue_artifact.png",
        "Hand-painted 2D TV cartoon. Same boy and girl inside a dark temple chamber beside a glowing blue "
        "circular artifact, carved sun symbols, gears, floating dust, amazed expressions, cinematic light.",
    ),
    (
        "scene_04_escape.png",
        "Hand-painted 2D TV cartoon. Same boy and girl running from a collapsing temple chamber, falling stones, "
        "dust, sparks, urgent faces, sunrise through exit, dynamic safe family adventure shot.",
    ),
]


def build_reference():
    with Image.open(REFERENCE_SOURCE) as loaded:
        pair = loaded.convert("RGB").crop((0, 0, 590, 512))
    canvas = Image.new("RGB", (768, 512), (247, 242, 221))
    pair.thumbnail((720, 500), Image.Resampling.LANCZOS)
    canvas.paste(pair, ((768 - pair.width) // 2, (512 - pair.height) // 2))
    canvas.save(REFERENCE)
    return canvas


def main():
    reference = build_reference()
    started = time.perf_counter()
    pipeline = AutoPipelineForImage2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        local_files_only=True,
    )
    pipeline.enable_model_cpu_offload()
    pipeline.enable_attention_slicing()
    pipeline.enable_vae_slicing()

    for index, (filename, prompt) in enumerate(SCENES):
        generator = torch.Generator(device="cpu").manual_seed(6100 + index)
        image = pipeline(
            prompt=prompt,
            image=reference,
            strength=0.78,
            num_inference_steps=4,
            guidance_scale=0.0,
            generator=generator,
        ).images[0]
        path = ASSETS / filename
        image.save(path)
        print(f"conditioned {index + 1}/{len(SCENES)}: {path}", flush=True)
    print(f"completed in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
