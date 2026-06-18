from pathlib import Path
import time

import torch
from diffusers import AutoPipelineForText2Image


MODEL_ID = "stabilityai/sdxl-turbo"
OUTPUT_DIR = Path(__file__).resolve().parent / "prototype_assets"
WIDTH = 768
HEIGHT = 512

CHARACTER_STYLE = (
    "Premium hand-painted 2D Hindi television cartoon. Recurring pair only: Aarav, Indian teen boy, "
    "brown skin, tousled black hair, red gold-trim tunic, olive pants, turquoise pendant; Meera, Indian "
    "teen girl, brown skin, black hair bun, round glasses, brass goggles, cream shirt, green overalls. "
    "Same faces and costumes. Clean outlines, expressive faces, rich cel shading, painted background."
)

SCENES = [
    (
        "scene_01_jungle_temple.png",
        "2D cartoon wide shot: dense Indian jungle at sunset, enormous ancient stone sun temple covered "
        "in vines. Aarav and Meera walk toward the temple, full body, curious and determined. Golden mist, "
        "cinematic adventure, coherent perspective, 16:9.",
    ),
    (
        "scene_02_secret_door.png",
        "2D cartoon medium shot: entrance of an ancient jungle temple. Meera studies glowing carved symbols; "
        "Aarav presses a circular stone mechanism. Giant stone door opening, falling dust, turquoise light, "
        "surprised expressions, cinematic 16:9.",
    ),
    (
        "scene_03_blue_artifact.png",
        "2D cartoon wide shot inside an ancient temple treasure chamber. Aarav and Meera beside a circular "
        "mechanical artifact glowing turquoise blue. Stone gears, carved sun symbols, floating dust. Meera "
        "points, Aarav looks amazed, dramatic blue light, 16:9.",
    ),
    (
        "scene_04_escape.png",
        "2D cartoon action shot inside a collapsing ancient temple. Aarav and Meera run toward the exit, "
        "falling stones behind them, dust and golden sparks, urgent expressive faces, sunrise through the "
        "doorway, safe family adventure, cinematic 16:9.",
    ),
]

NEGATIVE = (
    "photograph, photorealistic, live action, 3D render, flat geometric vector, stick figure, chibi, "
    "different clothes, different hairstyle, adult characters, duplicate person, extra character, "
    "deformed hands, extra limbs, cropped face, text, letters, logo, watermark, horror, blood"
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    started = time.perf_counter()
    pipeline = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        local_files_only=True,
    )
    pipeline.enable_model_cpu_offload()
    pipeline.enable_attention_slicing()
    pipeline.enable_vae_slicing()

    for index, (filename, scene) in enumerate(SCENES):
        generator = torch.Generator(device="cpu").manual_seed(880017)
        image = pipeline(
            prompt=scene,
            prompt_2=CHARACTER_STYLE,
            negative_prompt=NEGATIVE,
            width=WIDTH,
            height=HEIGHT,
            num_inference_steps=4,
            guidance_scale=0.0,
            generator=generator,
        ).images[0]
        path = OUTPUT_DIR / filename
        image.save(path, quality=95)
        print(f"generated {index + 1}/{len(SCENES)}: {path}", flush=True)

    print(f"completed in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
