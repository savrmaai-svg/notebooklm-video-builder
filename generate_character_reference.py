from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "prototype_assets" / "character_references"
MODEL_ID = "stabilityai/sdxl-turbo"

PROMPT = (
    "Exactly two people only, one boy and one girl, full body, side by side, plain cream background. "
    "2D Indian TV cartoon model sheet. Left Aarav: teen boy, brown skin, messy black hair, red tunic, "
    "olive pants, turquoise pendant. Right Meera: teen girl, brown skin, hair bun, round glasses, brass "
    "goggles, cream shirt, green overalls, tool belt. Clean outlines, cel shading."
)

NEGATIVE = (
    "photograph, 3d, four people, three people, extra person, crowd, adult, child, sari, duplicate, twins, "
    "cropped body, text, letters, logo, watermark, deformed hands, extra limbs"
)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
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

    for seed in (1907, 4408, 8816):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        image = pipeline(
            prompt=PROMPT,
            negative_prompt=NEGATIVE,
            width=768,
            height=512,
            num_inference_steps=4,
            guidance_scale=1.0,
            generator=generator,
        ).images[0]
        path = OUTPUT / f"characters_{seed}.png"
        image.save(path)
        print(path, flush=True)


if __name__ == "__main__":
    main()
