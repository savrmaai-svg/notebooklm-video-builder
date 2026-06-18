# Claude Handoff: NotebookLM Video Builder

## Links

- GitHub repository: https://github.com/savrmaai-svg/notebooklm-video-builder
- Branch: `main`
- Public Streamlit app: https://notebooklm-video-builder-flnj8dcdwh6ffwgwmqkwcu.streamlit.app/
- Visual target reference: https://youtu.be/87SFHzwjF20

## Product Goal

The app accepts an audio file and a transcript/topic, then creates a complete Hindi 2D illustrated story episode. The desired result is similar in production structure to the reference video: recurring illustrated characters, story-specific illustrated backgrounds, multiple poses and expressions, lip sync, camera movement, and effects. Do not copy characters, artwork, branding, or story content from the reference. Create an original visual identity.

The most important requirement is that every background and action must follow the transcript. Do not place static characters over random Pexels/Pixabay photos.

## Required Scene Logic

Split the transcript into timestamped 5-8 second shots. Each shot needs structured data:

- location and continuity location ID
- time of day and weather
- characters present and active speaker
- character action, pose, expression, and screen position
- props and environmental effects
- camera shot and movement
- dialogue timing and mouth cues

Keep the same background/location across adjacent shots until the story changes location. Change camera angle, pose, expression, and action inside the same location.

## Current Repository

- `app.py`: Streamlit UI, audio extraction, source selection, FFmpeg orchestration, full video and Shorts export.
- `api_fetcher.py`: Pexels/Pixabay/Coverr video fetching plus Pexels/Pixabay/Wikimedia/Openverse image fetching and credits.
- `cartoon_engine.py`: current simple code-drawn characters and audio-volume mouth animation.
- `video_utils.py`, `audio_utils.py`, `subtitle_utils.py`: older utility modules.
- `requirements.txt`: lightweight Streamlit Cloud dependencies.

The current public `2D Cartoon Episode (Lip Sync)` mode is not the desired final quality. It draws two simple static vector characters over stock-photo backgrounds. Treat it as a temporary prototype, not as the final architecture.

## Local Open-Model Prototype

The repository also includes experimental local scripts:

- `generate_character_reference.py`: creates character reference candidates.
- `generate_story_prototype.py`: initial text-to-image scene experiment.
- `generate_conditioned_scenes.py`: character-reference image-to-image scene generation.
- `render_story_prototype.py`: renders four scenes into a 20-second MP4 with camera motion, effects, original audio, and experimental mouth cues.
- `requirements-cartoon-local.txt`: separate heavy local dependencies. Do not merge these into Streamlit Cloud `requirements.txt`.

Local prototype output is intentionally not committed. Model caches and virtual environments must never be committed.

## Hardware

- NVIDIA GeForce RTX 4050 Laptop GPU
- 6 GB VRAM
- approximately 15 GB system RAM
- Windows

Use quantized/offloaded models suitable for 6 GB VRAM. The public Streamlit Cloud instance cannot access this local GPU. Keep heavy generation in a local worker or a separately hosted GPU worker.

## Recommended Architecture

1. Streamlit remains the controller/UI.
2. A local GPU worker transcribes audio and creates a timestamped scene plan.
3. Build and approve a reusable character bible before episode generation.
4. Use an open-weight image model with reference conditioning (IP-Adapter/ControlNet/LoRA or a compatible alternative) for consistent illustrated scenes.
5. Generate reusable mouth shapes, expressions, and poses for each recurring character.
6. Use speaker-aware phoneme/viseme timing, not only audio volume.
7. Composite scenes and effects with Blender/OpenToonz/Pillow/FFmpeg.
8. Preserve uploaded audio unchanged and mute all source media.
9. Save scene metadata, seeds, prompts, character IDs, source licenses, and credits.

## Prototype Findings

- SDXL-Turbo runs on the RTX 4050 using CPU offload.
- Direct text prompts did not preserve character count or identity reliably.
- Reference image-to-image produced story-specific scenes but clothing and facial details still drifted.
- IP-Adapter SDXL experiments hit a runtime compatibility issue in the tested Diffusers/Transformers combinations. Do not assume that path is working without a minimal isolated test.
- Manually positioned mouth overlays are fragile. Replace them with face landmarks or rigged character mouth layers.

## Acceptance Criteria

- The same named character is recognizably identical in every shot.
- Every shot visually represents its transcript segment.
- No random real-photo background appears behind illustrated characters.
- Only the active speaker lip-syncs.
- At least six useful mouth/viseme shapes are supported.
- Characters blink and have head, hand, and body movement.
- Locations maintain continuity between adjacent shots.
- Rain, fire, smoke, dust, and lighting effects follow the scene plan.
- A 20-second prototype must be approved before generating a full episode.
- Existing documentary mode and the public app URL must remain working.

## Security

Never commit API keys, GitHub tokens, model-service tokens, or Streamlit secrets. Existing secret names are `PEXELS_API_KEY`, `PIXABAY_API_KEY`, and optionally `COVERR_API_KEY`. Read them from Streamlit secrets or environment variables only.

