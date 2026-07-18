# Veo / AI-video PROMPT generator.
# The slow part of making an AI cartoon isn't the generating — it's writing a fresh prompt for every
# clip and keeping the characters looking identical across them. This builds every clip prompt for you
# from one story, repeating a LOCKED character + style block verbatim so the model keeps consistency.
import re
import streamlit as st

STYLES = {
    "Pixar-style 3D cartoon": (
        "Pixar-style 3D animated film, highly detailed character rendering, soft cinematic lighting, "
        "warm colour grade, shallow depth of field, 24fps film look"),
    "2D hand-drawn cartoon": (
        "2D hand-drawn animated film, clean bold outlines, flat vibrant colours, expressive character "
        "animation, storybook backgrounds"),
    "Anime / Japanese animation": (
        "Japanese anime style, cel-shaded characters, detailed painted backgrounds, dramatic lighting, "
        "expressive eyes, cinematic composition"),
    "Realistic cinematic (live-action look)": (
        "photorealistic cinematic film still, natural lighting, 35mm lens, shallow depth of field, "
        "filmic colour grade, no text"),
    "Claymation / stop-motion": (
        "stop-motion claymation style, tactile clay textures, handcrafted sets, soft studio lighting"),
}

CAMERAS = ["slow push in", "slow pull back", "static wide shot", "medium close-up", "over-the-shoulder",
           "low angle hero shot", "tracking shot following the character", "slow pan across the scene"]


def split_scenes(story, max_scenes=30):
    """Split a story into scene-sized beats (one prompt each)."""
    story = (story or "").strip()
    if not story:
        return []
    # explicit line breaks win; otherwise split on sentences
    lines = [l.strip(" -•\t") for l in story.split("\n") if l.strip(" -•\t")]
    if len(lines) > 1:
        beats = lines
    else:
        beats = [s.strip() for s in re.split(r"(?<=[.!?।])\s+", story) if s.strip()]
    return beats[:max_scenes]


def build_prompts(story, style_text, characters, setting, seconds=10, max_scenes=30):
    """-> list of (n, prompt_text). Every prompt repeats the same style + character block verbatim,
    which is what actually keeps characters looking the same from clip to clip."""
    beats = split_scenes(story, max_scenes)
    char_block = ""
    if characters.strip():
        char_block = "CHARACTERS (keep these EXACTLY identical in every shot):\n" + characters.strip() + "\n\n"
    set_block = f"SETTING (unchanged across shots): {setting.strip()}\n\n" if setting.strip() else ""
    out = []
    for i, beat in enumerate(beats, 1):
        cam = CAMERAS[(i - 1) % len(CAMERAS)]
        p = (f"STYLE: {style_text}\n\n"
             f"{char_block}{set_block}"
             f"SHOT {i} — ACTION: {beat}\n\n"
             f"CAMERA: {cam}. DURATION: about {seconds} seconds. "
             f"Continuous single shot, no cuts, no on-screen text, no subtitles, no watermark.")
        out.append((i, p))
    return out


# ---------------- UI (called by app.py as a mode; NOT auto-run) ----------------
def render_mode():
    st.markdown("**Prompt Generator** — apni kahani ek baar daalo → **har clip ka prompt** ban jaayega, "
                "characters **locked** (har prompt me same description), taaki AI har shot me **same character** "
                "banaye. Phir bas copy-paste karte jao. Baar-baar prompt likhne ka kaam khatam. ✍️")

    story = st.text_area("📖 Kahani / scene list", height=200, key="ps_story",
                         placeholder="Poori kahani daalo — ya har line me ek scene likho:\n"
                                     "Ek buzurg captain jahaz ke deck par khada hai\n"
                                     "Do pirates seedhi par chadhte hain\n"
                                     "Ladka darr ke peeche hat-ta hai")

    chars = st.text_area("👥 Characters — ek baar likho, har prompt me repeat hoga", height=140, key="ps_chars",
                         placeholder="ELDER: 70-year-old man, white beard, white knitted cap, cream kurta, calm face\n"
                                     "BOY: 16-year-old, red bandana, torn beige shirt, scared expression\n"
                                     "PIRATE: muscular man, black headband, dark sleeveless vest, scar on cheek",
                         help="Jitna detail (umar, kapde, rang, chehra) utni acchi consistency.")

    c1, c2 = st.columns(2)
    with c1:
        style_name = st.selectbox("🎨 Style", list(STYLES.keys()), key="ps_style")
    with c2:
        seconds = st.slider("⏱️ Har clip kitne second", 5, 20, 10, 1, key="ps_sec")
    setting = st.text_input("🌍 Setting (har shot me same)", key="ps_set",
                            placeholder="rusty cargo ship deck, open sea, overcast evening light")

    if st.button("✨ Generate all prompts", type="primary", use_container_width=True, key="ps_go"):
        if not story.strip():
            st.error("Pehle kahani daalo."); return
        prompts = build_prompts(story, STYLES[style_name], chars, setting, seconds)
        if not prompts:
            st.error("Koi scene nahi mila — kahani thodi lambi likho."); return

        st.success(f"✅ {len(prompts)} prompts ready — har ek ko copy karke AI me paste karo.")
        st.caption(f"Total video ≈ {len(prompts) * seconds} seconds "
                   f"({len(prompts) * seconds / 60:.1f} min). Clips banne ke baad app ke "
                   "**Concat + Voiceover** mode se jod lena.")

        all_txt = "\n\n" + ("\n\n" + "=" * 60 + "\n\n").join(f"CLIP {n}\n\n{p}" for n, p in prompts)
        st.download_button("⬇️ Download all prompts (.txt)", all_txt.encode("utf-8"),
                           "veo_prompts.txt", "text/plain", use_container_width=True, key="ps_dl")

        for n, p in prompts:
            with st.expander(f"🎬 CLIP {n}", expanded=(n == 1)):
                st.code(p, language=None)   # code block = built-in copy button
