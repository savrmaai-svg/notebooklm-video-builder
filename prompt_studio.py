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


SHOTS_MIN, SHOTS_MAX = 12, 15      # every story lands here, however long it is


def shot_count(n_parts):
    """Fixed 12–15 shots: 12 for a normal story, up to 15 only if there's a lot of material."""
    if n_parts <= 24:
        return SHOTS_MIN
    return min(SHOTS_MAX, SHOTS_MIN + (n_parts - 24) // 12)


def split_scenes(story, n_shots=None):
    """Turn a story of ANY length into 12–15 beats covering the whole thing, ending included.
    Long stories get merged into balanced chunks; nothing is ever truncated."""
    story = (story or "").strip()
    if not story:
        return []
    lines = [l.strip(" -•\t") for l in story.split("\n") if l.strip(" -•\t")]
    if len(lines) > 1:
        parts = lines
    else:
        parts = [s.strip() for s in re.split(r"(?<=[.!?।])\s+", story) if s.strip()]
    if not parts:
        return []
    n = n_shots or shot_count(len(parts))
    if len(parts) <= n:
        return parts
    beats, per = [], len(parts) / float(n)
    for i in range(n):
        chunk = parts[int(round(i * per)):int(round((i + 1) * per))] or [parts[min(i, len(parts) - 1)]]
        beats.append(" ".join(chunk))
    return beats


def build_prompts(story, style_text, characters, setting, seconds=10, n_shots=None):
    """-> list of (n, prompt_text). Every prompt repeats the same style + character block verbatim
    (that's what keeps characters identical), and carries the previous/next beat so the clips join
    into one continuous film instead of looking like separate disconnected shots."""
    beats = split_scenes(story, n_shots)
    total = len(beats)
    char_block = ""
    if characters.strip():
        char_block = "CHARACTERS (keep these EXACTLY identical in every shot):\n" + characters.strip() + "\n\n"
    set_block = f"SETTING (unchanged across shots): {setting.strip()}\n\n" if setting.strip() else ""
    out = []
    for i, beat in enumerate(beats, 1):
        cam = CAMERAS[(i - 1) % len(CAMERAS)]
        # continuity: tell the model where we just came from and where we're going
        if i == 1:
            cont = (f"CONTINUITY: This is shot 1 of {total} — the opening of one continuous film. "
                    f"Establish the scene. It must flow directly into: \"{beats[1]}\"" if total > 1 else
                    f"CONTINUITY: Single continuous shot.")
        else:
            cont = (f"CONTINUITY: This is shot {i} of {total} of ONE continuous film. "
                    f"It picks up EXACTLY where the previous moment left off: \"{beats[i-2]}\". "
                    f"Same location continuity, same time of day, same lighting and colour grade as the "
                    f"previous shot — the viewer should feel no jump, no scene reset, no time skip.")
            if i < total:
                cont += f" It then leads into: \"{beats[i]}\""
        # shot number FIRST: AI tools name the downloaded file after the start of the prompt,
        # so the file lands as "Shot_03_of_12...mp4" — already numbered and sortable.
        p = (f"Shot {i:02d} of {total:02d}\n\n"
             f"STYLE: {style_text}\n\n"
             f"{char_block}{set_block}"
             f"{cont}\n\n"
             f"SHOT {i} of {total} — ACTION: {beat}\n\n"
             f"CAMERA: {cam}. DURATION: about {seconds} seconds. "
             f"ONE continuous unbroken take — no cuts, no jump cuts, no montage, no fade in or out, "
             f"no on-screen text, no subtitles, no titles, no watermark. "
             f"Start and end on a steady frame so it joins smoothly with the neighbouring shots.")
        out.append((i, p))
    return out


def _gemini(prompt, key, model="gemini-2.5-flash", timeout=90):
    """Plain text call to the Gemini API (FREE tier is enough — only video/Veo is blocked)."""
    import json, urllib.request
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                                 method="POST")
    d = json.load(urllib.request.urlopen(req, timeout=timeout))
    return d["candidates"][0]["content"]["parts"][0]["text"]


def ai_breakdown(idea, style_name, n_scenes, key):
    """Rough idea -> (character_bible, setting, scene_list). This is the thinking work, done for you."""
    import json, re as _re
    ask = (
        f"You are a film director planning an AI-generated {style_name} short film.\n"
        f"STORY (may be a short idea OR a long full story, in any language):\n{idea}\n\n"
        f"Turn it into EXACTLY {n_scenes} shots that cover the ENTIRE story — beginning, middle AND ending.\n"
        f"If the story is long, CONDENSE it: pick the {n_scenes} most important visual moments and merge the "
        f"rest into them. Never stop partway through the story. If it is short, expand it naturally.\n"
        f"Write the scene descriptions in ENGLISH (image models understand it best), even if the story is "
        f"in Hindi or another language.\n\n"
        f"Reply with ONLY valid JSON, no markdown fences:\n"
        '{"characters":[{"tag":"<NAME_IN_CAPS>","desc":"<age, build, hair, exact clothing and colours>"}],'
        '"setting":"<one line: location and lighting, same for every shot>",'
        '"scenes":["<shot 1 action in one vivid sentence>","<shot 2 action>","..."]}\n\n'
        "Rules:\n"
        "- The angle-bracket text above is a PLACEHOLDER showing the shape only. Never copy it. Every "
        "character, name and detail must come from THIS story — if the story is about a truck driver, "
        "the character is that truck driver, not anyone from an example.\n"
        "- List ONLY characters that actually appear in THIS story. Do not invent or carry over others.\n"
        "- Character descriptions must be concrete and visual (age, build, hair, exact clothing and colours) "
        "so an image model draws them identically every time.\n"
        "- Scenes must be visual actions, not dialogue or narration.\n"
        "- The shots form ONE continuous film: each shot must pick up exactly where the previous one "
        "ended, in the same location continuity, same time of day and same lighting. No time jumps, "
        "no scene resets, no cuts away and back.\n"
        "- Refer to characters by their tag so they stay consistent.\n"
        f"- Exactly {n_scenes} scenes, in story order."
    )
    raw = _gemini(ask, key).strip()
    raw = _re.sub(r"^```(?:json)?|```$", "", raw, flags=_re.M).strip()
    data = json.loads(raw)
    chars = "\n".join(f"{c.get('tag','CHAR')}: {c.get('desc','')}" for c in data.get("characters", []))
    return chars, data.get("setting", ""), data.get("scenes", [])


# ---------------- UI (called by app.py as a mode; NOT auto-run) ----------------
def render_mode():
    st.markdown("**Prompt Generator** — apni kahani ek baar daalo → **har clip ka prompt** ban jaayega, "
                "characters **locked** (har prompt me same description), taaki AI har shot me **same character** "
                "banaye. Phir bas copy-paste karte jao. Baar-baar prompt likhne ka kaam khatam. ✍️")

    # ---- AI assist: rough idea -> characters + setting + scenes (FREE text API) ----
    _key = ""
    try:
        _key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        pass
    with st.expander("✨ Sochne ka kaam AI se karwao (FREE) — idea daalo, scenes + characters khud ban jaayenge",
                     expanded=False):
        if not _key:
            st.info("Ye chalane ke liye `.streamlit/secrets.toml` me daalo:\n\n"
                    '`GEMINI_API_KEY = "your-key-here"`\n\n'
                    "Free tier kaafi hai — text ke liye paisa nahi lagta (sirf video/Veo blocked hai).")
        idea = st.text_area("💡 Bas idea likho (2-3 line)", height=90, key="ps_idea",
                            placeholder="Somali pirates ek cargo ship hijack karte hain. Ek buzurg captain "
                                        "shanti se unka saamna karta hai aur crew ko bachata hai.")
        if st.button("✨ AI se scenes + characters banwao", use_container_width=True, key="ps_ai"):
            if not _key:
                st.error("Pehle GEMINI_API_KEY secrets.toml me daalo.")
            elif not idea.strip():
                st.error("Idea to likho.")
            else:
                try:
                    with st.spinner("AI kahani tod raha hai…"):
                        ch, setg, scenes = ai_breakdown(idea, st.session_state.get("ps_style",
                                                        "Pixar-style 3D cartoon"), SHOTS_MIN, _key)
                    st.session_state.ps_story = "\n".join(scenes)
                    st.session_state.ps_chars = ch
                    st.session_state.ps_set = setg
                    st.success(f"✅ {len(scenes)} scenes + characters ban gaye — neeche bhar diye. Ab Generate dabao.")
                    st.rerun()
                except Exception as e:
                    st.error("AI error: " + str(e)[:300])

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
    st.caption(f"🎬 Kitni bhi lambi kahani ho — **poori story {SHOTS_MIN}–{SHOTS_MAX} clips** me aa jaayegi, "
               "aur clips **bina cut ke** ek doosre se jude rahenge.")
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
