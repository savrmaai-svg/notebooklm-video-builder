# One Studio — the WHOLE pipeline on ONE page.
# Story in at the top, finished movie at the bottom. No hopping between modes:
#   1 story -> AI writes characters + scenes -> per-clip prompts (copy)
#   2 you paste each prompt into your AI video tool and download  (the only manual step)
#   3 clips are pulled straight out of Downloads, in order, nothing to upload
#   4 narration (auto-generated if you want) -> joined -> finished film, previewed right here
# Every step is always visible: if you already have clips from earlier, jump straight to 3 or 4.
import os, glob, time
import streamlit as st

import prompt_studio as PS
import clip_importer as CI
import concat_studio as CC
import faceless_studio as FS

VOICES = {
    "🇮🇳 Madhur — Hindi, male": "hi-IN-MadhurNeural",
    "🇮🇳 Swara — Hindi, female": "hi-IN-SwaraNeural",
    "🇮🇳 Prabhat — Indian English, male": "en-IN-PrabhatNeural",
    "🇮🇳 Neerja — Indian English, female": "en-IN-NeerjaNeural",
    "🇺🇸 Guy — US male, narrator": "en-US-GuyNeural",
    "🇬🇧 Ryan — UK male, documentary": "en-GB-RyanNeural",
}
AUDIO_EXT = (".mp3", ".m4a", ".wav", ".aac", ".ogg", ".flac")
DEST_DEFAULT = os.path.join(os.path.expanduser("~"), "Desktop", "my_story_clips")


def find_audio(folder, hours=24):
    now, out = time.time(), []
    for p in glob.glob(os.path.join(folder, "*")):
        if p.lower().endswith(AUDIO_EXT) and os.path.isfile(p):
            mt = os.path.getmtime(p)
            if not hours or (now - mt) <= hours * 3600:
                out.append((p, mt))
    out.sort(key=lambda x: -x[1])
    return [p for p, _ in out]


def folder_clips(folder):
    if not folder or not os.path.isdir(folder):
        return []
    return sorted(p for p in glob.glob(os.path.join(folder, "*"))
                  if p.lower().endswith((".mp4", ".mov", ".webm", ".mkv")))


def _step(n, title, done=False):
    st.markdown(f"### {'✅' if done else str(n) + '️⃣'}  {title}")


def render_mode():
    st.markdown("## 🎬 One Studio — kahani → poori movie")
    st.caption("Sab kuch **isi ek page pe**. Clips pehle se hain? Seedha neeche step 3 ya 4 pe chala ja.")

    ss = st.session_state
    ss.setdefault("sm_prompts", None)
    ss.setdefault("sm_clips", None)
    ss.setdefault("sm_imported", None)

    key = ""
    try:
        key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        pass

    # ------------------------------------------------------------ 1. STORY
    st.divider()
    _step(1, "Kahani daalo", done=bool(ss.sm_prompts))
    with st.expander("Kholo / band karo", expanded=not ss.sm_prompts):
        story = st.text_area("Poori kahani — ya bas 2-3 line ka idea", height=170, key="sm_story",
                             placeholder="Raat ke gyarah baje NH-44 par Arjun ka truck ja raha tha…")
        c1, c2 = st.columns(2)
        with c1:
            style_name = st.selectbox("🎨 Style", list(PS.STYLES.keys()), key="sm_style")
        with c2:
            seconds = st.slider("⏱️ Har clip kitne second", 5, 20, 10, 1, key="sm_sec")
        st.caption(f"Kitni bhi lambi kahani ho — **{PS.SHOTS_MIN}–{PS.SHOTS_MAX} clips** me poori aayegi, "
                   "clips **bina cut ke** jude rahenge.")
        if st.button("✨ Prompts banao", type="primary", use_container_width=True, key="sm_go1"):
            if not story.strip():
                st.error("Pehle kahani daalo.")
            else:
                chars, setting, src = "", "", story
                if key:
                    try:
                        with st.spinner("AI characters + scenes likh raha hai…"):
                            chars, setting, scenes = PS.ai_breakdown(story, style_name, PS.SHOTS_MIN, key)
                        src = "\n".join(scenes)
                    except Exception as e:
                        st.warning("AI se nahi hua, kahani seedha use kar raha hu. (" + str(e)[:120] + ")")
                else:
                    st.info("GEMINI_API_KEY nahi mili — AI assist band.")
                ss.sm_prompts = PS.build_prompts(src, PS.STYLES[style_name], chars, setting, seconds)
                ss.sm_chars = chars

    # ------------------------------------------------------------ 2. PROMPTS
    if ss.sm_prompts:
        n = len(ss.sm_prompts)
        st.divider()
        _step(2, f"{n} prompts — AI video tool me paste karo", done=bool(ss.sm_imported))
        st.caption("Har prompt copy → Gemini/Veo me paste → clip download. **Sirf yahi step haath se hai.**")
        if ss.get("sm_chars"):
            with st.expander("👥 Characters (har prompt me same)"):
                st.code(ss.sm_chars)
        all_txt = "\n\n".join(f"===== CLIP {i} =====\n\n{p}" for i, p in ss.sm_prompts)
        st.download_button("⬇️ Saare prompts (.txt)", all_txt.encode("utf-8"), "prompts.txt",
                           "text/plain", use_container_width=True, key="sm_dl")
        for i, p in ss.sm_prompts:
            with st.expander(f"🎬 CLIP {i} of {n}", expanded=(i == 1)):
                st.code(p, language=None)

    # ------------------------------------------------------------ 3. IMPORT  (always shown)
    st.divider()
    dest = st.session_state.get("sm_dest", DEST_DEFAULT)
    have = ss.sm_imported or folder_clips(dest)
    _step(3, "Clips uthao — Downloads se, kuch upload nahi", done=bool(have))
    with st.expander("Kholo / band karo", expanded=not have):
        i1, i2 = st.columns(2)
        with i1:
            hrs = st.number_input("🕒 Pichhle kitne ghante", 1, 720, 24, 1, key="sm_hrs")
        with i2:
            mx = st.number_input("⏱️ Max clip length (sec)", 5, 600, 30, 1, key="sm_mx")
        if st.button("🔍 Downloads scan karo", use_container_width=True, key="sm_scan"):
            ss.sm_clips = CI.scan(CI.DEFAULT_DIR, "", hrs, 0, mx)

        if ss.sm_clips:
            st.success(f"✅ {len(ss.sm_clips)} clips mile — purani se nayi (= story order)")
            keep, cols = [], st.columns(4)
            for i, f in enumerate(ss.sm_clips):
                with cols[i % 4]:
                    tp = CI.thumb(f["path"], os.path.join(CI.DEFAULT_DIR, f".thumb{i}.jpg"))
                    if tp:
                        st.image(tp, use_container_width=True)
                    if st.checkbox(f"**{i+1}.** {f['dur']:.0f}s", value=True, key=f"sm_k{i}"):
                        keep.append(f)
            st.text_input("📦 Folder", DEST_DEFAULT, key="sm_dest")
            if st.button(f"📦 {len(keep)} clips import karo", use_container_width=True, key="sm_imp"):
                if keep:
                    ss.sm_imported = CI.import_clips(keep, st.session_state.sm_dest)
                    st.success(f"✅ {len(ss.sm_imported)} clips ready")
                else:
                    st.error("Kam se kam ek clip chuno.")

    # ------------------------------------------------------------ 4. MOVIE  (always shown)
    st.divider()
    clips = ss.sm_imported or folder_clips(dest)
    _step(4, "Narration + movie banao")
    if not clips:
        st.info("Pehle step 3 se clips le aao (ya `my_story_clips` folder me daal do).")
        return

    st.success(f"🎞️ {len(clips)} clips taiyaar — {os.path.basename(os.path.dirname(clips[0]))} folder se")
    with st.expander("Clips dekh lo (preview)"):
        pc = st.columns(4)
        for i, c in enumerate(clips):
            with pc[i % 4]:
                tp = CI.thumb(c, os.path.join(os.path.dirname(c), f".p{i}.jpg"))
                if tp:
                    st.image(tp, caption=os.path.basename(c), use_container_width=True)

    nmode = st.radio("Narration kahan se?",
                     ["🎙️ Kahani se khud bana do (free)", "📁 Downloads se dhoondo", "⬆️ Khud daalo"],
                     horizontal=True, key="sm_nmode")
    npath, nup = "", None

    if nmode.startswith("🎙️"):
        v1, v2 = st.columns(2)
        with v1:
            voice = st.selectbox("Awaaz", list(VOICES.keys()), key="sm_voice")
        with v2:
            rate = st.select_slider("Speed", ["-20%", "-10%", "+0%", "+10%", "+20%"], "+0%", key="sm_rate")
        txt = st.text_area("Narration text (step 1 wali kahani — badal bhi sakte ho)",
                           value=ss.get("sm_story", ""), height=120, key="sm_ntext")
        if st.button("🎙️ Narration banao", use_container_width=True, key="sm_tts"):
            if not txt.strip():
                st.error("Text daalo.")
            else:
                try:
                    mp3 = os.path.join(os.path.dirname(clips[0]), "narration.mp3")
                    with st.spinner("Awaaz ban rahi hai…"):
                        FS.narrate(txt, VOICES[voice], mp3, rate=rate)
                    ss.sm_narr = mp3
                    st.success(f"✅ Narration ready ({FS._dur(mp3):.0f}s)")
                except Exception as e:
                    st.error("TTS error: " + str(e)[:200])
        if ss.get("sm_narr") and os.path.isfile(ss.sm_narr):
            st.audio(ss.sm_narr)
            npath = ss.sm_narr

    elif nmode.startswith("📁"):
        cand = find_audio(CI.DEFAULT_DIR, 24)
        if cand:
            pick = st.selectbox("Downloads me mile (naya sabse upar)", cand,
                                format_func=os.path.basename, key="sm_pick")
            if pick:
                st.audio(pick)
                npath = pick
        else:
            st.warning("Pichhle 24 ghante me koi audio nahi mili Downloads me.")
    else:
        npath = st.text_input("🎙️ Narration ka path", "", key="sm_npath")
        if not (npath and os.path.isfile(npath)):
            nup = st.file_uploader("…ya upload karo", key="sm_nup",
                                   type=["mp3", "m4a", "wav", "aac", "mp4", "mov", "mkv", "webm"])

    m1, m2 = st.columns(2)
    with m1:
        xf = st.slider("Crossfade (s)", 0.0, 1.5, 0.6, 0.1, key="sm_xf")
    with m2:
        smooth = st.checkbox("Smooth slow-motion", True, key="sm_smooth")

    if st.button("🎬 POORI MOVIE BANAO", type="primary", use_container_width=True, key="sm_make"):
        np_ = npath if (npath and os.path.isfile(npath)) else None
        if not np_ and nup:
            np_ = os.path.join(os.path.dirname(clips[0]), "narration" + os.path.splitext(nup.name)[1])
            with open(np_, "wb") as w:
                w.write(nup.getbuffer())
        if not np_:
            st.error("Pehle narration banao ya daalo.")
        else:
            out = os.path.join(os.path.dirname(clips[0]), "final_movie.mp4")
            box, logs = st.empty(), []
            def prog(m):
                logs.append(str(m)); box.code("\n".join(logs[-10:]))
            try:
                with st.spinner("Movie ban rahi hai… (thodi der lagegi)"):
                    CC.render(clips, np_, out, crossfade=xf, match=True, smooth=smooth, progress=prog)
                ss.sm_movie = out
            except Exception as e:
                st.error("Render error: " + str(e)[:400])

    if ss.get("sm_movie") and os.path.isfile(ss.sm_movie):
        st.divider()
        st.markdown("### 🎉 Movie ready — preview")
        st.video(ss.sm_movie)
        with open(ss.sm_movie, "rb") as f:
            st.download_button("⬇️ Download movie", f, "final_movie.mp4", "video/mp4",
                               use_container_width=True, key="sm_dlm")
