# One Studio — the WHOLE pipeline on ONE page.
# Story in at the top, finished movie at the bottom. No hopping between modes:
#   1 story -> AI writes characters + scenes -> per-clip prompts (copy)
#   2 you paste each prompt into your AI video tool and download  (the only manual step)
#   3 clips are pulled straight out of Downloads, in order, nothing to upload
#   4 joined with narration -> finished film
import os
import streamlit as st

import prompt_studio as PS
import clip_importer as CI
import concat_studio as CC


def _step(n, title, done=False):
    st.markdown(f"### {'✅' if done else str(n) + '️⃣'}  {title}")


def render_mode():
    st.markdown("## 🎬 One Studio — kahani → poori movie")
    st.caption("Sab kuch **isi ek page pe**, upar se neeche. Kahin aur jaane ki zarurat nahi.")

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
    story = st.text_area("Poori kahani — ya bas 2-3 line ka idea", height=170, key="sm_story",
                         placeholder="Raat ke gyarah baje NH-44 par Arjun ka truck ja raha tha jisme 50 crore cash tha…")
    c1, c2 = st.columns(2)
    with c1:
        style_name = st.selectbox("🎨 Style", list(PS.STYLES.keys()), key="sm_style")
    with c2:
        seconds = st.slider("⏱️ Har clip kitne second", 5, 20, 10, 1, key="sm_sec")
    st.caption(f"Kitni bhi lambi kahani ho — **{PS.SHOTS_MIN}–{PS.SHOTS_MAX} clips** me poori aayegi, "
               "aur clips **bina cut ke** jude rahenge.")

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
                    st.warning("AI se nahi hua, teri kahani seedha use kar raha hu. (" + str(e)[:120] + ")")
            else:
                st.info("GEMINI_API_KEY nahi mili — AI assist band, kahani seedha use ho rahi hai.")
            ss.sm_prompts = PS.build_prompts(src, PS.STYLES[style_name], chars, setting, seconds)
            ss.sm_chars = chars

    if ss.sm_prompts:
        n = len(ss.sm_prompts)
        st.success(f"✅ {n} prompts ready — total ≈ {n*seconds}s ({n*seconds/60:.1f} min)")
        if ss.get("sm_chars"):
            with st.expander("👥 Characters (har prompt me same rahenge)"):
                st.code(ss.sm_chars)
        all_txt = "\n\n".join(f"===== CLIP {i} =====\n\n{p}" for i, p in ss.sm_prompts)
        st.download_button("⬇️ Saare prompts (.txt)", all_txt.encode("utf-8"), "prompts.txt",
                           "text/plain", use_container_width=True, key="sm_dl")

        # -------------------------------------------------------- 2. GENERATE
        st.divider()
        _step(2, "Prompts AI video tool me paste karo", done=bool(ss.sm_clips))
        st.caption("Har prompt copy → Gemini/Veo me paste → clip download. **Sirf yahi step haath se hai.** "
                   "Files khud Downloads me chali jaayengi.")
        for i, p in ss.sm_prompts:
            with st.expander(f"🎬 CLIP {i} of {n}", expanded=(i == 1)):
                st.code(p, language=None)

        # -------------------------------------------------------- 3. IMPORT
        st.divider()
        _step(3, "Clips uthao — Downloads se, kuch upload nahi", done=bool(ss.sm_imported))
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
            dest = st.text_input("📦 Folder", os.path.join(os.path.expanduser("~"), "Desktop", "my_story_clips"),
                                 key="sm_dest")
            if st.button(f"📦 {len(keep)} clips import karo", use_container_width=True, key="sm_imp"):
                if keep:
                    ss.sm_imported = CI.import_clips(keep, dest)
                    st.success(f"✅ {len(ss.sm_imported)} clips ready → {dest}")
                else:
                    st.error("Kam se kam ek clip chuno.")

        # -------------------------------------------------------- 4. MOVIE
        if ss.sm_imported:
            st.divider()
            _step(4, "Narration daalo aur movie banao")
            npath = st.text_input("🎙️ Narration file ka path", "", key="sm_npath",
                                  placeholder=r"C:\Users\Sameer\Downloads\narration.mp3")
            nup = None
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
                    np_ = os.path.join(os.path.dirname(ss.sm_imported[0]),
                                       "narration" + os.path.splitext(nup.name)[1])
                    with open(np_, "wb") as w:
                        w.write(nup.getbuffer())
                if not np_:
                    st.error("Narration daalo (path ya upload).")
                else:
                    out = os.path.join(os.path.dirname(ss.sm_imported[0]), "final_movie.mp4")
                    box, logs = st.empty(), []
                    def prog(m):
                        logs.append(str(m)); box.code("\n".join(logs[-10:]))
                    try:
                        with st.spinner("Movie ban rahi hai… (thodi der lagegi)"):
                            CC.render(ss.sm_imported, np_, out, crossfade=xf, match=True,
                                      smooth=smooth, progress=prog)
                        st.balloons()
                        st.success("🎉 Movie ready!")
                        st.video(out)
                        with open(out, "rb") as f:
                            st.download_button("⬇️ Download movie", f, "final_movie.mp4", "video/mp4",
                                               use_container_width=True, key="sm_dlm")
                    except Exception as e:
                        st.error("Render error: " + str(e)[:400])
