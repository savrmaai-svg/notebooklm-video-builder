# Faceless Video Studio — script (text) -> narrated faceless video.
# Buyer gives a SCRIPT; this makes: AI voice-over (edge-tts, FREE neural TTS) + matching
# stock footage (Pexels) + burned captions -> a finished landscape/vertical mp4. Built for the
# Fiverr "script -> faceless video" gig: paste script, pick a voice, get a video in minutes.
import os, re, asyncio, subprocess, tempfile, importlib
import streamlit as st

FF = r"C:\Users\Sameer\Downloads\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"
if not os.path.isfile(FF):
    import imageio_ffmpeg
    FF = imageio_ffmpeg.get_ffmpeg_exe()
FP = FF.replace("ffmpeg.exe", "ffprobe.exe")
if not os.path.isfile(FP):
    FP = "ffprobe"

# curated, natural-sounding English voices (edge-tts, free)
VOICES = {
    "Guy — US male, warm narrator": "en-US-GuyNeural",
    "Jenny — US female, friendly": "en-US-JennyNeural",
    "Aria — US female, expressive": "en-US-AriaNeural",
    "Eric — US male, calm deep": "en-US-EricNeural",
    "Ryan — UK male, documentary": "en-GB-RyanNeural",
    "Sonia — UK female, crisp": "en-GB-SoniaNeural",
    "Christopher — US male, news": "en-US-ChristopherNeural",
    "Natasha — AU female, bright": "en-AU-NatashaNeural",
}


def _dur(path):
    try:
        r = subprocess.run([FP, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                           capture_output=True, text=True)
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def narrate(script, voice, out_mp3, rate="+0%"):
    """edge-tts: script text -> narration mp3 (free neural TTS)."""
    async def _go():
        c = __import__("edge_tts").Communicate(script, voice, rate=rate)
        data = b""
        async for ch in c.stream():
            if ch["type"] == "audio":
                data += ch["data"]
        with open(out_mp3, "wb") as f:
            f.write(data)
    asyncio.run(_go())
    return out_mp3


def _caption_lines(script, max_words=8):
    sents = re.split(r"(?<=[.!?])\s+", script.strip())
    lines = []
    for s in sents:
        w = s.split()
        for i in range(0, len(w), max_words):
            chunk = " ".join(w[i:i + max_words]).strip()
            if chunk:
                lines.append(chunk)
    return lines


def _ts(x):
    h = int(x // 3600); m = int((x % 3600) // 60); s = x % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def build_srt(script, total_dur, out_srt):
    """Sentence captions, each given a time slice proportional to its length (reliable, no ASR)."""
    lines = _caption_lines(script)
    if not lines:
        open(out_srt, "w", encoding="utf-8").write("")
        return out_srt
    weights = [max(4, len(l)) for l in lines]
    tot = sum(weights)
    t = 0.0; blocks = []
    for i, (l, w) in enumerate(zip(lines, weights), 1):
        d = total_dur * w / tot
        start, end = t, t + d; t = end
        safe = l.replace("{", "(").replace("}", ")")
        blocks.append(f"{i}\n{_ts(start)} --> {_ts(end)}\n{safe}\n")
    open(out_srt, "w", encoding="utf-8").write("\n".join(blocks))
    return out_srt


def _norm(src, out, W, H, seg=5.0):
    """Scale/crop any clip to W x H, drop audio, trim to `seg`s for visual variety."""
    subprocess.run([FF, "-y", "-v", "error", "-i", src, "-t", str(seg), "-vf",
                    f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps=30,format=yuv420p",
                    "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", out], check=True)
    return out


def assemble(clips, mp3, srt, out, W=1920, H=1080, captions=True, progress=None):
    """Normalized stock clips (looped to audio length) + captions + narration -> finished mp4."""
    log = progress or (lambda x: None)
    work = tempfile.mkdtemp(prefix="faceless_")
    log("preparing stock clips…")
    norm = []
    for i, c in enumerate(clips):
        try:
            norm.append(_norm(c, os.path.join(work, f"n{i}.mp4"), W, H))
        except Exception:
            continue
    if not norm:
        raise RuntimeError("koi stock clip usable nahi mili")

    # concat the normalized clips into one silent reel
    listf = os.path.join(work, "list.txt")
    with open(listf, "w", encoding="utf-8") as f:
        for n in norm:
            f.write(f"file '{n.replace(chr(92), '/')}'\n")
    reel = os.path.join(work, "reel.mp4")
    subprocess.run([FF, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", listf,
                    "-c", "copy", reel], check=True)

    adur = _dur(mp3)
    # captions srt written into the work dir so ffmpeg gets a clean relative path (Windows-safe)
    subs = ""
    if captions and srt and os.path.getsize(srt) > 0:
        local_srt = os.path.join(work, "subs.srt")
        with open(srt, "r", encoding="utf-8") as a, open(local_srt, "w", encoding="utf-8") as b:
            b.write(a.read())
        fs = int(H * 0.045)
        style = (f"FontName=Arial,Fontsize={fs},Bold=1,PrimaryColour=&H00FFFFFF,"
                 f"OutlineColour=&H00000000,BorderStyle=1,Outline=3,Shadow=1,"
                 f"Alignment=2,MarginV={int(H*0.06)}")
        subs = f"subtitles=subs.srt:force_style='{style}'"

    log(f"assembling {adur:.0f}s faceless video (loop reel to narration)…")
    vf = subs if subs else "null"
    # loop the reel to cover the narration, trim to audio length, burn captions, lay narration
    subprocess.run([FF, "-y", "-v", "error", "-stream_loop", "-1", "-i", reel, "-i", mp3,
                    "-t", f"{adur:.3f}", "-vf", vf, "-map", "0:v", "-map", "1:a",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out],
                   check=True, cwd=work)
    log(f"DONE → {out}")
    return out


# ---------------- UI (called by app.py as a mode; NOT auto-run) ----------------
def render_mode():
    if "facelessdir" not in st.session_state:
        st.session_state.facelessdir = tempfile.mkdtemp(prefix="facelessui_")
    D = st.session_state.facelessdir

    st.markdown("**Faceless Video Studio** — buyer ka **script** paste karo → AI voice-over (free neural TTS) + "
                "matching **stock footage** (Pexels) + **captions** → finished faceless video. "
                "Fiverr *script → video* gig ke liye: script daalo, voice chuno, minutes me video ready.")

    script = st.text_area("Script (narration text)", height=200,
                          placeholder="Paste the buyer's full script here. Har sentence ek caption banega, "
                                      "aur narration isi text se bolega.")
    kw = st.text_area("Visual keywords (one per line — Pexels search)", height=90,
                      placeholder="ocean waves\ncargo ship\nstorm at sea\nsailor on deck\nsunset horizon\n"
                                  "(khaali chhodo to script se auto-keywords banenge)")

    c1, c2, c3 = st.columns(3)
    with c1:
        voice_label = st.selectbox("Voice", list(VOICES.keys()), key="fl_voice")
    with c2:
        rate = st.select_slider("Speed", ["-20%", "-10%", "+0%", "+10%", "+20%"], "+0%", key="fl_rate")
    with c3:
        orient = st.radio("Format", ["Landscape 16:9", "Vertical 9:16"], key="fl_orient")
    caps = st.checkbox("Burn captions", True, key="fl_caps")

    W, H = (1920, 1080) if "Landscape" in orient else (1080, 1920)

    if st.button("Generate faceless video", type="primary", use_container_width=True, key="fl_gen"):
        if not script.strip():
            st.error("Script daalo."); return
        pexels = ""
        try:
            pexels = str(st.secrets.get("PEXELS_API_KEY", "")).strip()
        except Exception:
            pexels = ""
        if not pexels:
            st.error("PEXELS_API_KEY .streamlit/secrets.toml me daalo (free key pexels.com/api se)."); return

        box = st.empty(); logs = []
        def prog(m): logs.append(str(m)); box.code("\n".join(logs[-12:]))
        try:
            mp3 = os.path.join(D, "narration.mp3")
            srt = os.path.join(D, "subs.srt")
            out = os.path.join(D, "faceless.mp4")
            with st.spinner("Voice-over + stock footage + captions… (thodी der)"):
                prog("generating AI voice-over (edge-tts)…")
                narrate(script, VOICES[voice_label], mp3, rate=rate)
                adur = _dur(mp3)
                prog(f"narration = {adur:.0f}s")
                build_srt(script, adur, srt)

                # stock footage: keywords box, else auto from script sentences
                topic = kw.strip() or "\n".join(_caption_lines(script, max_words=4)[:14])
                prog("fetching matching stock footage (Pexels)…")
                import api_fetcher
                importlib.reload(api_fetcher)
                need = max(8, int(adur / 5) + 6)
                clips = api_fetcher.fetch_and_download_videos(
                    topic=topic, destination_dir=D, pexels_api_key=pexels, limit=min(60, need))
                prog(f"{len(clips)} stock clips mili")
                assemble(clips, mp3, srt, out, W=W, H=H, captions=caps, progress=prog)

            st.success("✅ Faceless video ready!")
            st.video(out)
            cc1, cc2 = st.columns(2)
            with cc1:
                with open(out, "rb") as vf:
                    st.download_button("⬇️ Download video", vf, "faceless_video.mp4", "video/mp4", key="fl_dlv")
            with cc2:
                if caps and os.path.exists(srt):
                    with open(srt, "rb") as sf:
                        st.download_button("⬇️ Download captions (.srt)", sf, "captions.srt", "text/plain", key="fl_dls")
        except Exception as e:
            st.error("Error: " + repr(e)); st.exception(e)
