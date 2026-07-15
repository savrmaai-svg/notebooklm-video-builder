# Concat + Voiceover mode for the NotebookLM Video Builder.
# Many story clips (in order) -> joined seamlessly (crossfades) -> smoothly slowed to MATCH the
# NotebookLM narration length (so each scene spans its part of the story, video never races ahead of
# the audio) -> narration laid on top as a VOICE-OVER (NO lip-sync) -> player-safe mp4.
import os, subprocess, tempfile
import streamlit as st

FF = r"C:\Users\Sameer\Downloads\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"
if not os.path.isfile(FF):
    import imageio_ffmpeg
    FF = imageio_ffmpeg.get_ffmpeg_exe()
FP = FF.replace("ffmpeg.exe", "ffprobe.exe")
if not os.path.isfile(FP):
    FP = "ffprobe"
W, H, FPS = 1280, 720, 24


def _dur(path):
    try:
        r = subprocess.run([FP, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                           capture_output=True, text=True)
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def _norm(src, out):
    """Scale/pad any clip to W x H @ FPS and drop its audio (we use the narration instead)."""
    subprocess.run([FF, "-y", "-v", "error", "-i", src, "-vf",
                    f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
                    f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps={FPS},format=yuv420p",
                    "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", out], check=True)
    return out


def render(clips, narration, out_path, crossfade=0.6, match=True, factor=2.0, smooth=True, progress=None):
    """clips (list, in order) + narration -> crossfaded, slowed-to-match, voiced-over video -> out_path."""
    log = progress or (lambda x: None)
    work = tempfile.mkdtemp(prefix="concat_")
    if isinstance(clips, str):
        clips = [clips]

    # 1) normalize every clip (uniform size/fps, no audio)
    log("normalizing clips…")
    norm = [_norm(c, os.path.join(work, f"n{i}.mp4")) for i, c in enumerate(clips)]

    # 2) join with crossfades (video only)
    concat = os.path.join(work, "concat.mp4")
    if len(norm) == 1:
        subprocess.run([FF, "-y", "-v", "error", "-i", norm[0], "-c", "copy", concat], check=True)
    else:
        log("joining clips with crossfades…")
        durs = [_dur(n) for n in norm]
        inputs = []
        for n in norm:
            inputs += ["-i", n]
        fc, prev, off = "", "0:v", 0.0
        for i in range(1, len(norm)):
            off += durs[i - 1] - crossfade
            lbl = f"v{i}"
            fc += f"[{prev}][{i}:v]xfade=transition=fade:duration={crossfade}:offset={off:.4f}[{lbl}];"
            prev = lbl
        subprocess.run([FF, "-y", "-v", "error"] + inputs + ["-filter_complex", fc.rstrip(";"),
                        "-map", f"[{prev}]", "-an", "-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "20", "-pix_fmt", "yuv420p", concat], check=True)

    cdur = _dur(concat) or 1.0
    ndur = _dur(narration)
    if match and ndur > 0:                       # stretch clips to span the WHOLE narration
        factor = ndur / cdur
    factor = max(1.0, float(factor))
    tgt = cdur * factor
    log(f"clips {cdur:.1f}s → {tgt:.1f}s  (slow {factor:.2f}x; narration {ndur:.1f}s)")

    # 3) smooth slow-motion (motion-interpolated) or plain
    slow = os.path.join(work, "slow.mp4")
    vf = f"setpts={factor:.4f}*PTS"
    if smooth:
        vf += f",minterpolate=fps={FPS}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
    log("slowing (smooth motion)…" if smooth else "slowing…")
    subprocess.run([FF, "-y", "-v", "error", "-i", concat, "-vf", vf, "-an",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", slow], check=True)

    # 4) lay the narration on top as a voice-over (NO lip-sync); both start at 0, trim to shorter
    log("adding narration voice-over…")
    subprocess.run([FF, "-y", "-v", "error", "-i", slow, "-i", narration, "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", out_path], check=True)
    log(f"DONE → {out_path}")
    return out_path


# ---------------- UI (called by app.py as a mode; NOT auto-run) ----------------
def render_mode():
    if "concatdir" not in st.session_state:
        st.session_state.concatdir = tempfile.mkdtemp(prefix="concatstudio_")
    D = st.session_state.concatdir

    st.markdown("**Concat + Voiceover** — story ke **clips** (order me) + **NotebookLM narration** daalo → "
                "clips seamless jud jaate hain (crossfade), narration ki poori length tak **smoothly slow** ho ke, "
                "narration **voice-over** (no lip-sync) ban jaata hai — har scene apni story ke part pe, video aur audio saath.")
    clips = st.file_uploader("Story clips — order me (mp4)", type=["mp4", "mov", "webm"],
                             accept_multiple_files=True, key="cc_clips")
    narr = st.file_uploader("NotebookLM narration (voice) — mp4/mp3/m4a/wav", key="cc_narr",
                            type=["mp4", "mov", "mkv", "webm", "mp3", "m4a", "wav", "aac"])

    c1, c2, c3 = st.columns(3)
    with c1:
        match = st.checkbox("Match narration length", True, key="cc_match",
                            help="ON = clips ko narration ki poori length tak slow (scenes story ke saath chalte hain). OFF = neeche wala fixed slow use hoga.")
    with c2:
        factor = st.slider("Slow (x) — if not matching", 1.0, 3.0, 2.0, 0.25, key="cc_factor", disabled=match)
    with c3:
        xf = st.slider("Crossfade (s)", 0.0, 1.5, 0.6, 0.1, key="cc_xf")
    smooth = st.checkbox("Smooth slow-motion (best, thoda slow render)", True, key="cc_smooth")

    saved = []
    if clips:
        for i, c in enumerate(clips):
            p = os.path.join(D, f"clip{i}.mp4")
            with open(p, "wb") as w: w.write(c.getbuffer())
            saved.append(p)
    npath = None
    if narr:
        npath = os.path.join(D, "narration" + os.path.splitext(narr.name)[1])
        with open(npath, "wb") as w: w.write(narr.getbuffer())

    if saved and npath:
        cdur = sum(_dur(p) for p in saved)
        ndur = _dur(npath)
        eff = (ndur / cdur) if (match and cdur) else factor
        st.caption(f"{len(saved)} clip(s) = {cdur:.0f}s · narration = {ndur:.0f}s → "
                   f"slow **{eff:.1f}x** → output ≈ {(cdur*eff)/60:.1f} min.")
        if eff > 4:
            st.warning(f"⚠️ {eff:.1f}x = bahut slow (clips kam hain narration ke liye). Behtar match ke liye "
                       f"~{int(ndur/8)}+ clips daalo (har ~8s narration ke liye 1 clip), ya chhota narration use karo.")

    if st.button("Generate narrated video", type="primary", use_container_width=True, key="cc_gen"):
        if not saved:
            st.error("Story clips daalo (order me)."); return
        if not npath:
            st.error("NotebookLM narration (voice) daalo."); return
        out = os.path.join(D, "concat_voice.mp4"); box = st.empty(); logs = []
        def prog(m): logs.append(str(m)); box.code("\n".join(logs[-14:]))
        try:
            with st.spinner("Join + slow-to-match + voice-over… (thodी der)"):
                render(saved, npath, out, crossfade=xf, match=match, factor=factor, smooth=smooth, progress=prog)
            st.success("✅ Narrated video ready!")
            st.video(out)
            with open(out, "rb") as vf:
                st.download_button("⬇️ Download", vf, "concat_voice.mp4", "video/mp4", key="cc_dl")
        except Exception as e:
            st.error("Render error: " + repr(e)); st.exception(e)
