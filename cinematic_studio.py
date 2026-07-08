# Cinematic Slow-Mo mode for the NotebookLM Video Builder.
# Take a SHORT animated clip (e.g. a 10s Veo/Ghibli clip) -> smoothly slow it to a target length ->
# add a continuously-moving camera (Ken Burns), drifting fireflies/light particles, and a warm grade
# so it reads as a cinematic shot, NOT slow-motion -> optional voice-over -> player-safe mp4.
import os, sys, math, subprocess, tempfile
import numpy as np
import streamlit as st
try:
    import cv2
except Exception:
    cv2 = None

# full ffmpeg build (has minterpolate + libx264); fall back to imageio's
FF = r"C:\Users\Sameer\Downloads\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"
if not os.path.isfile(FF):
    import imageio_ffmpeg
    FF = imageio_ffmpeg.get_ffmpeg_exe()

W, H, FPS = 1280, 720, 25


def _clip_seconds(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return (n / fps) if fps else 0.0


def render(clip, voice, target_sec, out_path, fireflies=True, camera=True, grade=True,
           n_particles=42, progress=None):
    """clip -> slow to target_sec + camera + fireflies + grade + optional voice -> out_path (yuv420p)."""
    log = progress or (lambda x: None)
    if cv2 is None:
        raise RuntimeError("opencv (cv2) not installed")
    work = tempfile.mkdtemp(prefix="cine_")
    dur = _clip_seconds(clip) or 1.0
    factor = max(1.0, target_sec / dur)
    log(f"clip {dur:.1f}s → {target_sec:.0f}s  (slow {factor:.1f}x)")

    # 1) smooth slow-motion via motion-interpolation
    slow = os.path.join(work, "slow.mp4")
    subprocess.run([FF, "-y", "-v", "error", "-i", clip, "-vf",
                    f"setpts={factor:.4f}*PTS,minterpolate=fps={FPS}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
                    "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", slow], check=True)
    log("slow-motion ready — adding camera + fireflies + grade…")

    # 2) cinematic pass: Ken Burns + fireflies + warm grade, piped to ffmpeg (+ voice)
    R = 15
    yy, xx = np.mgrid[-R:R+1, -R:R+1]
    sprite = np.exp(-(xx**2 + yy**2) / (2 * (R/2.3)**2)).astype(np.float32)
    warm = np.array([1.0, 0.9, 0.55], np.float32)
    rng = np.random.RandomState(7); Np = int(n_particles)
    bx = rng.rand(Np)*W; by = rng.rand(Np)*H
    vx = (rng.rand(Np)-0.5)*6; vy = -(rng.rand(Np)*5+1.5)
    ph = rng.rand(Np)*6.28; tw = rng.rand(Np)*1.8+1.2; amp = rng.rand(Np)*9+4

    cmd = [FF, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"]
    if voice:
        cmd += ["-i", voice, "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "160k", "-shortest"]
    else:
        cmd += ["-an"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    cap = cv2.VideoCapture(slow); NF = int(target_sec * FPS)
    for f in range(NF):
        ok, fr = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ok, fr = cap.read()
            if not ok:
                break
        frame = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32)
        tt = f / FPS
        if camera:
            z = 1.05 + 0.05*math.sin(tt*0.20)
            cx = W/2 + 42*math.sin(tt*0.15); cy = H/2 + 24*math.cos(tt*0.11)
            vw, vh = W/z, H/z
            l = min(max(cx-vw/2, 0), W-vw); tp = min(max(cy-vh/2, 0), H-vh)
            frame = cv2.resize(frame[int(tp):int(tp+vh), int(l):int(l+vw)], (W, H), interpolation=cv2.INTER_LINEAR)
        if grade:
            frame[:, :, 0] *= 1.04; frame[:, :, 2] *= 0.97
        if fireflies:
            for i in range(Np):
                x = (bx[i] + vx[i]*tt + math.sin(tt*tw[i]+ph[i])*amp[i]) % W
                y = (by[i] + vy[i]*tt) % H
                b = 0.3 + 0.7*(0.5 + 0.5*math.sin(tt*tw[i]*2 + ph[i]))
                xi, yi = int(x), int(y)
                x0, x1 = max(0, xi-R), min(W, xi+R+1); y0, y1 = max(0, yi-R), min(H, yi+R+1)
                sp = sprite[y0-(yi-R):y0-(yi-R)+(y1-y0), x0-(xi-R):x0-(xi-R)+(x1-x0)]
                frame[y0:y1, x0:x1] += sp[:, :, None] * warm * (b*110)
        try:
            proc.stdin.write(np.clip(frame, 0, 255).astype(np.uint8).tobytes())
        except (BrokenPipeError, OSError):
            break
        if f % (FPS*10) == 0:
            log(f"  rendering… {f//FPS}s / {target_sec:.0f}s")
    try: proc.stdin.close()
    except Exception: pass
    proc.wait(); cap.release()
    log(f"DONE → {out_path}")
    return out_path


# ---------------- UI (called by app.py as a mode; NOT auto-run) ----------------
def render_mode():
    if "cinedir" not in st.session_state:
        st.session_state.cinedir = tempfile.mkdtemp(prefix="cinestudio_")
    D = st.session_state.cinedir

    st.markdown("**Cinematic Slow-Mo** — ek chhoti animated clip (jaise 10s Veo/Ghibli) daalo → smoothly slow karke long banata hai + moving camera + magical fireflies + warm grade, taaki wo *slow-motion* na lage, *cinematic* lage. Voice optional.")
    clip = st.file_uploader("Animated clip (mp4)", type=["mp4", "mov", "webm"], key="cine_clip")
    voice = st.file_uploader("Voice-over (optional) — mp3/m4a/wav/mp4", type=["mp3", "m4a", "wav", "mp4"], key="cine_voice")

    c1, c2 = st.columns(2)
    with c1:
        target_min = st.slider("Target length (minutes)", 0.5, 5.0, 3.0, 0.5, key="cine_len")
    with c2:
        n_part = st.slider("Fireflies / particles", 0, 90, 42, key="cine_part")
    o1, o2, o3 = st.columns(3)
    with o1: cam = st.checkbox("Moving camera", True, key="cine_cam")
    with o2: ff = st.checkbox("Fireflies", True, key="cine_ff")
    with o3: gr = st.checkbox("Warm grade", True, key="cine_gr")

    if clip:
        cp = os.path.join(D, "in_clip.mp4")
        with open(cp, "wb") as w: w.write(clip.getbuffer())
        st.video(cp)
        dur = _clip_seconds(cp) if cv2 else 0
        if dur:
            st.caption(f"Clip = {dur:.1f}s → target {target_min:.1f} min  (slow ~{max(1, (target_min*60)/dur):.0f}x). "
                       + ("⚠️ bahut zyada slow — motion halkी lagegi." if (target_min*60)/max(dur,0.1) > 12 else ""))

    if st.button("Generate cinematic video", type="primary", use_container_width=True, key="cine_gen"):
        if not clip:
            st.error("Pehle ek animated clip daalo."); return
        cp = os.path.join(D, "in_clip.mp4")
        vp = None
        if voice:
            vp = os.path.join(D, "voice" + os.path.splitext(voice.name)[1])
            with open(vp, "wb") as w: w.write(voice.getbuffer())
        out = os.path.join(D, "cinematic.mp4"); box = st.empty(); logs = []
        def prog(m): logs.append(str(m)); box.code("\n".join(logs[-14:]))
        try:
            with st.spinner("Slow-mo + camera + fireflies + grade… (thodी der)"):
                render(cp, vp, target_min*60, out, fireflies=ff, camera=cam, grade=gr, n_particles=n_part, progress=prog)
            st.success("✅ Cinematic video ready!")
            st.video(out)
            with open(out, "rb") as vf:
                st.download_button("⬇️ Download", vf, "cinematic.mp4", "video/mp4", key="cine_dl")
        except Exception as e:
            st.error("Render error: " + repr(e)); st.exception(e)
