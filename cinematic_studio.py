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


def _cine_pass(slow, seg_out, seg_sec, seed, fireflies, camera, grade, leaves, n_particles, log):
    """One slowed clip -> add Ken Burns camera + fireflies + leaves sway + warm grade -> seg_out (no audio)."""
    R = 15
    yy, xx = np.mgrid[-R:R+1, -R:R+1]
    sprite = np.exp(-(xx**2 + yy**2) / (2 * (R/2.3)**2)).astype(np.float32)
    warm = np.array([1.0, 0.9, 0.55], np.float32)
    rng = np.random.RandomState(7 + seed); Np = int(n_particles)          # per-clip firefly seed
    bx = rng.rand(Np)*W; by = rng.rand(Np)*H
    vx = (rng.rand(Np)-0.5)*6; vy = -(rng.rand(Np)*5+1.5)
    ph = rng.rand(Np)*6.28; tw = rng.rand(Np)*1.8+1.2; amp = rng.rand(Np)*9+4
    mgx, mgy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))   # for leaves sway
    cmd = [FF, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", seg_out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(slow); NF = int(seg_sec * FPS)
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
        if leaves:
            disp = (2.4 * np.sin(mgy/26.0 + tt*1.7) * (1.0 - mgy/H*0.65)).astype(np.float32)
            frame = cv2.remap(frame, mgx + disp, mgy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
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
            log(f"    …{f//FPS}s / {seg_sec:.0f}s")
    try: proc.stdin.close()
    except Exception: pass
    proc.wait(); cap.release()


def render(clips, voice, factor, out_path, fireflies=True, camera=True, grade=True,
           leaves=True, n_particles=42, progress=None):
    """ONE or MANY short clips -> each GENTLY slowed by `factor` (NOT stretched to a huge target, so it
    reads as a cinematic shot, never extreme slow-mo) + camera/fireflies/leaves/grade -> joined into one
    story video (+ optional voice) -> out_path. Total length ≈ sum(clip_durations) × factor."""
    log = progress or (lambda x: None)
    if cv2 is None:
        raise RuntimeError("opencv (cv2) not installed")
    if isinstance(clips, str):
        clips = [clips]
    factor = max(1.0, min(3.0, float(factor)))          # cinematic slow, hard-capped so it never over-slows
    work = tempfile.mkdtemp(prefix="cine_")
    segs = []
    for ci, clip in enumerate(clips):
        dur = _clip_seconds(clip) or 1.0
        seg_sec = dur * factor
        log(f"clip {ci+1}/{len(clips)}: {dur:.1f}s → {seg_sec:.1f}s  (gentle slow {factor:.2f}x)")
        slow = os.path.join(work, f"slow{ci}.mp4")      # 1) smooth motion-interpolated slow
        subprocess.run([FF, "-y", "-v", "error", "-i", clip, "-vf",
                        f"setpts={factor:.4f}*PTS,minterpolate=fps={FPS}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
                        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", slow], check=True)
        seg = os.path.join(work, f"seg{ci}.mp4")        # 2) cinematic pass
        _cine_pass(slow, seg, seg_sec, ci, fireflies, camera, grade, leaves, n_particles, log)
        segs.append(seg)
    # 3) join the story clips
    if len(segs) == 1:
        joined = segs[0]
    else:
        log("joining clips into one story…")
        joined = os.path.join(work, "joined.mp4")
        lst = os.path.join(work, "list.txt")
        open(lst, "w").write("\n".join(f"file '{s.replace(chr(92), '/')}'" for s in segs))
        subprocess.run([FF, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
                        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", joined], check=True)
    # 4) optional voice-over, then finalise
    if voice:
        subprocess.run([FF, "-y", "-v", "error", "-i", joined, "-i", voice, "-map", "0:v", "-map", "1:a",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", out_path], check=True)
    else:
        subprocess.run([FF, "-y", "-v", "error", "-i", joined, "-c", "copy", "-movflags", "+faststart", out_path], check=True)
    log(f"DONE → {out_path}")
    return out_path


# ---------------- UI (called by app.py as a mode; NOT auto-run) ----------------
def render_mode():
    if "cinedir" not in st.session_state:
        st.session_state.cinedir = tempfile.mkdtemp(prefix="cinestudio_")
    D = st.session_state.cinedir

    st.markdown("**Cinematic Slow-Mo** — story ke **ek ya zyada clips** (jaise 3-4 chhoti Veo/Ghibli clips) daalo → har clip ko **halka-sa (cinematic) slow** karke, moving camera + fireflies + warm grade ke saath ek **story video** me jod deta hai. *Bahut* slow-mo nahi — bas cinematic feel. Voice optional.")
    clips = st.file_uploader("Animated clips (mp4) — ek ya zyada, story ke order me", type=["mp4", "mov", "webm"],
                             accept_multiple_files=True, key="cine_clips")
    voice = st.file_uploader("Voice-over (optional) — mp3/m4a/wav/mp4", type=["mp3", "m4a", "wav", "mp4"], key="cine_voice")

    c1, c2 = st.columns(2)
    with c1:
        slow = st.slider("Slow-mo strength (x)", 1.0, 3.0, 1.75, 0.25, key="cine_slow",
                         help="1x = normal · 1.75x = gentle cinematic (recommended) · 3x = max. Itna hi slow — extreme slow-mo nahi.")
    with c2:
        n_part = st.slider("Fireflies / particles", 0, 90, 42, key="cine_part")
    o1, o2, o3, o4 = st.columns(4)
    with o1: cam = st.checkbox("Moving camera", True, key="cine_cam")
    with o2: ff = st.checkbox("Fireflies", True, key="cine_ff")
    with o3: lv = st.checkbox("Leaves sway", True, key="cine_lv")
    with o4: gr = st.checkbox("Warm grade", True, key="cine_gr")

    saved = []
    if clips:
        total_in = 0.0
        for i, c in enumerate(clips):
            cp = os.path.join(D, f"in_clip{i}.mp4")
            with open(cp, "wb") as w: w.write(c.getbuffer())
            saved.append(cp); total_in += (_clip_seconds(cp) if cv2 else 0)
        st.video(saved[0])
        if total_in:
            st.caption(f"{len(saved)} clip(s), total {total_in:.1f}s → output ≈ **{total_in*slow:.0f}s** "
                       f"(~{total_in*slow/60:.1f} min) at {slow:.2f}x. Gentle cinematic slow — extreme slow-mo nahi.")

    if st.button("Generate cinematic video", type="primary", use_container_width=True, key="cine_gen"):
        if not saved:
            st.error("Pehle kam-se-kam ek animated clip daalo."); return
        vp = None
        if voice:
            vp = os.path.join(D, "voice" + os.path.splitext(voice.name)[1])
            with open(vp, "wb") as w: w.write(voice.getbuffer())
        out = os.path.join(D, "cinematic.mp4"); box = st.empty(); logs = []
        def prog(m): logs.append(str(m)); box.code("\n".join(logs[-14:]))
        try:
            with st.spinner("Gentle slow + camera + fireflies + grade… (thodी der)"):
                render(saved, vp, slow, out, fireflies=ff, camera=cam, grade=gr, leaves=lv, n_particles=n_part, progress=prog)
            st.success("✅ Cinematic story video ready!")
            st.video(out)
            with open(out, "rb") as vf:
                st.download_button("⬇️ Download", vf, "cinematic.mp4", "video/mp4", key="cine_dl")
        except Exception as e:
            st.error("Render error: " + repr(e)); st.exception(e)
