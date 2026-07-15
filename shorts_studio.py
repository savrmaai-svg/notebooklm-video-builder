# Short / Reel Maker — turn any content video into a 15-60s VERTICAL (9:16) short for
# YouTube Shorts / Instagram Reels / Facebook, with a "watch full video + SUBSCRIBE" CTA
# shown over the last 1-3 seconds. player-safe mp4.
import os, subprocess, tempfile
import streamlit as st

FF = r"C:\Users\Sameer\Downloads\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"
if not os.path.isfile(FF):
    import imageio_ffmpeg
    FF = imageio_ffmpeg.get_ffmpeg_exe()
FP = FF.replace("ffmpeg.exe", "ffprobe.exe")
if not os.path.isfile(FP):
    FP = "ffprobe"

_FONT = next((f for f in [r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\Arial.ttf",
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "NotoSansTelugu-Bold.ttf")]
              if os.path.isfile(f)), None)


def _dur(path):
    try:
        r = subprocess.run([FP, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                           capture_output=True, text=True)
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def _line(text, y, size, t0, color="white"):
    """One centered drawtext line, shown only from t0 onward (the CTA window)."""
    text = str(text).replace("\\", "").replace(":", "\\:").replace("'", "").replace("%", "")
    ff = ""
    if _FONT:
        _fp = _FONT.replace("\\", "/").replace(":", "\\:")     # ffmpeg drawtext needs C\:/path/font.ttf
        ff = f":fontfile='{_fp}'"
    return (f"drawtext=text='{text}':fontcolor={color}:fontsize={size}:x=(w-tw)/2:y={y}"
            f":borderw=4:bordercolor=black@0.85:enable='gte(t,{t0:.2f})'{ff}")


def make_short(src, out, start=0.0, length=30.0, line1="Poori kahani — FULL VIDEO ab dekho!",
               line2="LIKE   SHARE   SUBSCRIBE", cta_sec=3.0, fill="blur", progress=None):
    """src -> a `length`s vertical 9:16 short; a subscribe CTA is overlaid in the last `cta_sec` seconds."""
    log = progress or (lambda x: None)
    length = float(length); cta_sec = min(float(cta_sec), length)
    t0 = max(0.0, length - cta_sec)
    if fill == "crop":                                   # full-screen center crop
        base = "crop='min(iw,ih*9/16)':ih,scale=1080:1920,setsar=1"
    else:                                                # cinematic: video fit on a blurred vertical bg
        base = ("split[a][b];[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
                "boxblur=22:2[bg];[b]scale=1080:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1")
    cta = (f"drawbox=x=0:y=0:w=iw:h=ih:color=black@0.45:t=fill:enable='gte(t,{t0:.2f})',"
           + _line(line1, "h/2-160", 48, t0) + ","
           + _line(line2, "h/2-40", 46, t0) + ","
           + _line(">>  SUBSCRIBE  <<", "h/2+110", 72, t0, color="red"))
    vf = base + "," + cta
    log(f"making {length:.0f}s vertical short — CTA in last {cta_sec:.0f}s…")
    subprocess.run([FF, "-y", "-v", "error", "-ss", str(start), "-i", src, "-t", str(length),
                    "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out], check=True)
    log(f"DONE → {out}")
    return out


# ---------------- UI (called by app.py as a mode; NOT auto-run) ----------------
def render_mode():
    if "shortsdir" not in st.session_state:
        st.session_state.shortsdir = tempfile.mkdtemp(prefix="shorts_")
    D = st.session_state.shortsdir

    st.markdown("**Short / Reel Maker** — koi bhi video (ya apni full story) daalo → **vertical 9:16 short** "
                "(YouTube Shorts / Reels / Facebook) ban jaata hai; end ke last 1-3 sec me **'full video dekho + SUBSCRIBE' CTA**. Default 30 sec.")
    src = st.file_uploader("Video (jisse short banana hai)", type=["mp4", "mov", "webm", "mkv"], key="sh_src")

    c1, c2, c3 = st.columns(3)
    with c1: length = st.slider("Short length (sec)", 15, 60, 30, 1, key="sh_len")
    with c2: start = st.number_input("Start from (sec)", 0.0, 3600.0, 0.0, 1.0, key="sh_start")
    with c3: cta_sec = st.slider("CTA duration (sec)", 1.0, 5.0, 3.0, 0.5, key="sh_cta")
    fill = st.radio("Look", ["Cinematic (blurred bg)", "Full-screen crop"], horizontal=True, key="sh_fill")
    line1 = st.text_input("CTA line 1", "Poori kahani — FULL VIDEO ab dekho!", key="sh_l1")
    line2 = st.text_input("CTA line 2", "LIKE   SHARE   SUBSCRIBE", key="sh_l2")

    if src:
        p = os.path.join(D, "src.mp4")
        with open(p, "wb") as w: w.write(src.getbuffer())
        st.caption(f"Source = {_dur(p):.0f}s → short = {length}s vertical 9:16, CTA last {cta_sec:.0f}s.")

    if st.button("Generate short", type="primary", use_container_width=True, key="sh_gen"):
        if not src:
            st.error("Video daalo."); return
        p = os.path.join(D, "src.mp4"); out = os.path.join(D, "short.mp4")
        box = st.empty(); logs = []
        def prog(m): logs.append(str(m)); box.code("\n".join(logs[-10:]))
        try:
            with st.spinner("Vertical short + CTA banaya ja raha…"):
                make_short(p, out, start=start, length=length, line1=line1, line2=line2,
                           cta_sec=cta_sec, fill=("crop" if "crop" in fill else "blur"), progress=prog)
            st.success("✅ Short ready!")
            st.video(out)
            with open(out, "rb") as vf:
                st.download_button("⬇️ Download short", vf, "short.mp4", "video/mp4", key="sh_dl")
        except Exception as e:
            st.error("Error: " + repr(e)); st.exception(e)
