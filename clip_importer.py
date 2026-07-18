# Clip Importer — pull AI-generated clips out of a messy Downloads folder, in story order.
# You generate a clip, download it, generate the next... so DOWNLOAD TIME = STORY ORDER.
# This finds those files among hundreds of others, shows a thumbnail of each so you can verify,
# and copies them out as clip01.mp4, clip02.mp4 ... ready for Concat + Voiceover.
import os, glob, shutil, time, subprocess, tempfile
import streamlit as st

FF = r"C:\Users\Sameer\Downloads\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"
if not os.path.isfile(FF):
    import imageio_ffmpeg
    FF = imageio_ffmpeg.get_ffmpeg_exe()
FP = FF.replace("ffmpeg.exe", "ffprobe.exe")
if not os.path.isfile(FP):
    FP = "ffprobe"

VID_EXT = (".mp4", ".mov", ".webm", ".mkv", ".m4v")
DEFAULT_DIR = os.path.join(os.path.expanduser("~"), "Downloads")


def _dur(p):
    try:
        r = subprocess.run([FP, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                           capture_output=True, text=True)
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def scan(folder, name_filter="", hours=0, min_s=0.0, max_s=0.0):
    """Find candidate clips, oldest first (= the order they were generated in)."""
    out = []
    now = time.time()
    for p in glob.glob(os.path.join(folder, "*")):
        if not p.lower().endswith(VID_EXT) or not os.path.isfile(p):
            continue
        base = os.path.basename(p)
        if name_filter and name_filter.lower() not in base.lower():
            continue
        mt = os.path.getmtime(p)
        if hours and (now - mt) > hours * 3600:
            continue
        d = _dur(p)
        if min_s and d < min_s:
            continue
        if max_s and d > max_s:
            continue
        out.append({"path": p, "name": base, "mtime": mt, "dur": d,
                    "mb": os.path.getsize(p) / 1048576})
    out.sort(key=lambda x: x["mtime"])          # oldest first = story order
    return out


def thumb(src, out_png, at=1.0):
    subprocess.run([FF, "-y", "-v", "error", "-ss", str(at), "-i", src,
                    "-vf", "scale=240:-1", "-frames:v", "1", out_png],
                   capture_output=True)
    return out_png if os.path.isfile(out_png) else None


def import_clips(items, dest_dir):
    """Copy the chosen clips out as clip01.mp4, clip02.mp4 … in the given order."""
    os.makedirs(dest_dir, exist_ok=True)
    made = []
    for i, it in enumerate(items, 1):
        ext = os.path.splitext(it["path"])[1].lower()
        dst = os.path.join(dest_dir, f"clip{i:02d}{ext}")
        shutil.copy2(it["path"], dst)
        made.append(dst)
    return made


# ---------------- UI (called by app.py as a mode; NOT auto-run) ----------------
def render_mode():
    if "impdir" not in st.session_state:
        st.session_state.impdir = tempfile.mkdtemp(prefix="clipimp_")
    T = st.session_state.impdir

    st.markdown("**Clip Importer** — Downloads folder me sau files ke beech tere AI clips dabe pade hain, "
                "naam bhi ulte-pulte (`... (12).mp4`). Ye unhe dhoondh ke **download ke time ke hisaab se "
                "(= teri story ka order)** laga deta hai, thumbnail dikhata hai, aur "
                "**clip01, clip02…** naam se ek saaf folder me nikaal deta hai. 📦")

    folder = st.text_input("📁 Folder", DEFAULT_DIR, key="ci_dir")
    c1, c2, c3 = st.columns(3)
    with c1:
        nf = st.text_input("🔎 Naam me ye ho", "Create_the_EXACT", key="ci_nf",
                           help="Gemini clips ka common naam. Khaali chhodo to sab dikhenge.")
    with c2:
        hrs = st.number_input("🕒 Pichhle kitne ghante", 0, 720, 0, 1, key="ci_hrs",
                              help="0 = time ki koi limit nahi")
    with c3:
        mx = st.number_input("⏱️ Max length (sec)", 0, 600, 30, 1, key="ci_mx",
                             help="AI clips chhote hote hain (~10s). Isse lambi files chhod deta hai. 0 = sab")

    if st.button("🔍 Scan folder", use_container_width=True, key="ci_scan"):
        if not os.path.isdir(folder):
            st.error("Folder nahi mila.")
        else:
            with st.spinner("Files check ho rahi hain…"):
                st.session_state.ci_found = scan(folder, nf, hrs, 0, mx)
            st.rerun()

    found = st.session_state.get("ci_found")
    if not found:
        st.info("Scan dabao — phir clips yahan order me dikhenge.")
        return

    st.success(f"✅ {len(found)} clips mile — **purani se nayi** (yehi teri story ka order hai).")
    total = sum(f["dur"] for f in found)
    st.caption(f"Total ≈ {total:.0f}s ({total/60:.1f} min)")

    st.markdown("Har clip ka **thumbnail** dekh ke tick/untick karo — jo nahi chahiye hata do:")
    keep = []
    cols = st.columns(4)
    for i, f in enumerate(found):
        with cols[i % 4]:
            tp = thumb(f["path"], os.path.join(T, f"t{i}.jpg"))
            if tp:
                st.image(tp, use_container_width=True)
            ok = st.checkbox(f"**{i+1}.** {f['dur']:.0f}s", value=True, key=f"ci_k{i}")
            st.caption(f"{time.strftime('%d %b %H:%M', time.localtime(f['mtime']))} · {f['mb']:.1f}MB")
            if ok:
                keep.append(f)

    st.divider()
    dest = st.text_input("📦 Kahan nikalna hai",
                         os.path.join(os.path.expanduser("~"), "Desktop", "my_story_clips"), key="ci_dest")
    if st.button(f"📦 Import {len(keep)} clips (clip01, clip02 …)", type="primary",
                 use_container_width=True, key="ci_go"):
        if not keep:
            st.error("Kam se kam ek clip chuno."); return
        try:
            made = import_clips(keep, dest)
            st.success(f"✅ {len(made)} clips import ho gaye → `{dest}`")
            st.code("\n".join(os.path.basename(m) for m in made))
            st.info("Ab **Concat + Voiceover** mode me jao, ye clips order me upload karo, "
                    "narration daalo → poori movie ban jaayegi. 🎬")
        except Exception as e:
            st.error("Import error: " + str(e))
