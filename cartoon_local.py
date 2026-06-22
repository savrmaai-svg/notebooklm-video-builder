# Self-contained LOCAL cartoon-episode pipeline (GPU recommended). Wired into app.py's
# "2D Cartoon Episode (Lip Sync)" mode. Heavy deps (faster-whisper, torch, rembg, cv2, scipy,
# librosa) are imported lazily INSIDE render() so the lightweight Streamlit Cloud app still imports
# app.py fine -- this module only runs when the user generates a cartoon episode on a local machine.
#
# Pipeline: upload audio -> transcribe (whisper) -> ~6.5s blocks -> per-scene environment + clean
# flat-2D prompt + emotion via Pollinations-text (no Claude agents needed) -> recurring host
# narrators with face-WARP viseme lip-sync in a cartoon studio -> story cutaway scenes (Pollinations
# image) -> scene-aware ambient SFX + mood music -> burned Hindi captions -> mux the user's audio.
import os, io, json, math, time, shutil, subprocess, urllib.parse, re
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets"
HEAVY_IMPORT_HINT = ("Cartoon mode ke liye local dependencies chahiye. Install: "
                     "pip install -r requirements-cartoon-local.txt  (GPU recommended).")


def _av_stub():
    """faster-whisper imports PyAV; on some Windows boxes the av DLL is blocked. Stub it so the
    import succeeds, then feed whisper an ffmpeg-decoded numpy array (av is never actually used)."""
    import sys, types
    if "av" in sys.modules:
        return
    core = types.ModuleType("av._core"); core.time_base = 1; core.library_versions = {}; core.ffmpeg_version_info = ""
    av = types.ModuleType("av"); av._core = core
    ad = types.ModuleType("av.audio"); rs = types.ModuleType("av.audio.resampler"); rs.AudioResampler = object; ad.resampler = rs
    sys.modules.update({"av": av, "av._core": core, "av.audio": ad, "av.audio.resampler": rs})


def _ptext(prompt, timeout=70, tries=3):
    import requests
    for _ in range(tries):
        try:
            r = requests.get("https://text.pollinations.ai/" + urllib.parse.quote(prompt), timeout=timeout)
            if r.status_code == 200 and len(r.text) > 2:
                return r.text.strip()
        except Exception:
            pass
        time.sleep(3)
    return ""


def _pjson(prompt):
    txt = _ptext(prompt)
    m = re.search(r"\{.*\}", txt, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {}


VIS_OPEN = ("lip_bot", "jaw", "lip_top", "mouth_L", "mouth_R")
MAT = set("ािीुूृेैोौंँः्"); LAB = set("मबपभफ")


def _aksharas(t):
    out = []; cur = ""
    for ch in t:
        if ch == " ":
            if cur: out.append(cur); cur = ""
            out.append(" "); continue
        if ch in MAT: cur += ch
        else:
            if cur: out.append(cur)
            cur = ch
    if cur: out.append(cur)
    return out


def _viseme(ak):
    if ak in (" ", ""): return "REST"
    if any(c in LAB for c in ak): return "MBP"
    if "ा" in ak or ak in "आअ": return "A"
    if any(c in ak for c in "ोौओऔ"): return "O"
    if any(c in ak for c in "ुूउऊ"): return "U"
    if any(c in ak for c in "िीेैएऐ"): return "E"
    return "A"


def env_ambient(env):
    e = (env or "").lower()
    if any(k in e for k in ["market", "stall", "bazaar", "mela", "fair"]): return "market"
    if any(k in e for k in ["workshop", "craft", "carv", "carpenter", "potter", "forge"]): return "workshop"
    if any(k in e for k in ["city", "street", "road", "traffic", "highway"]): return "traffic"
    if any(k in e for k in ["office", "corporate", "newsroom"]): return "office"
    if any(k in e for k in ["crowd", "rally", "gathering", "temple", "public", "protest", "school"]): return "crowd"
    if any(k in e for k in ["studio", "host", "anchor", "news"]): return "studio"
    if any(k in e for k in ["shop", "store"]): return "shop"
    if any(k in e for k in ["village", "rural", "field", "farm", "hut", "forest", "jungle", "river", "outdoor", "garden"]): return "village"
    if any(k in e for k in ["kitchen", "home", "house", "room", "interior"]): return "home"
    return "soft"


def _pick_music(hi):
    def has(*w): return any(x in hi for x in w)
    if has("भाग", "दौड़", "पीछा", "हमल", "लड़", "युद्ध", "धमाक"): return "action_cinematic"
    if has("मौत", "मर", "खतर", "डर", "फँस", "मुश्क", "साजिश", "धोख"): return "tense_suspense"
    if has("बच", "जीत", "सफल", "उम्मीद", "खुश", "शांति", "जश्न"): return "triumphant"
    return "soft_emotional"


def render(audio_path, output_path, target_seconds=120, topic="", transcript="", progress=None, start_offset=0.0):
    def prog(p, t):
        if progress:
            try: progress(int(p), t)
            except Exception: pass

    try:
        import numpy as np, cv2, soundfile as sf, librosa, requests
        from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageFont
        import imageio_ffmpeg
        _av_stub()
        import warp_face as wf
    except Exception as exc:
        raise RuntimeError(HEAVY_IMPORT_HINT + f"  ({exc.__class__.__name__}: {exc})")

    FF = imageio_ffmpeg.get_ffmpeg_exe()
    audio_path = str(audio_path); output_path = str(output_path)
    W, H, FPS, SZ, BLOCK = 1280, 720, 20, wf.SZ, 6.5
    work = Path(output_path).parent / "cartoon_work"; work.mkdir(parents=True, exist_ok=True)
    ND = str(work)
    CART = (", clean detailed classic Hindi kahaniya 2D cartoon, sharp bold clean black outlines, cel-shaded flat vibrant colors, "
            "the main subject FULLY DETAILED, sharp, well-lit and clearly visible in the foreground, clearly separated from the "
            "background, NOT blurry, no soft focus, no low-detail faces, does not merge into the background, "
            "professional 2D animation, crisp, high detail")

    def run(a, **k): return subprocess.run(a, capture_output=True, **k)
    def runtext(a, **k):
        r = subprocess.run(a, capture_output=True, **k)
        return (r.stdout or b"").decode("utf-8", "replace") + (r.stderr or b"").decode("utf-8", "replace")
    def dur(p):
        o = runtext([FF, "-i", p]); m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", o)
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 0.0

    TOTAL = min(float(target_seconds), dur(audio_path) or float(target_seconds))

    # ---------- transcribe ----------
    prog(8, "Audio transcribe ho raha hai (whisper)...")
    pcm = run([FF, "-y", "-ss", str(start_offset), "-t", str(TOTAL), "-i", audio_path, "-f", "s16le", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-"]).stdout
    audio16 = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
    from faster_whisper import WhisperModel
    try:
        import torch; dev = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        dev = "cpu"
    model = WhisperModel("small", device=dev, compute_type="float16" if dev == "cuda" else "int8")
    segs_it, _ = model.transcribe(audio16, language="hi", vad_filter=True, beam_size=5)
    segs = [{"start": float(s.start), "end": float(s.end), "text": s.text.strip()} for s in segs_it]
    if not segs:
        raise RuntimeError("Audio se koi speech transcribe nahi hui.")
    TOTAL = min(TOTAL, segs[-1]["end"])

    def text_in(a, b): return " ".join(s["text"] for s in segs if s["end"] > a and s["start"] < b).strip()
    blocks = []; t = 0.0
    while t < TOTAL - 0.4:
        e = min(t + BLOCK, TOTAL); blocks.append({"i": len(blocks), "a": t, "b": e, "hi": text_in(t, e)}); t = e
    TOTAL = blocks[-1]["b"]; NB = len(blocks); NF = int(TOTAL * FPS)

    # ---------- character bible (consistency without agents) ----------
    prog(16, "Kahani ke characters samajhe ja rahe hain...")
    full = " ".join(s["text"] for s in segs)[:1600]
    bible = _ptext("From this Hindi story transcript, name the 1-3 MAIN recurring characters and give each a short, "
                   "fixed visual description for a 2D cartoon (age, clothes, look) so they can be drawn identically every time. "
                   "Be concise, one line each. Transcript: " + (topic + ". " + full if topic else full))[:500]

    # ---------- narration audio + envelope + pitch ----------
    NARR = os.path.join(ND, "narr.m4a"); WAVA = os.path.join(ND, "narr.wav")
    run([FF, "-y", "-ss", str(start_offset), "-t", str(TOTAL), "-i", audio_path, "-c:a", "aac", "-b:a", "160k", NARR])
    run([FF, "-y", "-ss", str(start_offset), "-t", str(TOTAL), "-i", audio_path, "-ar", "22050", "-ac", "1", WAVA])
    y, sr = sf.read(WAVA); y = y.astype(np.float32); hop = int(sr / FPS)
    rms = librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0]
    env = np.clip(rms / (np.percentile(rms, 95) + 1e-6), 0, 1.3); env = np.convolve(env, [0.25, 0.5, 0.25], mode="same")
    def openf(fi):
        v = env[min(len(env) - 1, fi)]; return 0.0 if v < 0.12 else min(1.0, (v - 0.12) / 0.78)
    def gender(a, b):
        seg = y[int(a * sr):int(b * sr)]
        if len(seg) < sr // 2: return "m"
        try:
            f0, _, _ = librosa.pyin(seg, fmin=75, fmax=340, sr=sr, frame_length=2048)
            voiced = f0[np.isfinite(f0)]
            return "f" if (len(voiced) >= 5 and float(np.median(voiced)) > 188.0) else "m"
        except Exception:
            return "m"

    # ---------- per-scene analysis (Pollinations-text) ----------
    EREC = {e["name"]: e for e in json.load(io.open(os.path.join(ASSETS, "anim_spec.json"), encoding="utf-8"))["result"]["emotions"]}
    # story cutaways should DOMINATE; the talking-head host appears only for the intro + sparse
    # transitions (and rarer as the episode gets longer, so it never feels like a repetitive anchor show).
    for i, bk in enumerate(blocks):
        bk["kind"] = "cut"   # pure STORY visuals, NO talking-head anchor (per approved demo)
        bk["mus"] = _pick_music(bk["hi"])
        if bk["kind"] == "host":
            bk["host"] = gender(bk["a"], bk["b"]); bk["sfx"] = "studio"; bk["emo"] = "neutral"; bk["emoI"] = 0.5
        else:
            prog(18 + int(20 * i / max(1, NB)), f"Scene {i+1}/{NB} ka background analyze ho raha hai...")
            j = _pjson(f"You are a 2D cartoon scene director for a Hindi story. Topic: {topic or 'Hindi kahaniya'}. "
                       f"Recurring characters: {bible}. For this narration line reply with ONLY a JSON object "
                       '{"environment":"<specific setting e.g. village-workshop/market/city-street/school/home/temple>",'
                       '"prompt":"<one clean English FLAT 2D cartoon scene depicting the line; the main character sharp in the '
                       'foreground with bold black outlines, well-proportioned, not merging with a simpler background, no text>",'
                       '"emotion":"<neutral/happy/sad/tense/hopeful/surprise>",'
                       '"char_voice":"<none | child | elderly_female | elderly_male | adult_male | adult_female>",'
                       '"char_line":"<if the line clearly is a STORY CHARACTER (a child/old woman/man) speaking, write the SHORT Hindi sentence they say; otherwise empty>"}. '
                       'Set char_voice to a person ONLY when the narration is clearly that character speaking dialogue; for plain narration use "none" and empty char_line. '
                       f'Narration line: {bk["hi"]}')
            bk["env"] = j.get("environment", "home")
            bk["prompt"] = (j.get("prompt") or ("a flat 2D cartoon scene of " + bk["hi"][:60])) + CART
            bk["sfx"] = env_ambient(bk["env"]); bk["emo"] = j.get("emotion", "neutral"); bk["emoI"] = 0.6
            cv = (j.get("char_voice") or "none").strip(); cl = (j.get("char_line") or "").strip()
            bk["cvoice"] = cv if (cv != "none" and len(cl) >= 3) else "none"; bk["cline"] = cl if bk["cvoice"] != "none" else ""

    # ---------- cutaway images ----------
    cut_idx = [i for i, bk in enumerate(blocks) if bk["kind"] == "cut"]
    def fetch_img(i):
        ip = os.path.join(ND, f"b{i}.jpg")
        if os.path.exists(ip) and os.path.getsize(ip) > 5000: return True
        for a in range(6):
            url = ("https://image.pollinations.ai/prompt/" + urllib.parse.quote(blocks[i]["prompt"]) +
                   f"?width=1536&height=864&seed={14000+i+a*37}&model=flux&nologo=true&enhance=true")
            try:
                r = requests.get(url, timeout=240)
                if r.status_code == 200 and len(r.content) > 5000: open(ip, "wb").write(r.content); return True
            except Exception: pass
            time.sleep(5)
        return False
    prog(40, "Scene backgrounds generate ho rahe hain...")
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=4) as ex: list(ex.map(fetch_img, cut_idx))
    def have(i): return os.path.exists(os.path.join(ND, f"b{i}.jpg")) and os.path.getsize(os.path.join(ND, f"b{i}.jpg")) > 5000
    for i in [i for i in cut_idx if not have(i)]: fetch_img(i)
    exist = [i for i in cut_idx if have(i)]
    for i in cut_idx:
        if not have(i) and exist:
            shutil.copy(os.path.join(ND, f"b{min(exist, key=lambda e: abs(e-i))}.jpg"), os.path.join(ND, f"b{i}.jpg"))

    # ---------- viseme / emotion timelines ----------
    def emo_disp(name):
        r = EREC.get(name) or EREC.get("neutral")
        if not r: return {}
        b, bx, l, c, dx, dy = r["brow_dy"], r["brow_in_dx"], r["lid_dy"], r["cheek_dy"], r["corner_dx"], r["corner_dy"]
        return {"brow_in_L": (bx, b), "brow_in_R": (-bx, b), "brow_out_L": (0, b), "brow_out_R": (0, b),
                "lid_L": (0, l), "lid_R": (0, l), "cheek_L": (0, c), "cheek_R": (0, c), "mouth_L": (-dx, dy), "mouth_R": (dx, dy)}
    vis_seq = ["REST"] * NF; host_at = [None] * NF; emo_at = [("neutral", 0.0)] * NF
    for bk in blocks:
        if bk["kind"] != "host": continue
        for f in range(max(0, int(bk["a"] * FPS)), min(NF, int(bk["b"] * FPS))):
            host_at[f] = bk["host"]; emo_at[f] = (bk["emo"], bk["emoI"])
        aks = _aksharas(bk["hi"])
        if not aks: continue
        per = (bk["b"] - bk["a"]) / len(aks)
        for n, ak in enumerate(aks):
            v = _viseme(ak)
            for f in range(max(0, int((bk["a"] + n * per) * FPS)), min(NF, int((bk["a"] + (n + 1) * per) * FPS))): vis_seq[f] = v
    POINTS = list(wf.LM_MALE.keys()); raw = {k: np.zeros((NF, 2)) for k in POINTS}
    for f in range(NF):
        if host_at[f] is None: continue
        o = openf(f); vis = vis_seq[f]; en, ei = emo_at[f]; d = {}
        for k, v in emo_disp(en).items(): d[k] = (d.get(k, (0, 0))[0] + v[0] * ei, d.get(k, (0, 0))[1] + v[1] * ei)
        for k, v in wf.VIS[vis].items():
            sc = o if k in VIS_OPEN else 1.0
            d[k] = (d.get(k, (0, 0))[0] + v[0] * sc, d.get(k, (0, 0))[1] + v[1] * sc)
        if o > 0.55:
            for k in ("brow_in_L", "brow_in_R", "brow_out_L", "brow_out_R"): d[k] = (d.get(k, (0, 0))[0], d.get(k, (0, 0))[1] - 0.004 * o)
        if (f % int(2.4 * FPS)) < 3:
            for k in ("lid_L", "lid_R"): d[k] = (d.get(k, (0, 0))[0], d.get(k, (0, 0))[1] + 0.018)
        for k in POINTS: dd = d.get(k, (0, 0)); raw[k][f] = [dd[0], dd[1]]
    ker = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    sm = {k: np.stack([np.convolve(raw[k][:, 0], ker, mode="same"), np.convolve(raw[k][:, 1], ker, mode="same")], 1) for k in POINTS}
    o_sm = np.convolve([openf(f) for f in range(NF)], ker, mode="same")

    # ---------- host studio assets ----------
    BASE = {"m": cv2.resize(cv2.imread(os.path.join(ASSETS, "host_male.png")), (SZ, SZ)),
            "f": cv2.resize(cv2.imread(os.path.join(ASSETS, "host_female.png")), (SZ, SZ))}
    STUDIO = Image.open(os.path.join(ASSETS, "studio_bg.png")).convert("RGB").resize((W, H), Image.LANCZOS)
    MASK = {"m": Image.open(os.path.join(ASSETS, "host_male_cut.png")).convert("RGBA").split()[-1].resize((720, 720), Image.LANCZOS),
            "f": Image.open(os.path.join(ASSETS, "host_female_cut.png")).convert("RGBA").split()[-1].resize((720, 720), Image.LANCZOS)}
    LMs = {"m": wf.LM_MALE, "f": wf.LM_FEMALE}; XOFF = (W - 720) // 2

    def fit(path):
        im = Image.open(path).convert("RGB")
        if np.asarray(im).mean() < 98: im = ImageEnhance.Brightness(im).enhance(1.16)
        im = ImageEnhance.Contrast(im).enhance(1.13); im = ImageEnhance.Color(im).enhance(1.13)
        im = im.filter(ImageFilter.UnsharpMask(radius=2.2, percent=150, threshold=2))
        s = max(W / im.width, H / im.height) * 1.14
        return im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    CUT = {i: fit(os.path.join(ND, f"b{i}.jpg")) for i in cut_idx if have(i)}
    CAMS = ["push_in", "pull_out", "pan_left", "pan_right", "static_drift"]
    def cam(img, p, mode, t=0.0):
        if mode == "push_in": z = 1.05 + 0.10 * p; ox = oy = 0.5
        elif mode == "pull_out": z = 1.15 - 0.10 * p; ox = oy = 0.5
        elif mode == "pan_left": z = 1.10; ox = 0.62 - 0.26 * p; oy = 0.5
        elif mode == "pan_right": z = 1.10; ox = 0.38 + 0.26 * p; oy = 0.5
        else: z = 1.07; ox = 0.5; oy = 0.5
        z += 0.010 * math.sin(t * 2 * math.pi * 0.28)                              # subtle breathing -> alive feel
        ox = min(1, max(0, ox + 0.006 * math.sin(t * 2 * math.pi * 0.19)))         # gentle sway
        oy = min(1, max(0, oy + 0.010 * math.sin(t * 2 * math.pi * 0.22)))
        cw, ch = int(W / z), int(H / z); px = int((img.width - cw) * ox); py = int((img.height - ch) * oy)
        px = max(0, min(img.width - cw, px)); py = max(0, min(img.height - ch, py))
        return img.crop((px, py, px + cw, py + ch)).resize((W, H), Image.LANCZOS)
    try: bfont = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 24)
    except Exception: bfont = ImageFont.load_default()
    def brand(c):
        d = ImageDraw.Draw(c, "RGBA"); d.text((W - 205, 20), "THE MYSTERY", font=bfont, fill=(255, 255, 255, 205))
    yy, xx = np.mgrid[0:H, 0:W]; vig = np.clip(1 - 0.34 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2), 0.55, 1)[..., None]
    BOUNDS = [bk["a"] for bk in blocks] + [TOTAL]
    def idx_at(tt):
        for i in range(NB):
            if BOUNDS[i] <= tt < BOUNDS[i + 1]: return i
        return NB - 1

    # ---------- render frames ----------
    prog(60, "Lip-sync animation render ho rahi hai...")
    SIL = os.path.join(ND, "_sil.mp4")
    enc = subprocess.Popen([FF, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", SIL], stdin=subprocess.PIPE)
    XF = 5; prev = None
    for f in range(NF):
        tt = f / FPS; k = idx_at(tt); bk = blocks[k]; p = (tt - bk["a"]) / max(0.1, bk["b"] - bk["a"])
        if bk["kind"] == "host":
            hid = bk["host"]; disp = {kk: (sm[kk][f, 0], sm[kk][f, 1]) for kk in POINTS}
            w768 = wf.warp(BASE[hid], disp, LMs[hid]); w768 = wf.mouth_interior(w768, float(o_sm[f]), vis_seq[f], *wf.MXY[hid])
            face = Image.fromarray(cv2.cvtColor(w768, cv2.COLOR_BGR2RGB)).resize((720, 720), Image.LANCZOS)
            dy = int(round(2.2 * math.sin(tt * 2.0) + 1.0 * o_sm[f])); cv_ = STUDIO.copy(); cv_.paste(face, (XOFF, dy), MASK[hid]); fr = cv_
        else:
            img = CUT.get(k) or (next(iter(CUT.values())) if CUT else Image.new("RGB", (W, H), (20, 20, 30))); fr = cam(img, p, CAMS[(k * 7 + 2) % 5], tt)
        a = np.asarray(fr.convert("RGB")).astype(np.float32) * vig; fr = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGBA"); brand(fr)
        lf = f - int(bk["a"] * FPS)
        if k > 0 and lf < XF and prev is not None: fr = Image.blend(prev, fr, lf / XF)
        if f == int(bk["b"] * FPS) - 1: prev = fr.copy()
        enc.stdin.write(fr.convert("RGB").tobytes())
    enc.stdin.close(); enc.wait()

    # ---------- SFX + music ----------
    prog(82, "Scene-aware ambient sound + music...")
    def lavfi(ins, fc, d, out):
        cmd = [FF, "-y"]
        for s in ins: cmd += ["-f", "lavfi", "-i", s]
        cmd += ["-filter_complex", fc, "-map", "[o]", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", "-t", str(round(d, 3)), out]; run(cmd)
    def sfx_bed(tag, d, out):
        T = {
         "wind": (["anoisesrc=d=%s:c=brown:a=0.9" % d], "[0:a]lowpass=f=550,highpass=f=70,tremolo=f=0.15:d=0.7,tremolo=f=0.5:d=0.3,volume=2.4[o]"),
         "rain": (["anoisesrc=d=%s:c=white:a=0.5" % d], "[0:a]highpass=f=450,lowpass=f=8000,volume=1.5[o]"),
         "market": (["anoisesrc=d=%s:c=brown:a=0.75" % d, "anoisesrc=d=%s:c=white:a=0.45" % d], "[0:a]lowpass=f=1100,volume=1.6[a];[1:a]highpass=f=1400,lowpass=f=6500,tremolo=f=9:d=0.7,tremolo=f=4:d=0.5,volume=0.55[b];[a][b]amix=inputs=2:normalize=0[o]"),
         "crowd": (["anoisesrc=d=%s:c=brown:a=0.7" % d, "anoisesrc=d=%s:c=white:a=0.4" % d], "[0:a]lowpass=f=1000,volume=1.5[a];[1:a]highpass=f=1500,lowpass=f=6000,tremolo=f=11:d=0.7,volume=0.45[b];[a][b]amix=inputs=2:normalize=0[o]"),
         "workshop": (["anoisesrc=d=%s:c=brown:a=0.32" % d, "anoisesrc=d=%s:c=white:a=0.4" % d], "[0:a]lowpass=f=520,volume=0.7[a];[1:a]highpass=f=1100,lowpass=f=5200,tremolo=f=1.3:d=0.96,volume=0.4[b];[a][b]amix=inputs=2:normalize=0[o]"),
         "village": (["anoisesrc=d=%s:c=brown:a=0.55" % d, "sine=f=2500:d=%s" % d], "[0:a]lowpass=f=620,highpass=f=70,tremolo=f=0.2:d=0.5,volume=1.2[a];[1:a]vibrato=f=7:d=0.9,tremolo=f=5:d=1.0,highpass=f=1800,volume=0.2[b];[a][b]amix=inputs=2:normalize=0[o]"),
         "traffic": (["anoisesrc=d=%s:c=brown:a=0.85" % d, "anoisesrc=d=%s:c=brown:a=0.5" % d], "[0:a]lowpass=f=300,volume=1.7[a];[1:a]lowpass=f=650,highpass=f=160,tremolo=f=7:d=0.5,volume=0.55[b];[a][b]amix=inputs=2:normalize=0[o]"),
         "office": (["anoisesrc=d=%s:c=brown:a=0.28" % d, "sine=f=120:d=%s" % d], "[0:a]lowpass=f=420,volume=0.5[a];[1:a]volume=0.05[b];[a][b]amix=inputs=2:normalize=0[o]"),
         "studio": (["anoisesrc=d=%s:c=brown:a=0.2" % d, "sine=f=110:d=%s" % d], "[0:a]lowpass=f=400,volume=0.4[a];[1:a]volume=0.03[b];[a][b]amix=inputs=2:normalize=0[o]"),
        }
        ins, fc = T.get(tag, (["anoisesrc=d=%s:c=brown:a=0.2" % d], "[0:a]lowpass=f=420,volume=0.4[o]"))
        lavfi(ins, fc, d, out)
    def music_bed(mood, d, out):
        M = {
         "triumphant": (["sine=f=261.63:d=%s" % d, "sine=f=329.63:d=%s" % d, "sine=f=392.0:d=%s" % d, "sine=f=523.25:d=%s" % d], "[0:a]volume=0.18[a];[1:a]volume=0.15[b];[2:a]volume=0.15[c];[3:a]volume=0.09[e];[a][b][c][e]amix=inputs=4:normalize=0,tremolo=f=0.3:d=0.2,aecho=0.8:0.5:300:0.3[o]"),
         "soft_emotional": (["sine=f=220:d=%s" % d, "sine=f=261.63:d=%s" % d, "sine=f=329.63:d=%s" % d], "[0:a]volume=0.14[a];[1:a]volume=0.12[b];[2:a]volume=0.10[c];[a][b][c]amix=inputs=3:normalize=0,tremolo=f=0.2:d=0.25,aecho=0.9:0.4:420:0.35,lowpass=f=2200[o]"),
         "tense_suspense": (["sine=f=73.42:d=%s" % d, "sine=f=77.78:d=%s" % d, "anoisesrc=d=%s:c=pink:a=0.05" % d], "[0:a]volume=0.30[a];[1:a]volume=0.18[b];[2:a]lowpass=f=900,volume=0.4[c];[a][b][c]amix=inputs=3:normalize=0,tremolo=f=0.8:d=0.4,aecho=0.9:0.5:500:0.4[o]"),
         "action_cinematic": (["sine=f=55:d=%s" % d, "sine=f=110:d=%s" % d, "sine=f=164.81:d=%s" % d, "anoisesrc=d=%s:c=brown:a=0.25" % d], "[0:a]volume=0.32,tremolo=f=2.0:d=0.85[a];[1:a]volume=0.18,tremolo=f=2.0:d=0.6[b];[2:a]volume=0.14[c];[3:a]lowpass=f=180,volume=0.6,tremolo=f=2.0:d=0.8[d];[a][b][c][d]amix=inputs=4:normalize=0,aecho=0.8:0.4:250:0.3[o]"),
        }
        ins, fc = M.get(mood, (["sine=f=60:d=%s" % d], "[0:a]volume=0.05[o]"))
        lavfi(ins, fc, d, out)
    sp = []; mp = []
    for i, bk in enumerate(blocks):
        d = bk["b"] - bk["a"]; fs = os.path.join(ND, f"_s{i}.wav"); sfx_bed(bk["sfx"], d, fs); sp.append(fs)
        fm = os.path.join(ND, f"_m{i}.wav"); music_bed(bk["mus"], d, fm); mp.append(fm)
    def concat(parts, out):
        lst = out + ".txt"
        with io.open(lst, "w", encoding="utf-8") as fh:        # close before ffmpeg reads it
            fh.write("".join(f"file '{os.path.basename(pp)}'\n" for pp in parts))
        run([FF, "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c:a", "pcm_s16le", out], cwd=ND)
        return os.path.exists(out) and os.path.getsize(out) > 1000
    SFXF = os.path.join(ND, "sfx_all.wav"); MUSF = os.path.join(ND, "mus_all.wav")
    ok_sfx = concat(sp, SFXF); ok_mus = concat(mp, MUSF)

    # ---------- captions ----------
    def at(tt): return f"{int(tt//3600)}:{int(tt%3600//60):02d}:{tt%60:05.2f}"
    for c in ["C:/Windows/Fonts/Nirmala.ttf", "C:/Windows/Fonts/Nirmala.ttc"]:
        if os.path.exists(c):
            try: shutil.copy(c, os.path.join(ND, os.path.basename(c)))
            except Exception: pass
    ass = ["[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}", "",
     "[V4+ Styles]", "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
     "Style: D,Nirmala UI,40,&H00FFFFFF,&H00101010,&HA0000000,1,3,3,2,2,70,70,40,1", "",
     "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    for bk in blocks:
        if bk["hi"]: ass.append(f"Dialogue: 0,{at(bk['a'])},{at(min(bk['b'], TOTAL))},D,,0,0,0,,{bk['hi']}")
    with io.open(os.path.join(ND, "cap.ass"), "w", encoding="utf-8") as _fh:
        _fh.write("\n".join(ass))
    CAPV = os.path.join(ND, "capped.mp4")
    runtext([FF, "-y", "-i", "_sil.mp4", "-vf", "subtitles=cap.ass:fontsdir=.", "-c:v", "libx264", "-pix_fmt", "yuv420p", "capped.mp4"], cwd=ND)
    vid = CAPV if (os.path.exists(CAPV) and os.path.getsize(CAPV) > 10000) else SIL

    # ---------- VOICE-HYBRID: at character-dialogue scenes, swap the host narration for the character's
    #            own TTS voice (child / elderly woman / man) for that window; elsewhere keep the NLM audio ----------
    NARR_USE = NARR
    char_scenes = [b for b in blocks if b.get("cvoice", "none") != "none"]
    if char_scenes:
        prog(90, f"{len(char_scenes)} character voice(s) add ho rahi hain...")
        try:
            import asyncio, edge_tts
            VMAP = {"child": ("hi-IN-SwaraNeural", "+18Hz", "+10%"), "adult_female": ("hi-IN-SwaraNeural", "+0Hz", "+0%"),
                    "elderly_female": ("hi-IN-SwaraNeural", "-12Hz", "-10%"), "adult_male": ("hi-IN-MadhurNeural", "-2Hz", "-2%"),
                    "elderly_male": ("hi-IN-MadhurNeural", "-10Hz", "-8%")}
            async def _mktts():
                for bk in char_scenes:
                    v, pit, rat = VMAP.get(bk["cvoice"], VMAP["adult_male"])
                    fp = os.path.join(ND, f"_cv{bk['i']}.mp3")
                    try: await edge_tts.Communicate(bk["cline"], v, pitch=pit, rate=rat).save(fp); bk["_tts"] = fp
                    except Exception: bk["_tts"] = None
            asyncio.run(_mktts())
            NARR_WAV = os.path.join(ND, "_narr_full.wav")    # PCM source -> sample-accurate slicing (no drift)
            run([FF, "-y", "-i", NARR, "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", NARR_WAV])
            parts = []
            for bk in blocks:
                d = max(0.2, bk["b"] - bk["a"]); seg = os.path.join(ND, f"_na{bk['i']}.wav"); tts = bk.get("_tts")
                if bk.get("cvoice", "none") != "none" and tts and os.path.exists(tts):    # character window -> their voice
                    run([FF, "-y", "-i", tts, "-af", f"aresample=44100,apad,atrim=0:{d:.3f},asetpts=PTS-STARTPTS", "-ac", "2", "-c:a", "pcm_s16le", "-t", f"{d:.3f}", seg])
                else:                                                                       # otherwise keep the NLM slice (PCM, exact)
                    run([FF, "-y", "-ss", f"{bk['a']:.3f}", "-t", f"{d:.3f}", "-i", NARR_WAV, "-ac", "2", "-c:a", "pcm_s16le", seg])
                parts.append(seg)
            if all(os.path.exists(p) and os.path.getsize(p) > 800 for p in parts):
                cmd = [FF, "-y"]
                for p in parts: cmd += ["-i", p]
                cmd += ["-filter_complex", "".join(f"[{i}:a]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=0:a=1[o]",
                        "-map", "[o]", "-ar", "44100", "-ac", "2", "-c:a", "aac", "-b:a", "160k", os.path.join(ND, "narr_final.m4a")]
                run(cmd)
                NF2 = os.path.join(ND, "narr_final.m4a")
                if os.path.exists(NF2) and os.path.getsize(NF2) > 5000: NARR_USE = NF2
        except Exception:
            NARR_USE = NARR     # voice-hybrid is best-effort; on any failure keep the original NLM narration

    # ---------- mux (adaptive: full scene-aware mix, or narration-only if a track is missing) ----------
    prog(94, "Final video taiyaar ho raha hai...")
    if ok_sfx and ok_mus:
        runtext([FF, "-y", "-i", vid, "-i", NARR_USE,"-i", SFXF, "-i", MUSF, "-filter_complex",
           "[1:a]volume=1.55,asplit=3[vmain][vk1][vk2];"
           "[2:a]volume=0.5[sfx];[sfx][vk1]sidechaincompress=threshold=0.05:ratio=8:attack=15:release=350[sfxd];"
           "[3:a]volume=0.34[mus];[mus][vk2]sidechaincompress=threshold=0.06:ratio=5:attack=20:release=400[musd];"
           "[vmain][sfxd][musd]amix=inputs=3:normalize=0:duration=first[mx];[mx]alimiter=limit=0.95[ao]",
           "-map", "0:v", "-map", "[ao]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", output_path])
    else:
        runtext([FF, "-y", "-i", vid, "-i", NARR_USE,"-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", output_path])
    if not (os.path.exists(output_path) and os.path.getsize(output_path) > 50000):
        raise RuntimeError("Final cartoon video mux nahi ho paaya.")
    prog(100, "Done")
    return TOTAL
