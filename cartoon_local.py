# Self-contained LOCAL cartoon-episode pipeline (GPU recommended). Wired into app.py's
# "2D Cartoon Episode (Lip Sync)" mode. Heavy deps (faster-whisper, torch, rembg, cv2, scipy,
# librosa) are imported lazily INSIDE render() so the lightweight Streamlit Cloud app still imports
# app.py fine -- this module only runs when the user generates a cartoon episode on a local machine.
#
# Pipeline: upload audio -> transcribe (whisper) -> ~6.5s blocks -> per-scene environment + clean
# flat-2D prompt + emotion via Pollinations-text (no Claude agents needed) -> recurring host
# narrators with face-WARP viseme lip-sync in a cartoon studio -> story cutaway scenes (Pollinations
# image) -> scene-aware ambient SFX + mood music -> burned Hindi captions -> mux the user's audio.
import os, io, json, math, time, shutil, subprocess, urllib.parse, re, hashlib, zlib
from pathlib import Path


def _salt(*parts):
    h = hashlib.sha1(("|".join(str(p) for p in parts)).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _ckey(label):
    """Stable identity key for a character so the same person reuses one seed+look across scenes."""
    k = re.sub(r"[^a-z0-9 ]", "", (label or "").lower())
    k = re.sub(r"\b(the|a|an|young|old|elderly|middle aged|man|woman|person|guy)\b", "", k)
    return re.sub(r"\s+", " ", k).strip()


def _emo_music(emo, fallback):
    e = (emo or "").lower()
    if e in ("tense", "fear", "afraid", "anxious"): return "tense_suspense"
    if e in ("sad", "grief", "sorrow"): return "soft_emotional"
    if e in ("happy", "hopeful", "triumph", "joy", "proud"): return "triumphant"
    if e in ("surprise", "shock", "action", "angry", "anger"): return "action_cinematic"
    return fallback

ASSETS = Path(__file__).resolve().parent / "assets"
HEAVY_IMPORT_HINT = ("Cartoon mode ke liye local dependencies chahiye. Install: "
                     "pip install -r requirements-cartoon-local.txt  (GPU recommended).")

# ============================================================================
# LIVING-ENVIRONMENT MOTION  (merged rain / snow-dust / fog-smoke / water-waves /
# foliage-leaves / city-light-streaks). Pure numpy + PIL only -- NO video model,
# NO proprietary tracking. ONE dispatcher, ONE static-buffer cache per (kind,seed).
# Returns a 1280x720 RGB PIL.Image, or the SAME object untouched for no-op tags.
# Called ONLY on cutaway frames; host (anchor) frames never reach it.
# Master dial below: set 0.0 to disable everything instantly.
# ============================================================================
from functools import lru_cache as _lru_cache

ENV_MOTION_INTENSITY = 1.0           # global 0..1 (0 = off)
_ENVM_W, _ENVM_H = 1280, 720

def _envm_kind(tag, env):
    """(sfx tag, raw env text) -> effect name. Rain & snow MUST read raw env because
    env_ambient() collapses rain->'water' and mountains/snow->'village'."""
    e = (env or "").lower()
    if any(w in e for w in ("rain", "monsoon", "storm", "downpour", "drizzle", "shower")):
        return "rain"
    if any(w in e for w in ("snow", "himalaya", "winter", "glacier", "kashmir",
                            "shimla", "manali", "ladakh", "blizzard", "snowfall")):
        return "snow"
    t = (tag or "").lower()
    if t == "water":                              return "water"
    if t == "forest":                             return "foliage"
    if t in ("temple", "workshop"):               return "fog"
    if t == "traffic":                            return "traffic"
    if t in ("market", "crowd"):                  return "crowd"
    if t == "night":                              return "spark"
    return None                                   # office/indoor/studio/village/... -> universal alive base layer only

@_lru_cache(maxsize=64)
def _envm_cache(kind, seed):
    """Build static per-scene buffers ONCE (lru_cache keyed on (kind, seed))."""
    import numpy as np
    H, W = _ENVM_H, _ENVM_W
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    C = {"kind": kind}
    if kind in ("rain", "snow", "spark"):
        n = {"rain": 240, "snow": 130, "spark": 150}[kind]
        C["x0"]  = rng.uniform(-0.15 * W, 1.05 * W, n).astype(np.float32)
        C["y0"]  = rng.uniform(0, H, n).astype(np.float32)
        C["ph"]  = rng.uniform(0, 6.2832, n).astype(np.float32)
        C["spd"] = rng.uniform(0.15, 1.0, n).astype(np.float32)
        C["amp"] = rng.uniform(0.4, 1.0, n).astype(np.float32)
        C["n"]   = n
        if kind == "rain":
            nf = 26                                          # foreground near-cam streaks
            C["fx0"]  = rng.uniform(-0.20 * W, 1.10 * W, nf).astype(np.float32)
            C["fy0"]  = rng.uniform(0, H, nf).astype(np.float32)
            C["fsp"]  = rng.uniform(0.60, 1.0, nf).astype(np.float32)
            C["famp"] = rng.uniform(0.70, 1.0, nf).astype(np.float32)
            C["nf"]   = nf
            ns = 30                                          # ground splash anchors
            C["sx"]    = rng.uniform(0.02 * W, 0.98 * W, ns).astype(np.float32)
            C["sy"]    = rng.uniform(0.84 * H, 0.985 * H, ns).astype(np.float32)
            C["sph"]   = rng.uniform(0.0, 1.0, ns).astype(np.float32)
            C["srate"] = rng.uniform(1.4, 2.4, ns).astype(np.float32)
            C["ssz"]   = rng.uniform(3.5, 8.0, ns).astype(np.float32)
            C["ns"]    = ns
    elif kind == "traffic":
        n = 26
        C["lane"] = rng.uniform(0.62, 0.985, n).astype(np.float32)
        C["x0"]   = rng.uniform(-0.2 * W, 1.2 * W, n).astype(np.float32)
        C["dir"]  = np.where(rng.random(n) < 0.5, -1.0, 1.0).astype(np.float32)
        C["spd"]  = rng.uniform(0.55, 1.0, n).astype(np.float32)
        C["len"]  = rng.uniform(48.0, 120.0, n).astype(np.float32)
        C["thk"]  = rng.integers(3, 8, n).astype(np.intp)
        C["warm"] = (rng.random(n) < 0.55).astype(np.float32)
        C["amp"]  = rng.uniform(0.55, 1.0, n).astype(np.float32)
        C["n"]    = n
        yb = np.arange(H, dtype=np.float32)
        C["shear_w"] = np.clip((yb - 0.60 * H) / (0.40 * H), 0.0, 1.0).astype(np.float32) ** 1.4
        C["shear_rows"] = np.nonzero(C["shear_w"] > 1e-3)[0]
    elif kind == "crowd":
        n = 40
        C["bx"]  = rng.uniform(0.02 * W, 0.98 * W, n).astype(np.float32)
        C["by"]  = rng.uniform(0.66 * H, 0.96 * H, n).astype(np.float32)
        C["ph"]  = rng.uniform(0.0, 6.2832, n).astype(np.float32)
        C["bsp"] = rng.uniform(0.7, 1.6, n).astype(np.float32)
        C["drift"] = (np.where(rng.random(n) < 0.5, -1.0, 1.0)
                      * rng.uniform(6.0, 16.0, n)).astype(np.float32)
        C["rad"] = rng.uniform(7.0, 14.0, n).astype(np.float32)
        C["dark"] = rng.uniform(0.30, 0.62, n).astype(np.float32)
        C["n"]   = n
        yb = np.arange(H, dtype=np.float32)
        C["shim_w"] = np.clip((yb - 0.62 * H) / (0.38 * H), 0.0, 1.0).astype(np.float32) ** 1.3
        C["shim_rows"] = np.nonzero(C["shim_w"] > 1e-3)[0]
    elif kind == "fog":
        small = rng.random((40, (2 * W) // 16)).astype(np.float32)
        from PIL import Image, ImageFilter
        sm = (Image.fromarray((small * 255).astype(np.uint8))
              .resize((2 * W, H + H // 4), Image.BICUBIC)
              .filter(ImageFilter.GaussianBlur(14)))
        tile = np.asarray(sm, dtype=np.float32) / 255.0
        tile = (tile - tile.min()) / (np.ptp(tile) + 1e-6)
        C["tile"]  = np.ascontiguousarray(tile)
        C["vgrad"] = np.linspace(0.26, 1.0, H).astype(np.float32)[:, None]
    elif kind == "water":
        y = np.arange(H, dtype=np.float32)
        y0 = 0.58 * H
        region = np.clip((y - y0) / (H - y0), 0.0, 1.0).astype(np.float32) ** 2
        C["region"]   = region
        C["ampv"]     = (1.6 + 5.4 * region).astype(np.float32)
        C["rowphase"] = rng.uniform(0, 2 * np.pi, size=H).astype(np.float32)
        C["active"]   = np.nonzero(region > 1e-3)[0]
    elif kind == "foliage":
        n = 7
        C["bx"] = rng.uniform(0.04, 0.96, n).astype(np.float32) * W
        C["by"] = rng.uniform(0.0, 1.0, n).astype(np.float32) * H
        C["ph"] = rng.uniform(0, 6.2832, n).astype(np.float32)
        C["sp"] = rng.uniform(0.45, 0.85, n).astype(np.float32)
        C["sz"] = rng.integers(5, 11, n).astype(np.float32)
        C["sw"] = rng.uniform(10.0, 26.0, n).astype(np.float32)
        C["n"]  = n
    return C

def _envm_emo_gain(emo):
    e = (emo or "").lower()
    if e in ("calm", "sad", "neutral", "hopeful"):                         return 0.8
    if e in ("tense", "fear", "angry", "anger", "surprise", "action"):     return 1.15
    return 1.0

def _envm_alive(a, t, seed, inten):
    """UNIVERSAL 'alive' base layer for EVERY cutaway frame. Face-safe:
    low-amplitude (<=~3px) very-low-spatial-frequency separable domain warp +
    slow drifting brightness plane. `a` is float32 (H,W,3); returns warped copy."""
    import numpy as np
    H, W = a.shape[0], a.shape[1]
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    p1, p2, p3, p4 = rng.uniform(0.0, 6.2832, 4).astype(np.float32)
    amp = np.float32(2.6 * float(inten))
    yy = np.arange(H, dtype=np.float32)
    xx = np.arange(W, dtype=np.float32)
    rowdx = amp * np.sin((2.0 * np.pi / (H / 1.4)) * yy + 0.55 * t + p1)            # (H,)
    coldy = (0.85 * amp) * np.sin((2.0 * np.pi / (W / 1.2)) * xx + 0.38 * t + p2)   # (W,)
    sx = np.rint(rowdx).astype(np.intp)
    xb = np.clip(xx[None, :].astype(np.intp) - sx[:, None], 0, W - 1)               # (H,W)
    a = a[np.arange(H)[:, None], xb, :]
    sy = np.rint(coldy).astype(np.intp)
    yb = np.clip(yy[:, None].astype(np.intp) - sy[None, :], 0, H - 1)               # (H,W)
    a = a[yb, np.arange(W)[None, :], :]
    gx = np.sin((2.0 * np.pi / (W / 0.9)) * xx + 0.21 * t + p3)                     # (W,)
    gy = np.sin((2.0 * np.pi / (H / 0.7)) * yy + 0.17 * t + p4)                     # (H,)
    plane = (gy[:, None] + gx[None, :]) * np.float32(1.6 * float(inten))            # (H,W)
    a = a + plane[..., None]
    np.clip(a, 0, 255, out=a)
    return a

def env_motion(fr, tag, tt, seed, emo="", env=""):
    """ONE-LINE entry for the cutaway frame loop. Always returns a 1280x720 RGB PIL.Image.
    Cannot raise on any tag/seed: unknown tag / disabled dial -> returns fr unchanged."""
    try:
        if ENV_MOTION_INTENSITY <= 0.001:
            return fr
        import numpy as np
        from PIL import Image
        inten = ENV_MOTION_INTENSITY * _envm_emo_gain(emo)
        H, W = _ENVM_H, _ENVM_W
        t = float(tt)
        sd = int(seed) & 0x7FFFFFFF

        # >>> UNIVERSAL ALIVE BASE LAYER: runs on EVERY cutaway tag (incl. office/
        #     indoor/studio/village/unmapped). Composes UNDER the tagged effect.
        a0 = _envm_alive(np.asarray(fr, dtype=np.float32).copy(), t, sd, inten)
        fr = Image.fromarray(a0.astype(np.uint8), "RGB")

        kind = _envm_kind(tag, env)
        if kind is None:
            return fr                      # no-tag scenes: alive layer alone keeps them moving
        C = _envm_cache(kind, sd)

        # --- additive particle streaks/dots: rain / snow / night-spark ---
        if kind in ("rain", "snow", "spark"):
            a = np.asarray(fr, dtype=np.float32)
            x0, y0, ph, spd, amp, n = C["x0"], C["y0"], C["ph"], C["spd"], C["amp"], C["n"]
            if kind == "rain":
                _Y, _X, _A = [], [], []
                def _drop(yi, xi, av):
                    _Y.append(yi.ravel()); _X.append(xi.ravel()); _A.append(av.ravel())
                # MAIN downpour: dense long bright slanted streaks
                slant = 0.30
                fall  = (700.0 + 460.0 * spd)
                yh = np.mod(y0 + fall * t, H + 60.0) - 30.0
                xh = np.mod(x0 - slant * fall * t, W * 1.35) - 0.18 * W
                STEPS = 20
                ln = 34.0 + 40.0 * amp
                frt = np.linspace(0.0, 1.0, STEPS, dtype=np.float32)
                ys = yh[:, None] + ln[:, None] * frt[None, :]
                xs = xh[:, None] + (slant * ln[:, None]) * frt[None, :]
                aa = (0.42 + 0.46 * amp)[:, None] * (0.25 + 0.75 * frt[None, :])
                _drop(np.round(ys).astype(np.intp), np.round(xs).astype(np.intp), aa * inten)
                # FOREGROUND depth: 26 big fast 3px-thick near-cam streaks
                fx0, fy0, fsp, famp, nf = C["fx0"], C["fy0"], C["fsp"], C["famp"], C["nf"]
                fslant = 0.34
                ffall  = (1150.0 + 520.0 * fsp)
                fyh = np.mod(fy0 + ffall * t, H + 90.0) - 45.0
                fxh = np.mod(fx0 - fslant * ffall * t, W * 1.4) - 0.20 * W
                FSTEPS = 26
                fln = 70.0 + 70.0 * famp
                ffrt = np.linspace(0.0, 1.0, FSTEPS, dtype=np.float32)
                fys = fyh[:, None] + fln[:, None] * ffrt[None, :]
                fxs = fxh[:, None] + (fslant * fln[:, None]) * ffrt[None, :]
                faa = (0.55 + 0.45 * famp)[:, None] * (0.2 + 0.8 * ffrt[None, :])
                fyi = np.round(fys).astype(np.intp); fxi = np.round(fxs).astype(np.intp)
                fav = faa * inten
                _drop(fyi, fxi, fav)
                _drop(fyi, fxi - 1, 0.5 * fav)
                _drop(fyi, fxi + 1, 0.5 * fav)
                # SPLASHES: expanding ring/flick particles on the ground band
                sx, sy, sph, srate, ssz = C["sx"], C["sy"], C["sph"], C["srate"], C["ssz"]
                cyc   = t * srate + sph
                life  = cyc - np.floor(cyc)
                rad   = life * ssz
                brite = (1.0 - life) ** 1.6
                TH = 10
                ang = np.linspace(0.0, np.pi, TH, dtype=np.float32)
                rx = np.cos(ang)[None, :] * rad[:, None]
                ry = -(np.sin(ang)[None, :] * rad[:, None] * 0.6)
                _drop(np.round(sy[:, None] + ry).astype(np.intp),
                      np.round(sx[:, None] + rx).astype(np.intp),
                      np.repeat((brite * 0.9)[:, None], TH, axis=1) * inten)
                _drop(np.round(sy).astype(np.intp), np.round(sx).astype(np.intp),
                      brite * 0.7 * inten)
                yi = np.concatenate(_Y); xi = np.concatenate(_X)
                av = np.concatenate(_A).astype(np.float32)
                m = (yi >= 0) & (yi < H) & (xi >= 0) & (xi < W)
                acc = np.bincount(yi[m] * W + xi[m], weights=av[m],
                                  minlength=H * W).astype(np.float32).reshape(H, W)
                acc[:, 1:] += 0.55 * acc[:, :-1]
                tint = np.array([200, 216, 240], np.float32)
            elif kind == "snow":
                yy = np.mod(y0 + (40.0 + 60.0 * spd) * t, H + 20.0) - 10.0
                xx = np.mod(x0 + 26.0 * np.sin(t * 0.5 + ph), W).astype(np.float32)
                yi = np.round(yy).astype(np.intp); xi = np.round(xx).astype(np.intp)
                av = (0.5 + 0.5 * amp) * 0.9 * inten
                m = (yi >= 0) & (yi < H) & (xi >= 0) & (xi < W)
                acc = np.bincount(yi[m] * W + xi[m], weights=av[m],
                                  minlength=H * W).astype(np.float32).reshape(H, W)
                acc[1:, :] += 0.5 * acc[:-1, :]; acc[:, 1:] += 0.5 * acc[:, :-1]
                tint = np.array([255, 255, 255], np.float32)
            elif kind == "spark":
                yy = np.mod(y0 + (3.0 * spd) * t, H)
                xx = np.mod(x0 + 5.0 * np.sin(t * 0.55 + ph), W)
                tw = (0.5 + 0.5 * np.sin(t * 2.3 + ph * 3.0)) * amp
                yi = np.round(yy).astype(np.intp); xi = np.round(xx).astype(np.intp)
                av = tw * 0.75 * inten
                m = (yi >= 0) & (yi < H) & (xi >= 0) & (xi < W)
                acc = np.bincount(yi[m] * W + xi[m], weights=av[m],
                                  minlength=H * W).astype(np.float32).reshape(H, W)
                acc[1:, :] += 0.5 * acc[:-1, :]; acc[:, 1:] += 0.5 * acc[:, :-1]   # 2px soft glow
                tint = np.array([205, 222, 255], np.float32)
            np.clip(acc, 0.0, 1.0, out=acc)
            a += acc[..., None] * tint[None, None, :]
            np.clip(a, 0, 255, out=a)
            return Image.fromarray(a.astype(np.uint8), "RGB")

        # --- fog/smoke drifting low-opacity tint (temple/workshop) ---
        if kind == "fog":
            tile, vgrad = C["tile"], C["vgrad"]
            Ht = tile.shape[0]
            vslack = Ht - H
            hslack = tile.shape[1] - W
            rising = (tag or "").lower() in ("temple", "workshop")
            axf = (t / 13.0) % 1.0
            ayf = ((-t / 17.0) % 1.0) if rising else ((t / 22.0) % 1.0)
            ox = int(axf * hslack); oy = int(ayf * vslack)
            A = tile[oy:oy + H, ox:ox + W]
            bxf = (-t / 7.0) % 1.0
            byf = (t / 9.5) % 1.0
            bx = int(bxf * hslack); by = int(byf * vslack)
            B = tile[by:by + H, bx:bx + W]
            m = np.float32(0.5) + np.float32(0.5) * np.sin(np.float32(2.0 * np.pi * 0.09) * t)
            dens = A + B + (np.float32(1.7) * m) * (A * B)
            dens -= np.float32(0.5)
            np.clip(dens, 0.0, 1.0, out=dens)
            al = ((dens * vgrad) * np.float32(0.34 * inten))[..., None]
            arr = np.asarray(fr, dtype=np.float32)
            arr += (np.float32([224, 228, 234]) - arr) * al
            np.clip(arr, 0, 255, out=arr)
            return Image.fromarray(arr.astype(np.uint8), "RGB")

        # --- water: per-row horizontal ripple in lower band + drifting glint ---
        if kind == "water":
            arr = np.asarray(fr, dtype=np.uint8)
            out = arr.copy()
            region, ampv, rowphase, active = C["region"], C["ampv"], C["rowphase"], C["active"]
            yidx = np.arange(H, dtype=np.float32)
            k1 = np.float32(2.0 * np.pi / 150.0); k2 = np.float32(2.0 * np.pi / 47.0); k3 = np.float32(2.0 * np.pi / 23.0)
            w1 = np.float32(2.0 * np.pi * 0.18);  w2 = np.float32(2.0 * np.pi * 0.42);  w3 = np.float32(2.0 * np.pi * 0.66)
            AMP = np.float32(2.1)
            bob = np.float32(1.7) * np.sin(np.float32(2.0 * np.pi * 0.09) * t)
            disp = ((AMP * ampv * np.sin(k1 * yidx - w1 * t + rowphase)
                     + np.float32(0.55) * AMP * ampv * np.sin(k2 * yidx - w2 * t)
                     + np.float32(0.30) * AMP * ampv * np.sin(k3 * yidx - w3 * t + rowphase * np.float32(1.7))
                     + bob) * inten).astype(np.int32)
            ax = np.arange(W)
            for yy in active:
                d = int(disp[yy])
                if d:
                    shifted = arr[yy][(ax - d) % W]
                    w = region[yy]
                    out[yy] = (arr[yy].astype(np.float32) * (1.0 - w)
                               + shifted.astype(np.float32) * w).astype(np.uint8)
            bc1 = (0.66 + 0.20 * np.sin(np.float32(2.0 * np.pi * 0.05) * t)) * H
            bc2 = (0.86 + 0.10 * np.sin(np.float32(2.0 * np.pi * 0.08) * t + 1.7)) * H
            g1 = np.exp(-((yidx - bc1) ** 2) / (2.0 * 38.0 * 38.0))
            g2 = np.exp(-((yidx - bc2) ** 2) / (2.0 * 26.0 * 26.0))
            glint = (g1 + 0.7 * g2) * region
            add = (glint[:, None, None] * np.float32(34.0 * inten))
            outf = np.clip(out.astype(np.float32) + add, 0, 255)
            return Image.fromarray(outf.astype(np.uint8), "RGB")

        # --- foliage: strong canopy sway + flag/cloth ripple + fluttering leaves ---
        if kind == "foliage":
            arr = np.asarray(fr, dtype=np.uint8)
            yidx = np.arange(H, dtype=np.float32)
            ax   = np.arange(W)
            # (1) eased multi-harmonic canopy sway (upper band only)
            top_h  = 0.58 * H
            canopy = np.clip((top_h - yidx) / top_h, 0.0, 1.0)
            canopy = canopy * canopy * (3.0 - 2.0 * canopy)
            sway = ( np.sin(2.0 * np.pi * 0.11 * t)
                     + 0.45 * np.sin(2.0 * np.pi * 0.27 * t + yidx * 0.018)
                     + 0.25 * np.sin(2.0 * np.pi * 0.43 * t + 1.7) )
            AMP_TOP = 26.0
            swi = (sway * AMP_TOP * inten * canopy).astype(np.int32)
            out = arr.copy()
            for yy in np.nonzero(canopy > 1e-3)[0]:
                s = int(swi[yy])
                if s:
                    out[yy] = arr[yy][(ax - s) % W]
            # (2) localized flag/cloth ripple (narrow strips, deterministic per seed)
            srng = np.random.default_rng((sd) ^ 0x5EED)
            cx     = srng.uniform(0.12, 0.88, 3).astype(np.float32) * W
            cwid   = srng.uniform(34.0, 70.0, 3).astype(np.float32)
            cph    = srng.uniform(0.0, 6.2832, 3).astype(np.float32)
            cspeed = srng.uniform(0.55, 0.95, 3).astype(np.float32)
            ytop   = srng.uniform(0.05, 0.22, 3).astype(np.float32) * H
            yhgt   = srng.uniform(0.20, 0.40, 3).astype(np.float32) * H
            FLAG_AMP = 14.0
            after = out.copy()
            for kf in range(3):
                yc = ytop[kf] + 0.5 * yhgt[kf]
                vy = np.exp(-((yidx - yc) ** 2) / (2.0 * (0.6 * yhgt[kf]) ** 2))
                vy = vy * np.clip((0.66 * H - yidx) / (0.66 * H), 0.0, 1.0)
                cm = np.exp(-((ax - cx[kf]) ** 2) / (2.0 * cwid[kf] ** 2)).astype(np.float32)
                rr = np.nonzero(vy > 0.04)[0]
                if rr.size == 0:
                    continue
                for yy in rr:
                    phase = (2.0 * np.pi * cspeed[kf] * t + cph[kf] + yidx[yy] * 0.05)
                    d = FLAG_AMP * np.sin(phase) * vy[yy] * inten
                    if abs(d) < 0.5:
                        continue
                    di = int(round(d))
                    shifted = out[yy][(ax - di) % W]
                    w = cm * vy[yy]
                    after[yy] = (out[yy].astype(np.float32) * (1.0 - w[:, None])
                                 + shifted.astype(np.float32) * w[:, None]).astype(np.uint8)
            out = after
            # (3) drifting/fluttering leaf particles (7 anchors x4 tiled)
            a = out.astype(np.float32)
            bx, by, ph, sp, sz, sw, n = C["bx"], C["by"], C["ph"], C["sp"], C["sz"], C["sw"], C["n"]
            reps = 4
            tile_dx = srng.uniform(0, W, reps).astype(np.float32)
            tile_dp = srng.uniform(0, 6.2832, reps).astype(np.float32)
            bx_t = (np.add.outer(tile_dx, bx) % W).ravel()
            by_t = np.tile(by, reps)
            ph_t = (np.add.outer(tile_dp, ph)).ravel()
            sp_t = np.tile(sp, reps)
            sw_t = np.tile(sw, reps) * 1.6
            y = np.mod(by_t + sp_t * t * 46.0, H + 40.0) - 20.0
            x = bx_t + sw_t * np.sin(2.0 * np.pi * 0.33 * t + ph_t) \
                     + 8.0 * np.sin(2.0 * np.pi * 0.9 * t + ph_t * 2.0)
            twinkle = 0.55 + 0.45 * np.sin(2.0 * np.pi * 0.7 * t + ph_t * 3.0)
            acc = np.zeros((H, W), np.float32)
            yi = np.round(y).astype(np.intp); xi = np.round(x).astype(np.intp)
            m = (yi >= 1) & (yi < H - 1) & (xi >= 1) & (xi < W - 1)
            np.add.at(acc, (yi[m], xi[m]), (0.85 * inten) * twinkle[m])
            acc[1:, :] += 0.7 * acc[:-1, :]; acc[:-1, :] += 0.7 * acc[1:, :]
            acc[:, 1:] += 0.7 * acc[:, :-1]; acc[:, :-1] += 0.7 * acc[:, 1:]
            np.clip(acc, 0.0, 1.0, out=acc)
            a += acc[..., None] * np.array([126, 158, 74], np.float32)[None, None, :]
            np.clip(a, 0, 255, out=a)
            return Image.fromarray(a.astype(np.uint8), "RGB")

        # --- traffic: bidirectional light/vehicle streaks + lower-band shear ---
        if kind == "traffic":
            a = np.asarray(fr, dtype=np.float32)
            lane, x0, dr, spd, ln, thk = C["lane"], C["x0"], C["dir"], C["spd"], C["len"], C["thk"]
            warm, amp, n = C["warm"], C["amp"], C["n"]
            yc = (lane * H)
            base = (210.0 + 250.0 * spd)
            xh = np.mod(x0 + dr * base * t, W + 2.0 * ln.max()) - ln.max()
            STEPS = 30
            frt = np.linspace(0.0, 1.0, STEPS, dtype=np.float32)
            xs = xh[:, None] + (dr * ln)[:, None] * (frt[None, :] - 1.0)
            taper = (0.25 + 0.75 * frt[None, :])
            acc = np.zeros((H, W), np.float32); accW = np.zeros((H, W), np.float32)
            for j in range(n):
                xi = np.round(xs[j]).astype(np.intp)
                vis = (xi >= 0) & (xi < W)
                if not vis.any():
                    continue
                xi = xi[vis]; bw = (taper[0][vis] * amp[j]).astype(np.float32)
                tk = int(thk[j])
                for dy in range(-tk, tk + 1):
                    yy = int(round(yc[j])) + dy
                    if 0 <= yy < H:
                        rowg = (1.0 - abs(dy) / (tk + 1.0))
                        np.add.at(acc[yy], xi, bw * rowg)
                        if warm[j] > 0.5:
                            accW[yy][xi] += bw * rowg
            acc *= (0.5 * inten)
            np.clip(acc, 0.0, 1.4, out=acc)
            warm_tint = np.array([255, 210, 130], np.float32)
            cool_tint = np.array([150, 180, 230], np.float32)
            wmask = (accW > 1e-4).astype(np.float32)[..., None]
            tint = cool_tint[None, None, :] * (1.0 - wmask) + warm_tint[None, None, :] * wmask
            a += acc[..., None] * tint
            sw = C["shear_w"]; rows = C["shear_rows"]
            sh = (np.sin(2.0 * np.pi * 0.18 * t) * 9.0 * inten) * sw
            shi = np.round(sh).astype(np.intp)
            ax = np.arange(W)
            for yy in rows:
                d = int(shi[yy])
                if d:
                    a[yy] = a[yy][(ax - d) % W]
            np.clip(a, 0, 255, out=a)
            return Image.fromarray(a.astype(np.uint8), "RGB")

        # --- crowd/market: bobbing drifting silhouette blobs + lower shimmer ---
        if kind == "crowd":
            a = np.asarray(fr, dtype=np.float32)
            bx, by, ph, bsp = C["bx"], C["by"], C["ph"], C["bsp"]
            drift, rad, dark, n = C["drift"], C["rad"], C["dark"], C["n"]
            cx = np.mod(bx + drift * t + 10.0 * np.sin(0.7 * t + ph), W)
            cy = by + (4.0 + 3.0 * bsp) * np.sin(2.0 * np.pi * 0.55 * bsp * t + ph)
            shade = np.zeros((H, W), np.float32)
            for j in range(n):
                r = float(rad[j]); cxj = float(cx[j]); cyj = float(cy[j])
                x0i = max(0, int(cxj - r)); x1i = min(W, int(cxj + r + 1))
                y0i = max(0, int(cyj - r)); y1i = min(H, int(cyj + r + 1))
                if x0i >= x1i or y0i >= y1i:
                    continue
                xs = np.arange(x0i, x1i, dtype=np.float32) - cxj
                ys = np.arange(y0i, y1i, dtype=np.float32) - cyj
                g = np.exp(-((xs[None, :] ** 2) / (r * r) + (ys[:, None] ** 2) / (1.5 * r * r)))
                np.maximum(shade[y0i:y1i, x0i:x1i], g * dark[j], out=shade[y0i:y1i, x0i:x1i])
            shade *= (0.85 * inten)
            np.clip(shade, 0.0, 0.7, out=shade)
            a *= (1.0 - shade[..., None])
            sw = C["shim_w"]
            sh = (0.06 * inten) * (0.5 + 0.5 * np.sin(2.0 * np.pi * 0.4 * t)) * sw
            a += sh[:, None, None] * np.array([255, 200, 150], np.float32)[None, None, :]
            rows = C["shim_rows"]; ax = np.arange(W)
            shr = np.round(np.sin(2.0 * np.pi * 0.3 * t) * 4.0 * inten * sw).astype(np.intp)
            for yy in rows:
                d = int(shr[yy])
                if d:
                    a[yy] = a[yy][(ax - d) % W]
            np.clip(a, 0, 255, out=a)
            return Image.fromarray(a.astype(np.uint8), "RGB")

        return fr
    except Exception:
        return fr     # never let an env overlay crash a render


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
    if any(k in e for k in ["night", "dark", "midnight", "evening", "moon", "stars"]): return "night"
    if any(k in e for k in ["river", "water", "lake", "pond", "stream", "rain", "sea", "ocean", "waterfall", "well"]): return "water"
    if any(k in e for k in ["forest", "jungle", "woods", "tree", "grove"]): return "forest"
    if any(k in e for k in ["temple", "shrine", "mandir", "prayer", "worship", "sacred", "holy"]): return "temple"
    if any(k in e for k in ["crowd", "rally", "gathering", "public", "protest", "school", "class"]): return "crowd"
    if any(k in e for k in ["studio", "host", "anchor", "news"]): return "studio"
    if any(k in e for k in ["shop", "store"]): return "market"
    if any(k in e for k in ["village", "rural", "field", "farm", "hut", "outdoor", "garden"]): return "village"
    if any(k in e for k in ["kitchen", "home", "house", "room", "interior", "indoor", "bedroom", "library"]): return "indoor"
    return "village"


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
    work = Path(output_path).parent / "cartoon_work"
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)   # P0: never inherit a prior story's b{i}.jpg/_s*/_m* (was the all-stories-same cause)
    work.mkdir(parents=True, exist_ok=True)
    ND = str(work)
    # Per-EPISODE base seed (placeholder; REAL value DERIVED below from story content once `full` exists).
    EPISODE_SEED = 4242
    SCENE_STEP = 1013   # each scene gets its OWN seed -> every scene visually distinct (characters + background change per narration, storyboard-style); art STYLE stays uniform via CART, not the seed
    CART = (", clean flat 2D cartoon storybook illustration in ONE consistent cel-shaded art style and colour palette across the whole video, stylised NOT photoreal, NOT 3D, "
            "bold even black outlines, simple flat colour fills with soft cel shading, "
            "correct human anatomy with natural body proportions and a normal-sized head, each face drawn clearly with two symmetric eyes that each have a visible round pupil and iris, one nose, one mouth, natural hands with five fingers, "
            "ENVIRONMENT-FIRST composition: the LOCATION, setting and action fill most of the frame (about 70 percent) with a rich detailed background - depth, sky, weather, light and atmosphere - and any characters are SMALLER figures placed naturally INSIDE that world (about 30 percent), part of the scene and never filling the screen; prefer a WIDE or medium establishing shot of the place and do NOT default to a centred portrait, a posed group photo or a head-and-shoulders close-up unless the line is truly about one person's face, "
            "when a character is small or distant render them as a clean simplified silhouette-style figure with a plain readable face or no facial detail at all, NOT a shrunken detailed face, and never a crowd of tiny detailed faces - this keeps small figures from melting or going googly-eyed; for a lone, quietly-watching or contemplative subject who is NOT speaking, prefer showing that ONE figure FROM BEHIND or in three-quarter back view (back to camera) so no face can distort, and add NO extra uninvited bystanders and NO random wires, cables, ropes or rigging cluttering the sky beyond what the line actually names, "
            "characters and the background sharp and clearly separated, NOT blurry, no soft focus, no sketch lines, no crosshatch, no pencil shading, "
            "borderless full-bleed picture, completely plain unbranded artwork, no channel logo, no studio logo, no signage, no letters or words anywhere in the frame, "
            "clean professional 2D animation. Avoid: distorted or melted face, asymmetric or pupil-less or googly eyes, malformed hands, extra or fused fingers, stretched head, mutated anatomy, a giant character covering the whole frame, an empty plain background, a flat posed group-photo line-up, extra uninvited bystanders the line never mentioned, a busy crowd for a line about one person, random wires or cables cluttering the sky")

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
    # ---------- skip the NotebookLM host intro -> start at the real story narration ----------
    if len(segs) >= 4:
        seg_list = " ".join(f"[{int(s['start'])}s] {s['text']}" for s in segs[:18])[:1500]
        _ans = _ptext("This Hindi transcript has [seconds] tags and often begins with 1-2 hosts introducing / discussing "
                      "the topic before the real STORY narration starts. Reply with ONLY the integer second at which the "
                      "actual story narration begins (0 if it starts right away). Transcript: " + seg_list)
        _m = re.search(r"\d+", _ans or "")
        _S0 = float(_m.group(0)) if _m else 0.0
        if 3.0 <= _S0 < (segs[-1]["end"] - 8):
            segs = [{"start": s["start"] - _S0, "end": s["end"] - _S0, "text": s["text"]} for s in segs if s["end"] > _S0 + 0.2]
            start_offset = float(start_offset) + _S0
            prog(13, f"NotebookLM intro skip (~{int(_S0)}s) -- seedha kahani se shuru")
    TOTAL = min(TOTAL, segs[-1]["end"])

    def text_in(a, b): return " ".join(s["text"] for s in segs if s["end"] > a and s["start"] < b).strip()
    blocks = []; t = 0.0
    while t < TOTAL - 0.4:
        e = min(t + BLOCK, TOTAL); blocks.append({"i": len(blocks), "a": t, "b": e, "hi": text_in(t, e)}); t = e
    TOTAL = blocks[-1]["b"]; NB = len(blocks); NF = int(TOTAL * FPS)
    # Each block's DIRECTOR text defaults to the audio transcript (bk["hi"]). If the user pasted
    # the clean story SCRIPT, the director instead reads that text (scene/character/environment/
    # weather come from the TEXT); audio still drives timing, captions and lip-sync.
    for bk in blocks: bk["dir"] = bk["hi"]
    _script = (transcript or "").strip()
    if _script:
        _sents = [s.strip() for s in re.split(r"(?<=[।.?!])\s+|[\n\r]+", _script) if s.strip()]
        if _sents:
            _tot = float(sum(len(s) for s in _sents)) or 1.0; _cum = 0.0
            _buck = [[] for _ in blocks]
            for s in _sents:
                bi = min(NB - 1, int(((_cum + len(s) * 0.5) / _tot) * NB)); _buck[bi].append(s); _cum += len(s)
            for i, bk in enumerate(blocks):
                if _buck[i]: bk["dir"] = " ".join(_buck[i])

    # ---------- character bible (consistency without agents) ----------
    prog(16, "Kahani ke characters samajhe ja rahe hain...")
    full = ((transcript or "").strip()[:2000] or " ".join(s["text"] for s in segs)[:1600])   # pasted text script (if any) drives scene understanding; audio still drives timing/captions/lip-sync
    # P1: hash the DECODED audio PCM (so different stories diverge) + a per-render token (so re-runs differ).
    _audio_sig = hashlib.sha256(np.ascontiguousarray(audio16).tobytes()).hexdigest()[:16]
    _run_salt = f"{output_path}|{time.time_ns()}|{os.getpid()}"
    _seed_src = (str(topic) + "|" + full + "|" + _audio_sig + "|" + _run_salt).encode("utf-8", "replace")
    EPISODE_SEED = int(hashlib.sha256(_seed_src).hexdigest(), 16) % 2_000_000
    bible = _ptext("From this Hindi story, briefly note the kinds of people/characters that appear across it (e.g. 'a young soldier; an elderly farmer; village women; a child; a police officer') so a scene director knows who exists. One short line, no preamble. "
                   "Transcript: " + (topic + ". " + full if topic else full))[:500]
    # loose cast hint for the scene director (NOT a forced single hero)
    CAST = " ".join(bible.split())[:320] or "scene-appropriate characters"

    # ---------- whole-story visual world (computed ONCE, before the per-line loop) ----------
    # Understand the WHOLE story first, THEN fix the world so every scene's location/era/
    # weather stays consistent with the overall arc (cheap: one extra text call per episode).
    STORY = _pjson(
        "Read this WHOLE Hindi story ONCE, then summarise its overall VISUAL WORLD so a "
        "storyboard director keeps every scene's location, era and weather consistent. "
        "Reply with ONLY this JSON: "
        '{"places":"<2-4 main physical locations the story actually happens in, comma list>",'
        '"era":"<modern day | a specific past period | timeless village - one short phrase>",'
        '"region":"<broad setting e.g. rural north-india village, indian metro city, himalayan border>",'
        '"weather":"<dominant weather/season/time-of-day the narration supports e.g. monsoon rain, dry summer afternoon, cold foggy night>",'
        '"mood":"<one or two words for the overall emotional tone>",'
        '"motifs":"<3-6 recurring environmental things to keep showing e.g. wet roads, tea stall, paddy fields, army trucks>"}. '
        "Story: " + (topic + ". " + full if topic else full)
    )
    _sp = lambda k, d: " ".join(str(STORY.get(k, "") or "").split())[:120] or d
    STORY_REGION  = _sp("region",  "India")
    STORY_WEATHER = _sp("weather", "natural daytime weather")
    STORY_WORLD = (f"WHOLE-STORY WORLD (decided from the entire narration; keep every scene consistent with this): "
                   f"region={STORY_REGION}; era={_sp('era','modern day')}; main locations={_sp('places','the storys main locations')}; "
                   f"dominant weather/season={STORY_WEATHER}; overall mood={_sp('mood','')}; recurring motifs={_sp('motifs','')}. "
                   f"This scene must sit in one of those locations (or a believable spot in the same world), and its sky/light/weather "
                   f"must fit the dominant weather unless THIS line explicitly says otherwise. ")

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
    if not bible.strip():   # P2: never let the cast anchor collapse to empty on a text-endpoint hiccup
        bible = ("characters from this Hindi story: " + (str(topic) + " " if topic else "") + full[:240]).strip()
    CHAR_LEDGER = {}   # ckey -> locked look string (same person -> same look across scenes)
    LEDGER_ORDER = []  # recency of characters seen
    prev_recap = ""
    for i, bk in enumerate(blocks):
        bk["kind"] = "cut"   # pure STORY visuals, NO talking-head anchor (per approved demo)
        bk["salt"] = _salt(topic, full[:120], i)
        bk["mus"] = _pick_music(bk["hi"])
        if bk["kind"] == "host":
            bk["host"] = gender(bk["a"], bk["b"]); bk["sfx"] = "studio"; bk["emo"] = "neutral"; bk["emoI"] = 0.5
        else:
            prog(18 + int(20 * i / max(1, NB)), f"Scene {i+1}/{NB} ka background analyze ho raha hai...")
            _known = "; ".join(f"{kk} -> {CHAR_LEDGER[kk]}" for kk in LEDGER_ORDER[-6:]) or "none yet"
            _ctx = (f"CONTINUITY MEMORY (reuse these EXACT looks, do NOT redesign an already-shown person): {_known}. "
                    f"Previous scene was: {prev_recap or 'none'}. ")
            j = _pjson(f"You are an ENVIRONMENT-FIRST STORYBOARD director for a Hindi story. Topic: {topic or 'Hindi kahaniya'}. People who may appear: {bible}. " + STORY_WORLD + _ctx +
                       "First understand the WHOLE story (the world above); THEN build THIS line as a PLACE, not a portrait. ORDER OF THINKING: (1) WHERE is it happening - pick a SPECIFIC location; (2) WHAT action/event happens there; (3) build a RICH living environment for it; (4) ONLY THEN place any characters NATURALLY inside it, SMALL. "
                       "FRAMING RULE: default to a WIDE / ESTABLISHING shot that clearly shows the location, with the environment filling about 70 percent of the frame and any people about 30 percent (small, part of the scene, NOT filling the screen, NOT centred portraits, NOT a posed group photo). Use a closer MEDIUM (waist-up) shot ONLY for a single genuine emotional or dialogue beat; never default to close-ups. Characters are PART of the scene, never the whole scene. "
                       "DEFAULT TO no_person: if the line is about a place, the weather, time passing, an object, an event, a mood, a journey, prices or an idea - anything NOT a specific person acting/speaking/feeling right now - choose scene_kind=no_person and draw a PURE LOCATION / OBJECT / ESTABLISHING shot (no people, or tiny incidental distant figures only). It is GOOD for roughly half the scenes to be no_person establishing shots; do NOT add a decorative person just to fill the frame. "
                       "WHO (only after the place is set, and only when the line is genuinely about a person): a single explicit subject (a named person, first-person 'main', or an aloneness cue 'akela/sunsaan/deserted') = EXACTLY ONE small person inside the wide scene, no extra bystanders; for such a lone subject when they are NOT speaking (a quiet watching/thinking/standing beat or 'akela/alone'), prefer framing that one person FROM BEHIND or in three-quarter back view, small-to-medium, looking into the scene - it reads cinematic and avoids any face distortion, and keep the frame clean with no random wires or clutter (if the character is actually speaking, keep them facing forward instead). Hindi PLURALS (verb endings -gaye/-ruke/-aaye/-rahe the, words log/sab/logon, group nouns) and any number (do/teen/paanch/kai/bheed) = MULTIPLE distinct people scattered NATURALLY through the environment (about that many, cap 7); soldiers/jawans/squad/patrol/troops = several soldiers spread across the location; a named officer or hero still appears WITH his men/team/family inside the wide scene. A place word used only as a destination/backdrop for one named person ('apne gaon ki or') = distant/near-empty scenery, not a crowd. NEVER render 'a group of people standing in a room' unless the narration literally describes a meeting/gathering. "
                       "When people ARE present keep them SMALL and well-spaced; draw AT MOST 4 figures readably and render any larger group as soft simplified distant background figures - NEVER many tiny detailed faces. Even small, draw each visible face cleanly: a normal-sized head, two symmetric eyes each with a round pupil, one nose, one mouth, natural hands with five fingers - no distortion. "
                       "LOCK each person's GENDER (name, kinship maa/beti/pita/beta, or gendered verb dekha/dekhi), AGE (baccha/nanha=child, naujawan=young, buzurg/budha/dadi=elderly) and PROFESSION attire (jawan=fatigues+rifle, police=khaki+cap, farmer=dhoti+field tools, doctor=white coat, judge=black robe); never default to a generic middle-aged male. "
                       "WHERE = a SPECIFIC setting: sub-location + time-of-day + weather/season. Match any stated time/weather (night/dawn/rain/fog/heat) in the lighting and sky, keep ALL concrete nouns the line hinges on in-frame, give every location its own detail/motion/atmosphere, and make every scene's backdrop different; never default to a generic living room. LOCATION DETAIL (once you pick the setting, fill 'environment' and the scene with its signature props/weather; put the CAPS keyword into 'environment' so the matching ambience fires): RAIN/MONSOON -> grey clouds, slanting rain, wet shining roads, puddles, dripping rooftops, soaked swaying trees; WIDE. VILLAGE/FIELD/FARM -> mud-thatch huts, green or golden fields, a well, cattle, a banyan tree, dawn/dusk haze; WIDE. MARKET/BAZAAR -> rows of stalls, hanging goods, produce baskets, hand-carts, a stream of customers; MEDIUM-WIDE. BORDER/military -> a fenced border post, sandbag bunkers, jeeps, a flag, a watchtower, patrolling soldiers as small figures; WIDE. TEMPLE/MANDIR -> carved stone spire, an idol, oil lamps, marigold garlands, incense smoke, devotees; WIDE-to-MEDIUM. FOREST/JUNGLE -> dense trees, light shafts, undergrowth, a winding trail, mist; WIDE. CITY STREET/ROAD -> tall buildings, moving traffic and auto-rickshaws, streetlights and wires, pavements with pedestrians (no readable text); WIDE. RIVER/WATER -> flowing water with reflections, a ghat or bridge, boats, reeds; WIDE. SCHOOL/CLASS -> classroom or playground, blackboard/desks or a courtyard, children in uniform, a flagpole; MEDIUM-WIDE. COURTROOM -> tall judge's bench, witness stand, benches, law books; MEDIUM-WIDE. HOME/ROOM (ONLY when truly indoors) -> a specific lived-in room with furniture, a window with light, warm lamp glow; MEDIUM, do NOT default here for outdoor lines. Add NIGHT for a dark blue sky, moon, lit windows, long shadows. For any other setting give 4-6 concrete signature props plus a weather/time cue and keep small figures inside a wide living world. "
                       "An unseen/ambiguous subject ('something moved', 'a shadow', 'koi aahat') stays SUGGESTED (silhouette/shadow/off-frame), not a clear person. For no_person / explanatory lines pick a B-ROLL establishing shot echoing the subject: army/border/war -> a guarded border post, tanks, a wide patrol shot (no single hero); rain/storm/monsoon -> heavy clouds, rain on empty streets, rising water; city/economy/market/prices -> traffic and a skyline, a busy market, rupee notes; village/farm/harvest -> open fields, crops, a lone hut at the right time-of-day; hope -> a lit diya in darkness. A clean wide or medium establishing frame of the place or object, not a posed person. "
                       "EMOTION = the emotional CORE of the key event, carried by BOTH the environment (light/weather/colour/atmosphere) AND any small figures (a lost child in a busy bazaar is tense/sad, not happy); if the line turns (despair->hope) use the RESOLVING end-state and keep both cues visible; give each visible person their own correct expression. "
                       "For consecutive lines about the SAME person, hold their gender/age/build/clothing constant; only the location, action and emotion change. "
                       "If a person here is ALREADY in CONTINUITY MEMORY, REUSE that person's exact gender/age/build/clothing/colours verbatim; only their action, the scene and emotion may change. "
                       "Reply with ONLY a JSON object: "
                       '{"environment":"<specific setting + time-of-day + weather for THIS line>",'
                       '"scene_kind":"<one_person | multiple_people | no_person>",'
                       '"who":"<short stable identity tag of the PRIMARY person if one_person (e.g. ravi soldier); empty otherwise>",'
                       '"look":"<if one_person: that persons locked visual age and gender and build and hair and clothing and colours and props; empty otherwise>",'
                       '"prompt":"<one clean stylised 2D cartoon storyboard scene, NOT photoreal, built ENVIRONMENT-FIRST and mostly WIDE: OPEN by describing the LOCATION and its signature props, weather and atmosphere (the rich background fills about 70 percent of the frame), THEN place any small-to-medium character(s) naturally inside it (at most 4 readable, any extra as soft distant background figures; for no_person draw no people or only tiny distant figures; a closer waist-up ONLY for a single intimate emotional/dialogue beat) - never a centred portrait and never a posed group photo unless the line is a meeting; each visible person keeps locked gender/age/profession look and their own expression, well-proportioned with a normal-sized head, two symmetric eyes with visible pupils, one nose, one mouth, natural hands with five fingers, small/distant figures drawn as clean simplified shapes not shrunken detailed faces; for a lone non-speaking subject prefer a from-behind or over-the-shoulder back view and add no uninvited bystanders or random wire/cable clutter; clean professional 2D-animation linework, bold black outlines, no distortion, no logo and no letters anywhere>",'
                       '"emotion":"<neutral/happy/hopeful/calm/uneasy/tense/sad/angry/surprise/emphatic>",'
                       '"char_voice":"<none | child | elderly_female | elderly_male | adult_male | adult_female>",'
                       '"char_line":"<if the line is clearly a character speaking, the SHORT Hindi sentence they say; otherwise empty>"}. '
                       'Use char_voice for a person ONLY when the narration is clearly that character speaking; otherwise "none" and empty char_line. '
                       f'Narration line: {bk["dir"]}')
            bk["env"] = j.get("environment") or (STORY_REGION + ", " + STORY_WEATHER)   # director hiccup -> stay in the story's world, never a generic room
            sk = (j.get("scene_kind") or "").strip().lower()
            bk["scene_kind"] = sk if sk in ("one_person", "multiple_people", "no_person") else ""
            who = _ckey(j.get("who")); look = " ".join((j.get("look") or "").split())[:240]
            bk["ckey"] = who if bk["scene_kind"] == "one_person" else ""
            if bk["ckey"]:
                if bk["ckey"] in CHAR_LEDGER: look = CHAR_LEDGER[bk["ckey"]] or look   # P2: reuse established look verbatim
                elif look: CHAR_LEDGER[bk["ckey"]] = look
                if bk["ckey"] in LEDGER_ORDER: LEDGER_ORDER.remove(bk["ckey"])
                LEDGER_ORDER.append(bk["ckey"])
            _scene = (j.get("prompt") or ("a wide establishing environment-first 2D cartoon shot set in " + STORY_REGION +
                      " during " + STORY_WEATHER + ", the rich location and weather fill the frame, only small distant figures if any, depicting: " + bk["dir"][:70]))
            if bk["ckey"] and look: _scene = _scene + ". The SAME recurring person, consistent design: " + look
            bk["prompt"] = (_scene + ", set in a detailed " + str(bk["env"]) +
                            " environment - a distinct background matching THIS scene, consistent with the story world (" +
                            STORY_REGION + ", " + STORY_WEATHER + "), the environment fills most of the frame with rich "
                            "location detail and atmosphere, cast context: " + CAST + CART)
            bk["sfx"] = env_ambient(bk["env"]); bk["emo"] = j.get("emotion", "neutral"); bk["emoI"] = 0.6
            bk["mus"] = _emo_music(bk["emo"], bk["mus"])
            cv = (j.get("char_voice") or "none").strip(); cl = (j.get("char_line") or "").strip()
            bk["cvoice"] = cv if (cv != "none" and len(cl) >= 3) else "none"; bk["cline"] = cl if bk["cvoice"] != "none" else ""
            _whotxt = (look or ("several people" if bk["scene_kind"] == "multiple_people"
                       else ("no people, scenery only" if bk["scene_kind"] == "no_person" else bk["dir"][:50])))
            prev_recap = (f"{_whotxt} at {bk['env']}")[:200]

    # ---------- cutaway images ----------
    cut_idx = [i for i, bk in enumerate(blocks) if bk["kind"] == "cut"]
    def fetch_img(i):
        ip = os.path.join(ND, f"b{i}.jpg")
        if os.path.exists(ip) and os.path.getsize(ip) > 5000: return True
        _ck = blocks[i].get("ckey") or ""
        if _ck:
            scene_seed = (EPISODE_SEED + _salt(_ck)) % 2_000_000                                  # P2e: same person -> stable face
        else:
            scene_seed = (EPISODE_SEED + i * SCENE_STEP + blocks[i].get("salt", 0)) % 2_000_000   # crowd/no-person -> per-scene distinct
        ATTEMPTS = 5
        for a in range(ATTEMPTS):
            seed = (scene_seed + a * 131071) % 2_000_000
            url = ("https://image.pollinations.ai/prompt/" + urllib.parse.quote(blocks[i]["prompt"]) +
                   f"?width=1280&height=720&seed={seed}&model=flux&nologo=true&enhance=false&private=true")
            wait = 0.0
            try:
                r = requests.get(url, timeout=120)
                if r.status_code == 200 and len(r.content) > 5000: open(ip, "wb").write(r.content); return True
                if r.status_code in (429, 500, 502, 503, 504):
                    try: wait = float(r.headers.get("Retry-After") or 0.0)
                    except Exception: wait = 0.0
            except Exception:
                pass
            if a == ATTEMPTS - 1: break
            backoff = min(90.0, 8.0 * (2 ** a))                  # P4: 8,16,32,64,90 instead of flat 5s
            if wait > backoff: backoff = min(180.0, wait)         # honour Retry-After
            backoff += (_salt(str(i)) % 1000) / 1000.0 * 3.0      # jitter de-syncs workers
            time.sleep(backoff)
        return False
    prog(40, "Scene backgrounds generate ho rahe hain...")
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=2) as ex: list(ex.map(fetch_img, cut_idx))
    def have(i): return os.path.exists(os.path.join(ND, f"b{i}.jpg")) and os.path.getsize(os.path.join(ND, f"b{i}.jpg")) > 5000
    for _pass in range(2):                                   # P4: extra serial retries; limiter usually relaxes after the burst
        miss = [i for i in cut_idx if not have(i)]
        if not miss: break
        for i in miss: fetch_img(i)
    exist = [i for i in cut_idx if have(i)]
    missing = [i for i in cut_idx if not have(i)]
    if cut_idx and (not exist or len(missing) > max(2, len(cut_idx) // 3)):   # P3: never ship a one-frame video
        raise RuntimeError(f"Pollinations rate-limited: sirf {len(exist)}/{len(cut_idx)} scene images bani. "
                           "Video NAHI banayi (taaki sab same frame na ho). Thodi der baad dobara try karein.")
    def _placeholder(i):                                     # P3: scene-DISTINCT gradient, NOT a copy of a neighbour
        sd = (EPISODE_SEED + i * SCENE_STEP + blocks[i].get("salt", 0)) % 2_000_000
        rnd = np.random.RandomState(sd)
        c0 = tuple(int(x) for x in rnd.randint(25, 210, 3)); c1 = tuple(int(x) for x in rnd.randint(25, 210, 3))
        gx = (np.linspace(0, 1, 1280)[None, :] + np.linspace(0, 1, 720)[:, None]) / 2.0
        base = np.zeros((720, 1280, 3), np.uint8)
        for ch in range(3): base[..., ch] = (c0[ch] * (1 - gx) + c1[ch] * gx).astype(np.uint8)
        im = Image.fromarray(base, "RGB"); dr = ImageDraw.Draw(im)
        for _ in range(6):
            x0, y0 = rnd.randint(0, 1150), rnd.randint(0, 640)
            dr.ellipse([x0, y0, x0 + rnd.randint(60, 280), y0 + rnd.randint(60, 280)],
                       fill=tuple(int(x) for x in rnd.randint(20, 235, 3)))
        im.save(os.path.join(ND, f"b{i}.jpg"), "JPEG", quality=88)
    if missing:
        prog(40, f"WARNING: {len(missing)}/{len(cut_idx)} scenes distinct placeholder se bhare (Pollinations partial fail).")
        for i in missing: _placeholder(i)

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
        im = ImageEnhance.Contrast(im).enhance(1.08); im = ImageEnhance.Color(im).enhance(1.10)
        im = im.filter(ImageFilter.UnsharpMask(radius=1.3, percent=80, threshold=3))
        s = max(W / im.width, H / im.height) * 1.14
        return im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    CUT = {i: fit(os.path.join(ND, f"b{i}.jpg")) for i in cut_idx if have(i)}
    CAMS = ["push_in", "pull_out", "pan_left", "pan_right", "static_drift"]
    def _ease(p):
        p = 0.0 if p < 0 else (1.0 if p > 1 else p)
        return p * p * (3.0 - 2.0 * p)                                             # smoothstep ease-in/out
    def cam(img, p, mode, t=0.0, scene=0):
        h = (EPISODE_SEED * 2654435761 + scene * 40503 + 0x9E3779B1) & 0xFFFFFFFF   # per-scene deterministic variety
        r0 = ((h >> 8) & 0xFFFF) / 65535.0; r1 = ((h >> 16) & 0xFFFF) / 65535.0
        amp = 0.85 + 0.45 * r0; phab = r1 * 6.28318; pe = _ease(p)
        strong = 1.35 if r0 > 0.75 else 1.0                                        # ~25% of scenes get a bolder move
        if   mode == "push_in":  z = 1.05 + 0.10 * amp * strong * pe; ox = oy = 0.5
        elif mode == "pull_out": z = 1.15 - 0.10 * amp * strong * pe; ox = oy = 0.5
        elif mode == "pan_left": z = 1.10; ox = 0.62 - 0.26 * amp * pe; oy = 0.5
        elif mode == "pan_right":z = 1.10; ox = 0.38 + 0.26 * amp * pe; oy = 0.5
        else:                    z = 1.07 + 0.018 * amp * pe; ox = 0.5; oy = 0.5
        z += 0.010 * amp * math.sin(t * 2 * math.pi * 0.28 + phab)                  # subtle breathing -> alive feel
        ox = min(1, max(0, ox + 0.006 * amp * math.sin(t * 2 * math.pi * 0.19 + phab)))  # gentle sway
        oy = min(1, max(0, oy + 0.010 * amp * math.sin(t * 2 * math.pi * 0.22 + phab)))
        cw, ch = int(W / z), int(H / z); px = int((img.width - cw) * ox); py = int((img.height - ch) * oy)
        px = max(0, min(img.width - cw, px)); py = max(0, min(img.height - ch, py))
        return img.crop((px, py, px + cw, py + ch)).resize((W, H), Image.LANCZOS)
    try: bfont = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 24)
    except Exception: bfont = ImageFont.load_default()
    def brand(c):
        return   # watermark disabled (was hardcoded "THE MYSTERY" on every frame)
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
            img = CUT.get(k) or (next(iter(CUT.values())) if CUT else Image.new("RGB", (W, H), (20, 20, 30))); fr = cam(img, p, CAMS[(k * 7 + 2) % 5], tt, scene=k); fr = env_motion(fr, bk.get("sfx", ""), tt, (EPISODE_SEED + k * SCENE_STEP + bk.get("salt", 0)) % 2_000_000, bk.get("emo", "neutral"), bk.get("env", ""))
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
    def _seed_noise(ins, base):                              # P5: seed anoisesrc so noise differs per scene & per story
        out = []
        for n, s in enumerate(ins):
            if s.startswith("anoisesrc="): s = s + (":s=%d" % ((base + n * 2654435761) % 2_000_000_000))
            out.append(s)
        return out
    def _jit(fc, jf, jd):
        return fc[:fc.rfind("[o]")] + (",tremolo=f=%.3f:d=%.2f" % (jf, jd)) + "[o]"
    def sfx_bed(tag, d, out, salt=0, idx=0):
        T = {
         "night": (["anoisesrc=d=%s:c=brown:a=0.35" % d, "sine=f=900:d=%s" % d], "[0:a]lowpass=f=380,highpass=f=55,volume=0.9[a];[1:a]tremolo=f=6:d=1.0,highpass=f=700,volume=0.07[b];[a][b]amix=inputs=2:normalize=0[o]"),
         "water": (["anoisesrc=d=%s:c=white:a=0.55" % d, "anoisesrc=d=%s:c=brown:a=0.35" % d], "[0:a]highpass=f=300,lowpass=f=3200,tremolo=f=1.1:d=0.6,volume=0.7[a];[1:a]lowpass=f=500,tremolo=f=0.4:d=0.5,volume=0.8[b];[a][b]amix=inputs=2:normalize=0[o]"),
         "forest": (["anoisesrc=d=%s:c=brown:a=0.4" % d, "sine=f=3200:d=%s" % d], "[0:a]lowpass=f=700,highpass=f=80,tremolo=f=0.3:d=0.6,volume=0.9[a];[1:a]vibrato=f=9:d=1.0,tremolo=f=6:d=1.0,highpass=f=2400,volume=0.12[b];[a][b]amix=inputs=2:normalize=0[o]"),
         "temple": (["anoisesrc=d=%s:c=brown:a=0.22" % d, "sine=f=130.81:d=%s" % d], "[0:a]lowpass=f=360,volume=0.5[a];[1:a]tremolo=f=0.25:d=0.6,aecho=0.8:0.6:600:0.4,volume=0.14[b];[a][b]amix=inputs=2:normalize=0[o]"),
         "indoor": (["anoisesrc=d=%s:c=brown:a=0.26" % d, "sine=f=115:d=%s" % d], "[0:a]lowpass=f=430,volume=0.5[a];[1:a]volume=0.04[b];[a][b]amix=inputs=2:normalize=0[o]"),
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
        ins, fc = T.get(tag, (["anoisesrc=d=%s:c=brown:a=0.45" % d], "[0:a]lowpass=f=600,highpass=f=70,volume=1.0[o]"))
        ins = _seed_noise(ins, (EPISODE_SEED + salt + idx * 7919) % 2_000_000_000)
        jf = 0.10 + (salt % 17) * 0.02; jd = 0.16 + (salt // 17 % 7) * 0.06
        cut = 220 + (salt % 23) * 86                          # salt-chosen lowpass -> audible timbre shift per scene
        fc2 = fc[:fc.rfind("[o]")] + (",lowpass=f=%d,tremolo=f=%.3f:d=%.2f" % (cut, jf, jd)) + "[o]"
        lavfi(ins, fc2, d, out)
    def music_bed(mood, d, out, salt=0, idx=0):
        M = {
         "triumphant": (["sine=f=261.63:d=%s" % d, "sine=f=329.63:d=%s" % d, "sine=f=392.0:d=%s" % d, "sine=f=523.25:d=%s" % d], "[0:a]volume=0.18[a];[1:a]volume=0.15[b];[2:a]volume=0.15[c];[3:a]volume=0.09[e];[a][b][c][e]amix=inputs=4:normalize=0,tremolo=f=0.3:d=0.2,aecho=0.8:0.5:300:0.3[o]"),
         "soft_emotional": (["sine=f=220:d=%s" % d, "sine=f=261.63:d=%s" % d, "sine=f=329.63:d=%s" % d], "[0:a]volume=0.14[a];[1:a]volume=0.12[b];[2:a]volume=0.10[c];[a][b][c]amix=inputs=3:normalize=0,tremolo=f=0.2:d=0.25,aecho=0.9:0.4:420:0.35,lowpass=f=2200[o]"),
         "tense_suspense": (["sine=f=73.42:d=%s" % d, "sine=f=77.78:d=%s" % d, "anoisesrc=d=%s:c=pink:a=0.05" % d], "[0:a]volume=0.30[a];[1:a]volume=0.18[b];[2:a]lowpass=f=900,volume=0.4[c];[a][b][c]amix=inputs=3:normalize=0,tremolo=f=0.8:d=0.4,aecho=0.9:0.5:500:0.4[o]"),
         "action_cinematic": (["sine=f=55:d=%s" % d, "sine=f=110:d=%s" % d, "sine=f=164.81:d=%s" % d, "anoisesrc=d=%s:c=brown:a=0.25" % d], "[0:a]volume=0.32,tremolo=f=2.0:d=0.85[a];[1:a]volume=0.18,tremolo=f=2.0:d=0.6[b];[2:a]volume=0.14[c];[3:a]lowpass=f=180,volume=0.6,tremolo=f=2.0:d=0.8[d];[a][b][c][d]amix=inputs=4:normalize=0,aecho=0.8:0.4:250:0.3[o]"),
        }
        ins, fc = M.get(mood, (["sine=f=60:d=%s" % d], "[0:a]volume=0.05[o]"))
        ins = _seed_noise(ins, (EPISODE_SEED + salt + idx * 6271) % 2_000_000_000)
        k = (salt % 8) - 3; ratio = 2.0 ** (k / 12.0)        # transpose chord by a salt-chosen interval
        ins = [(re.sub(r"f=([0-9.]+)", lambda m: "f=%.3f" % (float(m.group(1)) * ratio), s) if s.startswith("sine=") else s) for s in ins]
        jf = 0.12 + (salt % 13) * 0.02; jd = 0.18 + (salt // 13 % 6) * 0.05
        fc2 = fc[:fc.rfind("[o]")] + (",tremolo=f=%.3f:d=%.2f" % (jf, jd)) + "[o]"
        lavfi(ins, fc2, d, out)
    sp = []; mp = []
    for i, bk in enumerate(blocks):
        slt = bk.get("salt", _salt(topic, i))
        d = bk["b"] - bk["a"]; fs = os.path.join(ND, f"_s{i}.wav"); sfx_bed(bk["sfx"], d, fs, slt, i); sp.append(fs)
        fm = os.path.join(ND, f"_m{i}.wav"); music_bed(bk["mus"], d, fm, slt, i); mp.append(fm)
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
           "[1:a]volume=1.28,asplit=3[vmain][vk1][vk2];"
           "[2:a]volume=0.40[sfx];[sfx][vk1]sidechaincompress=threshold=0.05:ratio=8:attack=15:release=350[sfxd];"
           "[3:a]volume=0.22[mus];[mus][vk2]sidechaincompress=threshold=0.06:ratio=5:attack=20:release=400[musd];"
           "[vmain][sfxd][musd]amix=inputs=3:normalize=0:duration=first[mx];[mx]alimiter=limit=0.90[ao]",
           "-map", "0:v", "-map", "[ao]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", output_path])
    else:
        runtext([FF, "-y", "-i", vid, "-i", NARR_USE,"-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", output_path])
    if not (os.path.exists(output_path) and os.path.getsize(output_path) > 50000):
        raise RuntimeError("Final cartoon video mux nahi ho paaya.")
    prog(100, "Done")
    return TOTAL
