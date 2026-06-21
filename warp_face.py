# Face-WARP rig: instead of pasting an ellipse, we move control points on the character's OWN face
# (lips, jaw, brows, eyelids, cheeks) and warp the image (thin-plate-spline) so the mouth/expression is
# part of the face -> no cut-out / border / overlap. Visemes (A/E/O/U/M) + emotions. Soft mouth interior.
import os, io, sys, math
import numpy as np, cv2
from scipy.interpolate import RBFInterpolator
from PIL import Image
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
ND = r"C:\Users\Sameer\Desktop\notebooklm-video-builder\prototype_assets\cartoon"
SZ = 768

# control points (normalized) per host (pinned from the face grids)
LM_MALE = {   # clean-shaven host_m2 (beard removed -> mouth clearly visible)
 "brow_in_L":(0.465,0.305),"brow_out_L":(0.430,0.315),"brow_in_R":(0.535,0.305),"brow_out_R":(0.575,0.315),
 "lid_L":(0.455,0.338),"lid_R":(0.550,0.338),"cheek_L":(0.440,0.450),"cheek_R":(0.560,0.450),
 "mouth_L":(0.470,0.470),"mouth_R":(0.530,0.470),"lip_top":(0.500,0.458),"lip_bot":(0.500,0.480),"jaw":(0.500,0.560),
}
LM_FEMALE = {
 "brow_in_L":(0.465,0.285),"brow_out_L":(0.420,0.292),"brow_in_R":(0.535,0.285),"brow_out_R":(0.580,0.292),
 "lid_L":(0.450,0.312),"lid_R":(0.555,0.312),"cheek_L":(0.430,0.435),"cheek_R":(0.565,0.435),
 "mouth_L":(0.465,0.462),"mouth_R":(0.535,0.462),"lip_top":(0.500,0.450),"lip_bot":(0.500,0.473),"jaw":(0.500,0.552),
}
LM = LM_MALE                 # default (used by the __main__ test)
MX, MY = 0.500, 0.470        # male (clean-shaven) mouth center
MXY = {"m":(0.500,0.470), "f":(0.500,0.462)}
# anchors that stay fixed -> localize the warp to the face
ANCH = [(0.0,0.0),(0.5,0.0),(1.0,0.0),(0.0,0.5),(1.0,0.5),(0.0,1.0),(0.5,1.0),(1.0,1.0),
        (0.487,0.165),(0.487,0.235),(0.30,0.34),(0.70,0.34),(0.487,0.62),(0.33,0.50),(0.64,0.50),
        (0.27,0.43),(0.71,0.43),(0.20,0.80),(0.80,0.80),(0.487,0.40)]   # incl nose-base fixed-ish

# viseme displacements (fraction of face); mouth-opening ones get scaled by openness
VIS = {   # bigger / more expressive openings (clean-shaven mouth is visible now)
 "REST":{},
 "A":{"lip_bot":(0,0.064),"jaw":(0,0.048),"mouth_L":(0.005,0.008),"mouth_R":(-0.005,0.008),"lip_top":(0,-0.009)},
 "E":{"mouth_L":(-0.020,0.003),"mouth_R":(0.020,0.003),"lip_bot":(0,0.019),"lip_top":(0,-0.006)},
 "O":{"mouth_L":(0.022,0.0),"mouth_R":(-0.022,0.0),"lip_bot":(0,0.040),"lip_top":(0,-0.019),"jaw":(0,0.022)},
 "U":{"mouth_L":(0.024,0.0),"mouth_R":(-0.024,0.0),"lip_bot":(0,0.024),"lip_top":(0,-0.009)},
 "MBP":{"lip_top":(0,0.005),"lip_bot":(0,-0.005)},
}
# emotion displacements (fraction of face); scaled by intensity
EMO = {
 "neutral":{},
 "happy":{"mouth_L":(-0.009,-0.011),"mouth_R":(0.009,-0.011),"cheek_L":(0,-0.009),"cheek_R":(0,-0.009),"lid_L":(0,0.003),"lid_R":(0,0.003)},
 "hopeful":{"mouth_L":(-0.005,-0.007),"mouth_R":(0.005,-0.007),"brow_in_L":(0,-0.005),"brow_in_R":(0,-0.005)},
 "uneasy":{"brow_in_L":(0.004,-0.005),"brow_in_R":(-0.004,-0.005),"mouth_L":(0,0.005),"mouth_R":(0,0.005)},
 "tense":{"brow_in_L":(0.004,0.005),"brow_in_R":(-0.004,0.005),"lid_L":(0,-0.003),"lid_R":(0,-0.003)},
 "emphatic":{"brow_in_L":(0,-0.006),"brow_in_R":(0,-0.006),"lid_L":(0,-0.004),"lid_R":(0,-0.004)},
 "sad":{"brow_in_L":(0,-0.007),"brow_in_R":(0,-0.007),"mouth_L":(0,0.009),"mouth_R":(0,0.009)},
 "surprise":{"brow_in_L":(0,-0.013),"brow_out_L":(0,-0.011),"brow_in_R":(0,-0.013),"brow_out_R":(0,-0.011),"lid_L":(0,-0.009),"lid_R":(0,-0.009)},
}

GX = 90
gx = np.linspace(0, SZ-1, GX); gridx, gridy = np.meshgrid(gx, gx)
GRID = np.stack([gridx.ravel(), gridy.ravel()], 1).astype(np.float64)
XX, YY = np.meshgrid(np.arange(SZ, dtype=np.float32), np.arange(SZ, dtype=np.float32))

def warp(base, disp, lm=LM):  # disp: dict point->(dx,dy) in fraction units
    src=[]; dst=[]
    for k,(nx,ny) in lm.items():
        px,py=nx*SZ,ny*SZ; dx,dy=disp.get(k,(0.0,0.0))
        src.append((px,py)); dst.append((dx*SZ, dy*SZ))
    for (nx,ny) in ANCH:
        src.append((nx*SZ,ny*SZ)); dst.append((0.0,0.0))
    src=np.array(src,np.float64); dst=np.array(dst,np.float64)
    rbf=RBFInterpolator(src,dst,kernel="thin_plate_spline",smoothing=2.0)
    field=rbf(GRID).reshape(GX,GX,2).astype(np.float32)
    fx=cv2.resize(field[...,0],(SZ,SZ),interpolation=cv2.INTER_CUBIC)
    fy=cv2.resize(field[...,1],(SZ,SZ),interpolation=cv2.INTER_CUBIC)
    mapx=(XX-fx).astype(np.float32); mapy=(YY-fy).astype(np.float32)
    return cv2.remap(base,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)

def mouth_interior(img, openness, vis, mx=MX, my=MY):
    # soft-edged dark interior + SUBTLE teeth, only when open; kept small + inside the warped lips
    if openness<0.12 or vis in ("REST","MBP"): return img
    h=int(SZ*0.012*(0.55+vis_hf(vis))*openness*1.95); h=min(h,int(SZ*0.052))   # bigger, still tidy
    w=int(SZ*0.030*(0.75+vis_wf(vis)*0.45)); w=min(w,int(SZ*0.056))
    if h<3: return img
    cx,cy=int(mx*SZ),int(my*SZ)+int(h*0.20)
    layer=np.zeros((SZ,SZ,4),np.uint8); d=layer
    cv2.ellipse(d,(cx,cy),(w,h),0,0,360,(20,12,18,255),-1)                      # very dark interior (BGRA)
    if h>=11: cv2.ellipse(d,(cx,cy-int(h*0.64)),(int(w*0.55),max(2,int(h*0.18))),0,0,360,(196,202,196,235),-1)  # thin dim teeth sliver
    if h>=16: cv2.ellipse(d,(cx,cy+int(h*0.42)),(int(w*0.38),int(h*0.22)),0,0,360,(58,44,120,235),-1)           # small tongue
    a=layer[...,3:].astype(np.float32)/255.0
    a=cv2.GaussianBlur(a,(0,0),2.6)[...,None]                                    # SOFT edges, no hard border
    rgb=layer[...,:3].astype(np.float32)
    out=img.astype(np.float32)*(1-a)+rgb*a
    return out.astype(np.uint8)
def vis_hf(v): return {"A":1.0,"O":0.85,"U":0.55,"E":0.45}.get(v,0.6)
def vis_wf(v): return {"A":0.6,"O":0.4,"U":0.3,"E":1.1}.get(v,0.6)

if __name__=="__main__" and "test" in sys.argv:
    base=cv2.imread(os.path.join(ND,"host_4411.png")); base=cv2.resize(base,(SZ,SZ))
    tiles=[]
    cases=[("REST","neutral",0.0),("A","happy",1.0),("E","happy",1.0),("O","hopeful",1.0),("U","neutral",0.9),("MBP","neutral",0.0),("A","surprise",1.0),("REST","sad",0.0)]
    for vis,emo,o in cases:
        disp={}
        for k,v in EMO[emo].items():
            disp[k]=(disp.get(k,(0,0))[0]+v[0], disp.get(k,(0,0))[1]+v[1])
        for k,v in VIS[vis].items():
            sc=o if k in ("lip_bot","jaw","lip_top","mouth_L","mouth_R") else 1.0
            disp[k]=(disp.get(k,(0,0))[0]+v[0]*sc, disp.get(k,(0,0))[1]+v[1]*sc)
        w=warp(base,disp); w=mouth_interior(w,o,vis)
        crop=w[int(0.18*SZ):int(0.62*SZ), int(0.28*SZ):int(0.72*SZ)]
        crop=cv2.resize(crop,(300,300)); cv2.putText(crop,f"{vis}/{emo}",(6,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
        tiles.append(crop)
    rows=[np.hstack(tiles[i:i+4]) for i in range(0,8,4)]; mont=np.vstack(rows)
    cv2.imwrite(os.path.join(ND,"_warp_test.png"),mont); print("saved _warp_test.png", mont.shape)
