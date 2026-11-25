#!/usr/bin/env python3
"""Procedural Winter-Arc-inspired wallpapers. Generate random color fields and grain."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageCms

# Hand-authored palettes, unconstrained by Arc's picker. Colors have no fixed
# spatial roles; the layout generator permutes them independently.
PALETTES = {
    'glacier': ['5140dc','5524ac','5269d5','272668','a6dfe0','b0d1de','b3e2dd'],
    'ember': ['fa3029','fb3c13','b97d3e','395c62','bce9ed','ebc6f0','b5e6e3'],
    'meadow': ['75cbb5','21ab78','5cbe8d','70a96b','bce7df','d5db88','d4bdff'],
    'mauve': ['b57c94','883b9f','705068','69603a','b8a0a5','b0b3a4','b5dddd'],
    'lagoon': ['075873','169b9c','3c63af','243457','c2e6cf','e5c2a4','a7d9e5'],
    'saffron': ['e99554','e86758','ae5d91','644668','efcdb5','e9e0ac','bcd3e8'],
}
M1 = np.array([[.4122214708,.5363325363,.0514459929], [.2119034982,.6806995451,.1073969566], [.0883024619,.2817188376,.6299787005]],dtype=np.float32)
M2 = np.array([[.2104542553,.793617785,-.0040720468], [1.9779984951,-2.428592205,.4505937099], [.0259040371,.7827717662,-.808675766]],dtype=np.float32)
I1 = np.array([[1,.3963377774,.2158037573], [1,-.1055613458,-.0638541728], [1,-.0894841775,-1.291485548]],dtype=np.float32)
I2 = np.array([[4.0767416621,-3.3077115913,.2309699292], [-1.2684380046,2.6097574011,-.3413193965], [-.0041960863,-.7034186147,1.707614701]],dtype=np.float32)

def colors(text):
    result=[]
    for value in text:
        value=value.lstrip('#')
        if len(value)!=6:
            raise ValueError('Colors must be six-digit RGB hex values')
        result.append([int(value[i:i+2],16)/255 for i in (0,2,4)])
    return np.array(result,dtype=np.float32)

def to_lab(rgb):
    linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    return np.cbrt(linear @ M1.T) @ M2.T

def from_lab(lab):
    linear=(lab @ I1.T)**3 @ I2.T
    return np.where(np.abs(linear)<=.0031308,12.92*linear,
                    np.sign(linear)*(1.055*np.abs(linear)**(1/2.4)-.055))

def layout(rng, variation, color_count=7, composition="free"):
    if composition == "bold":
        count=int(rng.integers(3,6))
        # Broad, asymmetrical zones; placed around the canvas.
        centers=[rng.uniform(.1,.9,2)]
        for _ in range(count-1):
            candidates=rng.uniform(-.18,1.18,(48,2))
            distances=((candidates[:,None,:]-np.array(centers)[None,:,:])**2).sum(axis=-1)
            centers.append(candidates[np.argmax(distances.min(axis=1))])
        indices=np.resize(rng.permutation(color_count),count)
        patches=[]
        for center,index in zip(centers,indices):
            radius=rng.uniform(.33,.58);stretch=np.exp(rng.uniform(-.65,.65))
            patches.append([*center,radius*stretch,radius/stretch,rng.uniform(0,np.pi),rng.uniform(2.2,3.8),rng.uniform(-.65,.65),int(index)])
        waves=[(rng.uniform(2.5,5.5),rng.uniform(-np.pi,np.pi),rng.uniform(0,2*np.pi),rng.uniform(.05,.12)+variation*.2) for _ in range(2)]
        return patches,waves,float(rng.uniform(4.0,6.5))

    count=int(rng.integers(6,11))
    # Farthest-candidate placement keeps a few large fields spread over the
    # canvas, without a fixed grid, left/right roles, or a compulsory seam.
    centers=[rng.uniform(-.15,1.15,2)]
    for _ in range(count-1):
        candidates=rng.uniform(-.2,1.2,(40,2))
        distance=((candidates[:,None,:]-np.array(centers)[None,:,:])**2).sum(axis=-1)
        centers.append(candidates[np.argmax(distance.min(axis=1))])
    indices=np.resize(rng.permutation(color_count),count)
    rng.shuffle(indices)
    patches=[]
    for center,index in zip(centers,indices):
        radius=rng.uniform(.20,.43)
        stretch=np.exp(rng.uniform(-.8,.8))
        patches.append([*center,radius*stretch,radius/stretch,
                        rng.uniform(0,np.pi),rng.uniform(1.7,2.8),
                        rng.uniform(-.45,.45),int(index)])
    waves=[(rng.uniform(3,8),rng.uniform(-np.pi,np.pi),rng.uniform(0,2*np.pi),
            rng.uniform(.025,.075)+variation*.25) for _ in range(3)]
    return patches,waves,float(rng.uniform(2.0,3.7))


def render(width,height,palette,rng,grain,variation,mode,composition="free"):
    lab=to_lab(colors(palette))
    # Low-resolution continuous field, then full-resolution grain: blur never
    # destroys the grain. Working size preserves the output aspect ratio.
    w=min(width,960);h=max(2,round(w*height/width))
    y,x=np.mgrid[0:h,0:w].astype(np.float32);x/=(w-1);y/=(h-1)
    if mode=='space':
        ratio=width/height
        start=(.5-ratio*.5,0) if ratio<=1 else (0,.5-.5/ratio)
        end=(.5+ratio*.5,1) if ratio<=1 else (1,.5+.5/ratio)
        dx,dy=end[0]-start[0],end[1]-start[1]
        def at(a,b):return np.clip(((a-start[0])*dx+(b-start[1])*dy)/(dx*dx+dy*dy),0,1)
        t=at(0,0)*(1-x)*(1-y)+at(1,0)*x*(1-y)+at(0,1)*(1-x)*y+at(1,1)*x*y
        t=np.clip(t*4/3,0,1)[...,None]
        field=from_lab(lab[0]*(1-t)+lab[1]*t)
    else:
        patches,waves,sharpness=layout(rng,variation,len(palette),composition)
        xx=x.copy(); yy=y.copy()
        for frequency,angle,phase,amplitude in waves:
            wave=amplitude*np.sin(frequency*(x*np.cos(angle)+y*np.sin(angle))+phase)
            xx+=wave*np.cos(angle+1.1)
            yy+=wave*np.sin(angle+1.1)
        weights=[]
        for cx,cy,sx,sy,angle,power,bias,index in patches:
            dx=xx-cx;dy=yy-cy
            u=(dx*np.cos(angle)+dy*np.sin(angle))/sx
            v=(-dx*np.sin(angle)+dy*np.cos(angle))/sy
            weights.append(-.5*(np.abs(u)**power+np.abs(v)**power)*sharpness+bias)
        weights=np.stack(weights,axis=-1)
        weights=np.exp(weights-weights.max(axis=-1,keepdims=True))
        weights/=weights.sum(axis=-1,keepdims=True)
        field=from_lab(weights @ lab[[int(p[7]) for p in patches]])
    # Resize floating point channels rather than an already quantized RGB image.
    rgb=np.stack([np.asarray(Image.fromarray(field[:,:,i]).resize((width,height),Image.Resampling.BICUBIC)) for i in range(3)],axis=-1)
    # Texture composition: O*(G + 1-G.a) + (1-O.a)*G.
    # For opaque G and a premultiplied black grain texture O.rgb=0,
    # this reduces to G*(1-O.a). Texture generation is our own adaptation.
    # Full-resolution random field avoids small repeated tile patterns.
    noise=rng.random((height,width),dtype=np.float32)
    alpha=grain*noise**2
    rgb*=(1-alpha[...,None]) / (1-grain/3)  # wallpaper mean-lightness compensation
    # Unbiased sub-LSB dither for 8-bit PNG export, independent of grain.
    rgb+=rng.uniform(-.5/255,.5/255,(height,width,1)).astype(np.float32)
    return Image.fromarray(np.uint8(np.clip(rgb,0,1)*255+.5))

MIST_PALETTES = [['073dd4', 'f89bb8', '939ddf'], ['209a79', 'c8a9df', 'b8ddd1'], ['1949c8', 'efb28e', 'a4c0df'], ['6238b5', 'a7dfdc', 'c4afd2'], ['ed624c', 'afe0e0', 'bca5c9'], ['1789a3', 'eeb2c4', 'a3bfdf'], ['5650c6', 'd9dca0', 'ada7d5'], ['3bb294', 'edbb8b', 'd0e3c1'], ['2064d6', 'cf8cc7', 'a7cbe0'], ['bf6e9d', '99c4b1', 'dcd0c6']]

def composition(rng):
 # Balanced interior/edge coverage, but no fixed zones or dark dividing seam.
 centers=[rng.uniform(.15,.85,2)]
 for _ in range(6):
  candidates=rng.uniform(-.12,1.12,(36,2))
  distances=((candidates[:,None]-np.array(centers)[None])**2).sum(axis=-1)
  centers.append(candidates[np.argmax(distances.min(axis=1))])
 roles=np.array([0,0,0,1,1,2,2]);rng.shuffle(roles)
 patches=[]
 for center,role in zip(centers,roles):
  radius=rng.uniform(.23,.38);stretch=np.exp(rng.uniform(-.35,.35))
  patches.append([*center,radius*stretch,radius/stretch,rng.uniform(0,np.pi),int(role)])
 waves=[(rng.uniform(3,6),rng.uniform(0,2*np.pi),rng.uniform(0,2*np.pi),rng.uniform(.025,.07)) for _ in range(3)]
 return dict(patches=patches,waves=waves,softness=float(rng.uniform(.018,.033)))

def render_mist(width,height,palette,rng,grain):
 params=composition(rng);w=min(width,1000);h=max(2,round(w*height/width))
 y,x=np.mgrid[0:h,0:w].astype(np.float32);x/=w-1;y/=h-1
 xx=x.copy();yy=y.copy()
 for freq,angle,phase,amplitude in params['waves']:
  wave=amplitude*np.sin(freq*(x*np.cos(angle)+y*np.sin(angle))+phase)
  xx+=wave*np.cos(angle+1);yy+=wave*np.sin(angle+1)
 weights=[]
 for cx,cy,sx,sy,angle,role in params['patches']:
  dx=xx-cx;dy=yy-cy
  u=(dx*np.cos(angle)+dy*np.sin(angle))/sx
  v=(-dx*np.sin(angle)+dy*np.cos(angle))/sy
  # Gaussian fields, unlike previous power-distance competition. Moderately
  # crisp interiors coexist with long overlapping tails at the boundaries.
  weights.append(np.exp(-1.65*(u*u+v*v)))
 weights=np.stack(weights,axis=-1)+.0001
 weights/=weights.sum(axis=-1,keepdims=True)
 lab=to_lab(colors(palette))
 field=weights @ lab[[p[5] for p in params['patches']]]
 # Blur the color weights, not quantized RGB. Pillow GaussianBlur doesn't
 # accept F images, so use a floating-point
 # separable finite Gaussian convolution with reflected edge extension.
 sigma=w*params['softness'];radius=int(np.ceil(3*sigma))
 kernel=np.exp(-.5*(np.arange(-radius,radius+1)/sigma)**2);kernel/=kernel.sum()
 for axis in (0,1):
  pad=[(0,0)]*3;pad[axis]=(radius,radius)
  padded=np.pad(field,pad,mode='reflect')
  field=np.apply_along_axis(lambda row:np.convolve(row,kernel,mode='valid'),axis,padded).astype(np.float32)
 rgb=from_lab(field)
 rgb=np.stack([np.asarray(Image.fromarray(rgb[:,:,i]).resize((width,height),Image.Resampling.BICUBIC)) for i in range(3)],axis=-1)
 alpha=grain*rng.random((height,width),dtype=np.float32)**2
 rgb*=(1-alpha[...,None])/(1-grain/3)
 rgb+=rng.uniform(-.5/255,.5/255,(height,width,1)).astype(np.float32)
 return Image.fromarray(np.uint8(np.clip(rgb,0,1)*255+.5))


def main():
    parser = argparse.ArgumentParser(description="Generate textured gradient wallpapers.")
    parser.add_argument("--style", choices=("mist", "free", "bold", "space"), default="mist")
    parser.add_argument("--colors", nargs="+", metavar="RRGGBB")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--width", type=int, default=4800)
    parser.add_argument("--height", type=int, default=3428)
    parser.add_argument("--grain", type=float, default=.27)
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()
    if not (2 <= args.width <= 16384 and 2 <= args.height <= 16384 and args.width * args.height <= 70_000_000):
        parser.error("Use dimensions from 2 to 16384 and at most 70 million pixels.")
    if args.count < 1 or not 0 <= args.grain <= 1:
        parser.error("Count must be positive and grain must be between zero and one.")
    if args.colors:
        if not 2 <= len(args.colors) <= 12:
            parser.error("Provide 2 to 12 colors.")
        if args.style == "mist" and len(args.colors) != 3:
            parser.error("Mist uses three colors.")
        if args.style == "space" and len(args.colors) != 2:
            parser.error("Space uses two colors.")
        try:
            colors(args.colors)
        except ValueError as error:
            parser.error(str(error))

    rng = np.random.default_rng()
    palettes = MIST_PALETTES if args.style == "mist" else list(PALETTES.values())
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    args.output.mkdir(parents=True, exist_ok=True)
    number = 0
    for _ in range(args.count):
        palette = args.colors or palettes[int(rng.integers(len(palettes)))]
        if args.style == "mist":
            image = render_mist(args.width, args.height, palette, rng, args.grain)
        else:
            image = render(args.width, args.height, palette, rng, args.grain, .065, args.style, args.style)
        while True:
            number += 1
            path = args.output / f"{args.style}-{number:03}.png"
            try:
                stream = path.open("xb")
            except FileExistsError:
                continue
            break
        try:
            with stream:
                image.save(stream, format="PNG", icc_profile=profile)
        except BaseException:
            path.unlink()
            raise
        print(path, flush=True)


if __name__ == "__main__":
    main()
