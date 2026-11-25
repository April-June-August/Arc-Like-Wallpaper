# Arc-Like-Wallpaper

Generate soft gradients with fine grain as sRGB PNG wallpapers. Each run chooses a fresh layout and palette. The default size is 4800 × 3428 pixels.

## Install

Use Python 3.10 or later. From the project directory, run

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install 'numpy>=1.24' 'Pillow>=9.1'
```

## Generate

Create ten mist wallpapers in the output directory.

```sh
python generate.py --count 10
```

Choose your own colors and dimensions.

```sh
python generate.py --style mist --colors 073dd4 f89bb8 939ddf --width 3840 --height 2160 --output wallpapers
```

Choose `mist` for three softly blended colors, `free` for scattered color fields, `bold` for large asymmetric shapes, or `space` for a two-color diagonal gradient. Set `--grain` between 0 and 1 to adjust the texture. Existing PNGs are preserved when you generate another batch in the same folder.
