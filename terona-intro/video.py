"""Render the TERONA intro video frames and mux them with music.wav.

Usage: python3 video.py music.wav out.mp4 [--preview T1,T2,...]
Everything is timed to the score: 90 BPM, one bar = 2.6667 s.
"""
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageChops

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, 'assets')
W, H, FPS = 1920, 1080, 30
BPM = 90
BEAT = 60 / BPM
BAR = 4 * BEAT
TOTAL = 24 * BAR + 4.5


def font(name, size, var=None):
    f = ImageFont.truetype(os.path.join(A, 'fonts', name), size)
    if var:
        f.set_variation_by_name(var)
    return f


F_ROLE = font('Cinzel.ttf', 38, 'Bold')
F_ROLE_S = font('Cinzel.ttf', 30, 'Regular')
F_NAME = font('Oswald.ttf', 132, 'Bold')
F_NAME_M = font('Oswald.ttf', 96, 'Bold')
F_SUB = font('Oswald.ttf', 46, 'Medium')
F_SMALL = font('Oswald.ttf', 34, 'Regular')
F_PIX = font('PressStart2P.ttf', 200)
F_PIX_S = font('PressStart2P.ttf', 28)


def ease_out(x):
    x = min(max(x, 0), 1)
    return 1 - (1 - x) ** 3


def ease_in(x):
    x = min(max(x, 0), 1)
    return x ** 3


def clamp01(x):
    return min(max(x, 0.0), 1.0)


# ------------------------------------------------------------ pixel art scenes
def blocky(arr, scale):
    im = Image.fromarray(arr.astype(np.uint8), 'RGB')
    return im.resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.NEAREST)


def sky(h, w, top, bot, bands=7):
    a = np.zeros((h, w, 3))
    for y in range(h):
        k = math.floor(y / h * bands) / (bands - 1)
        a[y] = np.array(top) * (1 - k) + np.array(bot) * k
    return a


def terrain(a, rng, base, rough, grass=(86, 168, 60), dirt=(134, 96, 67), stone=(110, 110, 118), trees=True,
            snow=False):
    h, w, _ = a.shape
    y = base
    heights = []
    for x in range(w):
        if rng.random() < rough:
            y += rng.choice([-1, 1])
        y = int(np.clip(y, h * 0.35, h - 4))
        heights.append(y)
    for x, y in enumerate(heights):
        for yy in range(y, h):
            d = yy - y
            if d == 0:
                c = (235, 240, 245) if snow else grass
            elif d < 4:
                c = dirt
            else:
                c = stone
            jitter = rng.integers(-10, 10)
            a[yy, x] = np.clip(np.array(c) + jitter, 0, 255)
    if trees:
        x = rng.integers(2, 8)
        while x < w - 3:
            y = heights[x]
            th = rng.integers(3, 6)
            a[max(0, y - th):y, x] = (102, 76, 46)
            for dy in range(-th - 3, -th + 1):
                for dx in range(-2, 3):
                    if abs(dx) == 2 and dy in (-th - 3, -th):
                        continue
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w:
                        a[yy, xx] = np.clip(np.array((48, 110, 40)) + rng.integers(-15, 15), 0, 255)
            x += rng.integers(7, 15)
    return heights


def scene_sunset(seed=1, w=120, h=68):
    rng = np.random.default_rng(seed)
    a = sky(h, w, (60, 30, 90), (250, 140, 60), 8)
    cx, cy = int(w * 0.68), int(h * 0.45)
    a[cy - 5:cy + 5, cx - 5:cx + 5] = (255, 220, 120)
    a[cy - 4:cy + 4, cx - 4:cx + 4] = (255, 240, 170)
    terrain(a, rng, int(h * 0.6), 0.35)
    return a


def scene_castle(seed=2, w=120, h=68):
    rng = np.random.default_rng(seed)
    a = sky(h, w, (10, 20, 50), (40, 90, 130), 6)
    for _ in range(60):
        a[rng.integers(0, h // 2), rng.integers(0, w)] = (220, 230, 255)
    a[8:15, 14:21] = (235, 235, 210)
    a[9:14, 15:20] = (250, 250, 230)
    heights = terrain(a, rng, int(h * 0.72), 0.15, trees=False)
    g = min(heights[60:100])
    col = np.array((70, 72, 84))
    a[g - 30:g, 66:94] = col                     # keep
    for tx in (60, 94):
        a[g - 40:g, tx:tx + 8] = col + 10
        for k in range(0, 8, 2):
            a[g - 42:g - 40, tx + k] = col + 10
    for k in range(66, 94, 3):
        a[g - 32:g - 30, k:k + 2] = col
    a[g - 10:g, 77:83] = (40, 28, 20)
    for wx, wy in ((70, g - 22), (88, g - 22), (62, g - 32), (96, g - 32), (79, g - 22)):
        a[wy:wy + 3, wx:wx + 2] = (255, 200, 80)
    a += rng.integers(-6, 6, a.shape)
    return np.clip(a, 0, 255)


def scene_cave(seed=3, w=120, h=68):
    rng = np.random.default_rng(seed)
    a = np.zeros((h, w, 3)) + (58, 58, 64)
    a += rng.integers(-14, 14, (h, w, 1))
    # tunnel
    yy, xx = np.mgrid[0:h, 0:w]
    tunnel = (np.abs(yy - h * 0.55 - 6 * np.sin(xx / 13)) < 13 + 3 * np.sin(xx / 7))
    a[tunnel] = a[tunnel] * 0.35
    for _ in range(26):
        x, y = rng.integers(0, w - 2), rng.integers(0, h - 2)
        if tunnel[y, x]:
            continue
        c = [(90, 230, 230), (250, 210, 60), (200, 60, 50), (40, 40, 40)][rng.integers(0, 4)]
        a[y:y + 2, x:x + 2] = c
        if rng.random() < 0.5:
            a[y + 1, x + 2 if x + 2 < w else x] = c
    for tx in (22, 64, 101):
        ty = int(h * 0.55 + 6 * np.sin(tx / 13)) - 2
        glow = np.exp(-(((xx - tx) ** 2 + (yy - ty) ** 2) / 120))[..., None]
        a = a + glow * np.array((180, 110, 40))
        a[ty - 3:ty, tx] = (110, 80, 50)
        a[ty - 4, tx] = (255, 220, 90)
    a[h - 5:, :] = (230, 90, 20)
    a[h - 5:, :] += rng.integers(-20, 25, (5, w, 1))
    return np.clip(a, 0, 255)


def scene_night(seed=4, w=120, h=68):
    rng = np.random.default_rng(seed)
    a = sky(h, w, (5, 8, 25), (30, 40, 90), 6)
    for _ in range(90):
        a[rng.integers(0, int(h * 0.6)), rng.integers(0, w)] = (230, 230, 255)
    # aurora bands
    xx = np.arange(w)
    for k, c in enumerate([(60, 220, 160), (90, 140, 230)]):
        ys = (12 + k * 5 + 4 * np.sin(xx / 9 + k)).astype(int)
        for x, y in zip(xx, ys):
            a[y:y + 3, x] = a[y:y + 3, x] * 0.4 + np.array(c) * 0.6
    heights = terrain(a, rng, int(h * 0.7), 0.3, snow=True)
    hx = 30
    hy = heights[hx + 5]
    a[hy - 10:hy, hx:hx + 14] = (120, 84, 50)
    for i in range(8):
        a[hy - 11 - i, hx - 1 + i:hx + 15 - i] = (90, 50, 40)
    a[hy - 7:hy - 4, hx + 3:hx + 6] = (255, 210, 90)
    a[hy - 5:hy, hx + 9:hx + 12] = (60, 40, 25)
    return np.clip(a, 0, 255)


def scene_ocean(seed=5, w=120, h=68):
    rng = np.random.default_rng(seed)
    a = sky(h, w, (120, 190, 240), (250, 200, 150), 6)
    cy = int(h * 0.55)
    a[cy - 12:cy - 4, 20:28] = (255, 230, 150)
    for y in range(cy, h):
        k = (y - cy) / (h - cy)
        a[y] = np.array((40, 110, 190)) * (1 - k) + np.array((15, 45, 110)) * k
        for x in range(w):
            if rng.random() < 0.06:
                a[y, x] = (150, 210, 250)
    # island
    a[cy - 6:cy, 70:104] = (230, 210, 150)
    a[cy - 9:cy - 6, 76:98] = (86, 168, 60)
    a[cy - 20:cy - 9, 86:88] = (110, 80, 50)
    a[cy - 24:cy - 20, 80:94] = (48, 120, 40)
    return np.clip(a, 0, 255)


SWORD = """
..............KK
.............KCK
............KCLK
...........KCLK.
..........KCLK..
.........KCLK...
..KK....KCLK....
..KBK..KCLK.....
...KBKKCLK......
...KBBCLK.......
....KBBK........
...KHKBBK.......
..KHK.KKBK......
.KHK....KK......
KGK.............
KK..............
"""
PICK = """
....KKKKKK......
..KKCCCCCCKK....
.KCCLLLLLLCCK...
.KCKKKKHKKKCK...
.KK...KHK..KK...
.....KHK........
....KHK.........
...KHK..........
..KHK...........
.KHK............
KHK.............
KK..............
"""
SPR_COL = {'K': (20, 20, 24), 'C': (60, 200, 210), 'L': (180, 250, 245), 'B': (90, 60, 40), 'H': (140, 100, 60),
           'G': (60, 200, 210)}


def sprite_scene(art, bg1, bg2, w=64, h=36):
    rows = [r for r in art.strip('\n').split('\n')]
    a = np.zeros((h, w, 3))
    yy, xx = np.mgrid[0:h, 0:w]
    stripe = ((xx + yy) // 4) % 2
    a[stripe == 0] = bg1
    a[stripe == 1] = bg2
    sh, sw = len(rows), len(rows[0])
    s = 2
    oy, ox = (h - sh * s) // 2, (w - sw * s) // 2
    for y, r in enumerate(rows):
        for x, ch in enumerate(r):
            if ch in SPR_COL:
                a[oy + y * s:oy + y * s + s, ox + x * s:ox + x * s + s] = SPR_COL[ch]
    return a


# ------------------------------------------------------------ assets
def load_rgba(p):
    return Image.open(os.path.join(A, p)).convert('RGBA')


KORI_ART = load_rgba('kori-art.webp')
KORI_ICON = load_rgba('kori-icon.webp')
DZ_BODY = load_rgba('darkzmaj-body.png')
DZ_HEAD = load_rgba('darkzmaj-head.png')


def on_bg(img, size, bg, fit=0.9, align=0.5):
    """Place an RGBA image on a solid/gradient panel background."""
    w, h = size
    base = Image.new('RGB', size, bg[0])
    g = Image.linear_gradient('L').resize(size)
    base = Image.composite(Image.new('RGB', size, bg[1]), base, g)
    sc = min(w * fit / img.width, h * fit / img.height)
    im = img.resize((max(1, int(img.width * sc)), max(1, int(img.height * sc))), Image.LANCZOS)
    base.paste(im, ((w - im.width) // 2, int((h - im.height) * align)), im)
    return base


def halftone(size, c1, c2, dot=14):
    w, h = size
    im = Image.new('RGB', size, c1)
    d = ImageDraw.Draw(im)
    for y in range(0, h + dot, dot):
        for x in range(0, w + dot, dot):
            r = dot * 0.32 * (0.4 + 0.9 * (x / w))
            d.ellipse((x - r + (y // dot % 2) * dot / 2, y - r, x + r + (y // dot % 2) * dot / 2, y + r), fill=c2)
    return im


PANEL_SRC = {
    'sunset': blocky(scene_sunset(), 16),
    'castle': blocky(scene_castle(), 16),
    'cave': blocky(scene_cave(), 16),
    'night': blocky(scene_night(), 16),
    'ocean': blocky(scene_ocean(), 16),
    'sword': blocky(sprite_scene(SWORD, (240, 120, 40), (220, 95, 30)), 20),
    'pick': blocky(sprite_scene(PICK, (40, 90, 160), (30, 70, 135)), 20),
    'sunset2': blocky(scene_sunset(seed=9), 16),
}


def art_panel(kind, size):
    """A panel source larger than `size` so it can pan."""
    w, h = size
    bw, bh = int(w * 1.25), int(h * 1.25)
    if kind in PANEL_SRC:
        src = PANEL_SRC[kind]
        s = max(bw / src.width, bh / src.height)
        im = src.resize((math.ceil(src.width * s), math.ceil(src.height * s)), Image.NEAREST)
        return im
    if kind == 'dz_body':
        bg = halftone((bw, bh), (24, 30, 90), (40, 60, 170))
        im = DZ_BODY.copy()
        im.thumbnail((int(bw * 0.9), int(bh * 1.6)), Image.NEAREST)
        bg.paste(im, ((bw - im.width) // 2, int(bh * 0.08)), im)
        return bg
    if kind == 'dz_head':
        return on_bg(DZ_HEAD, (bw, bh), ((250, 200, 40), (240, 120, 30)), 0.75)
    if kind == 'kori':
        return on_bg(KORI_ART, (bw, bh), ((30, 40, 90), (150, 170, 240)), 1.0, 0.0)
    if kind == 'kori_icon':
        return on_bg(rounded(KORI_ICON, 26), (bw, bh), ((12, 18, 40), (50, 80, 160)), 0.6)
    raise KeyError(kind)


# ------------------------------------------------------------ title logo
def title_logo(text='TERONA', f=F_PIX):
    """Blocky title: grass-topped dirt/stone texture inside pixel letters."""
    d0 = ImageDraw.Draw(Image.new('L', (10, 10)))
    bb = d0.textbbox((0, 0), text, font=f)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    pad = 40
    w, h = tw + pad * 2, th + pad * 2
    mask = Image.new('L', (w, h), 0)
    ImageDraw.Draw(mask).text((pad - bb[0], pad - bb[1]), text, font=f, fill=255)
    # texture at block resolution
    blk = f.size // 8
    rng = np.random.default_rng(7)
    tx = np.zeros((h // blk + 1, w // blk + 1, 3))
    rows = tx.shape[0]
    top = (pad - bb[1] + bb[1]) // blk
    for y in range(rows):
        rel = y - pad // blk
        if rel < 2:
            c = (100, 190, 70)
        elif rel < 5:
            c = (150, 105, 70)
        else:
            c = (125, 125, 135)
        tx[y] = c
    tx += rng.integers(-18, 18, tx.shape)
    tex = blocky(np.clip(tx, 0, 255), blk).crop((0, 0, w, h))
    letters = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    letters.paste(tex, (0, 0), mask)
    # outline + shadow
    outline = mask.filter(ImageFilter.MaxFilter(13))
    shadow = Image.new('RGBA', (w + 30, h + 30), (0, 0, 0, 0))
    sh = Image.new('L', (w + 30, h + 30), 0)
    sh.paste(outline, (18, 18))
    shadow.paste((0, 0, 0, 200), (0, 0), sh)
    out = Image.new('RGBA', (w + 30, h + 30), (0, 0, 0, 0))
    out.alpha_composite(shadow)
    ol = Image.new('RGBA', (w, h), (12, 12, 16, 255))
    out.paste(ol, (0, 0), outline)
    out.alpha_composite(letters)
    # per-letter boxes for the drop-in animation
    boxes = []
    x = pad - bb[0]
    for ch in text:
        cw = d0.textlength(ch, font=f)
        boxes.append((int(x - 8), int(x + cw + 8)))
        x += cw
    return out, boxes


LOGO, LOGO_BOXES = title_logo()


# ------------------------------------------------------------ helpers
def text_img(txt, f, fill=(255, 255, 255), spacing=0, shadow=True):
    d0 = ImageDraw.Draw(Image.new('L', (10, 10)))
    if spacing:
        widths = [d0.textlength(c, font=f) for c in txt]
        tw = int(sum(widths) + spacing * (len(txt) - 1))
    else:
        tw = int(d0.textlength(txt, font=f))
    asc, desc = f.getmetrics()
    th = asc + desc
    im = Image.new('RGBA', (tw + 20, th + 20), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if spacing:
        x = 10
        for c, cw in zip(txt, widths):
            if shadow:
                d.text((x + 4, 14), c, font=f, fill=(0, 0, 0, 180))
            d.text((x, 10), c, font=f, fill=fill)
            x += cw + spacing
    else:
        if shadow:
            d.text((14, 14), txt, font=f, fill=(0, 0, 0, 180))
        d.text((10, 10), txt, font=f, fill=fill)
    return im


def with_alpha(im, a):
    if a >= 0.999:
        return im
    im = im.copy()
    al = im.getchannel('A').point(lambda v: int(v * a))
    im.putalpha(al)
    return im


def paste_center(frame, im, cx, cy, alpha=1.0):
    if alpha <= 0.003:
        return
    frame.alpha_composite(with_alpha(im, alpha), (int(cx - im.width / 2), int(cy - im.height / 2)))


# ------------------------------------------------------------ logo flash cards (bars 0-2)
def streak_card(frame, img, t, dur, tint):
    """Rockstar-style-but-ours: horizontal light streak resolves into the logo."""
    tin, tout = 0.28, 0.35
    if t < 0 or t > dur:
        return
    cx, cy = W / 2, H / 2 - 30
    if t < tin:
        k = t / tin
        sx = 1 + 14 * (1 - ease_out(k))
        sy = 0.06 + 0.94 * ease_out(k)
        a = clamp01(k * 1.5)
    elif t > dur - tout:
        k = (t - (dur - tout)) / tout
        sx = 1 + 10 * ease_in(k)
        sy = 1 - 0.9 * ease_in(k)
        a = 1 - ease_in(k)
    else:
        k = (t - tin) / (dur - tin - tout)
        sx = sy = 1 + 0.04 * k
        a = 1
    iw, ih = max(2, int(img.width * sx)), max(2, int(img.height * sy))
    im = img.resize((iw, ih), Image.BILINEAR)
    if sx > 1.5:
        im = im.filter(ImageFilter.BoxBlur(min(30, int(sx * 2))))
    # streak bar
    bar_a = clamp01(1 - abs(t - 0.05) / 0.4) * 0.8
    if bar_a > 0:
        bar = Image.new('RGBA', (W, 26), tint + (int(255 * bar_a),)).filter(ImageFilter.GaussianBlur(8))
        frame.alpha_composite(bar, (0, int(cy - 13)))
    paste_center(frame, im, cx, cy, a)


def rounded(img, r):
    m = Image.new('L', img.size, 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, img.width - 1, img.height - 1), r, fill=255)
    out = img.convert('RGBA')
    out.putalpha(ImageChops.multiply(out.getchannel('A'), m))
    return out


def card_kori():
    icon = rounded(KORI_ICON.resize((300, 300), Image.LANCZOS), 60)
    lbl = text_img('KORI NETWORK', F_SUB, (200, 215, 255), spacing=10, shadow=False)
    c = Image.new('RGBA', (max(icon.width, lbl.width), icon.height + lbl.height + 20), (0, 0, 0, 0))
    c.alpha_composite(icon, ((c.width - icon.width) // 2, 0))
    c.alpha_composite(lbl, ((c.width - lbl.width) // 2, icon.height + 20))
    return c


def card_dz():
    head = DZ_HEAD.resize((300, 300), Image.NEAREST)
    box = Image.new('RGBA', (340, 340), (25, 40, 160, 255))
    box.alpha_composite(head, (20, 20))
    box = rounded(box, 40)
    lbl = text_img('DARKZMAJ', F_SUB, (255, 230, 120), spacing=12, shadow=False)
    c = Image.new('RGBA', (max(box.width, lbl.width), box.height + lbl.height + 20), (0, 0, 0, 0))
    c.alpha_composite(box, ((c.width - box.width) // 2, 0))
    c.alpha_composite(lbl, ((c.width - lbl.width) // 2, box.height + 20))
    return c


def card_kori_art():
    art = KORI_ART.resize((340, 340), Image.LANCZOS)
    circ = Image.new('RGBA', (360, 360), (0, 0, 0, 0))
    m = Image.new('L', (340, 340), 0)
    ImageDraw.Draw(m).ellipse((0, 0, 339, 339), fill=255)
    bg = Image.new('RGBA', (340, 340), (90, 120, 230, 255))
    bg.alpha_composite(art)
    ring = Image.new('RGBA', (360, 360), (0, 0, 0, 0))
    ImageDraw.Draw(ring).ellipse((0, 0, 359, 359), fill=(220, 230, 255, 255))
    circ.alpha_composite(ring)
    circ.paste(bg, (10, 10), m)
    lbl = text_img('kori.my', F_SUB, (210, 220, 255), spacing=6, shadow=False)
    c = Image.new('RGBA', (circ.width, circ.height + lbl.height + 20), (0, 0, 0, 0))
    c.alpha_composite(circ, (0, 0))
    c.alpha_composite(lbl, ((c.width - lbl.width) // 2, circ.height + 20))
    return c


CARDS = [(card_kori(), (90, 140, 255)), (card_dz(), (255, 200, 60)), (card_kori_art(), (170, 150, 255))]


# ------------------------------------------------------------ credit sections
PX0, PX1, PY0, PY1, GAP = 860, 1850, 70, 1010, 18


def layout(kind, side):
    """Return 3 panel rects inside the art area. side='right' => art on right."""
    if side == 'right':
        x0, x1 = PX0, PX1
    else:
        x0, x1 = W - PX1, W - PX0
    xm = (x0 + x1) // 2
    ym = int(PY0 + (PY1 - PY0) * 0.55)
    if kind == 0:
        return [(x0, PY0, x1, ym), (x0, ym + GAP, xm - GAP // 2, PY1), (xm + GAP // 2, ym + GAP, x1, PY1)]
    if kind == 1:
        xa = x0 + int((x1 - x0) * 0.42)
        return [(x0, PY0, xa, PY1), (xa + GAP, PY0, x1, ym), (xa + GAP, ym + GAP, x1, PY1)]
    ya = int(PY0 + (PY1 - PY0) * 0.42)
    return [(x0, PY0, xm - GAP // 2, ya), (xm + GAP // 2, PY0, x1, ya), (x0, ya + GAP, x1, PY1)]


SECTIONS = [
    # bar, bars, roles/names, art kinds, layout, side
    dict(bar=10, n=2, role='Created by', names=['Obe107'], art=['sunset', 'sword', 'castle'], lay=0, side='right'),
    dict(bar=12, n=2, role='Created by', names=['samcci'], art=['ocean', 'night', 'pick'], lay=1, side='left'),
    dict(bar=14, n=2, role='Lead Developer', names=['DarkZmaj'], art=['dz_body', 'cave', 'dz_head'], lay=1,
         side='right', extra='darkzmaj.eu'),
    dict(bar=16, n=2, role='Plugins & Bots', names=['KORI Network'], art=['night', 'kori', 'kori_icon'],
         lay=2, side='left', extra='kori.my', name_font='M'),
    dict(bar=18, n=2, role='Founders of Terona', names=['Obe107', 'samcci'], art=['castle', 'sunset2', 'sword'],
         lay=0, side='right'),
    dict(bar=20, n=2, role='Starring', names=['Every player', 'of Terona'], art=['cave', 'ocean', 'sunset'], lay=2,
         side='left', name_font='M'),
]

_cache = {}


def section_assets(si):
    if si in _cache:
        return _cache[si]
    s = SECTIONS[si]
    rects = layout(s['lay'], s['side'])
    arts = [art_panel(k, (r[2] - r[0], r[3] - r[1])) for k, r in zip(s['art'], rects)]
    nf = F_NAME_M if s.get('name_font') == 'M' else F_NAME
    role = text_img(s['role'].upper(), F_ROLE, (215, 205, 190), spacing=4)
    names = [text_img(n, nf) for n in s['names']]
    extra = text_img(s['extra'], F_SMALL, (150, 170, 220)) if s.get('extra') else None
    _cache[si] = (rects, arts, role, names, extra)
    return _cache[si]


def draw_section(frame, si, t):
    s = SECTIONS[si]
    dur = s['n'] * BAR
    if t < 0 or t > dur:
        return
    rects, arts, role, names, extra = section_assets(si)
    out_k = clamp01((t - (dur - 0.45)) / 0.45)
    dirs = [(-1, 0), (0, 1), (1, 0)] if s['side'] == 'right' else [(1, 0), (0, -1), (-1, 0)]
    for i, (r, art) in enumerate(zip(rects, arts)):
        t0 = i * BEAT * 0.75
        k = ease_out((t - t0) / 0.35)
        if k <= 0:
            continue
        pw, ph = r[2] - r[0], r[3] - r[1]
        # pan inside panel
        p = (t - t0) / dur
        ox = int((art.width - pw) * (0.15 + 0.7 * p if i % 2 == 0 else 0.85 - 0.7 * p))
        oy = int((art.height - ph) * (0.5 + 0.3 * math.sin(p * 2 + i)))
        crop = art.crop((ox, oy, ox + pw, oy + ph))
        # wipe-in: reveal growing from the panel's entry side
        dx, dy = dirs[i]
        if k < 1:
            m = Image.new('L', (pw, ph), 0)
            dm = ImageDraw.Draw(m)
            if dx:
                ww = int(pw * k)
                dm.rectangle((0, 0, ww, ph) if dx > 0 else (pw - ww, 0, pw, ph), fill=255)
            else:
                hh = int(ph * k)
                dm.rectangle((0, 0, pw, hh) if dy > 0 else (0, ph - hh, pw, ph), fill=255)
        else:
            m = None
        slide = int(60 * (1 - k))
        ex = ease_in(out_k)
        x = r[0] - dx * slide + int(dx * ex * 700 * (1 if i != 1 else 0))
        y = r[1] - dy * slide + int((ex * 900 if i == 1 else 0) * (-dy if dy else 1))
        flash = clamp01(1 - (t - t0) / 0.18)
        cimg = crop.convert('RGBA')
        if flash > 0:
            cimg = Image.blend(cimg, Image.new('RGBA', cimg.size, (255, 255, 255, 255)), flash * 0.6)
        if m is not None:
            cimg.putalpha(m)
        if out_k > 0:
            cimg = with_alpha(cimg, 1 - out_k)
        # thin dark frame
        fr = Image.new('RGBA', (pw + 10, ph + 10), (0, 0, 0, int(255 * k * (1 - out_k))))
        frame.alpha_composite(fr, (x - 5, y - 5))
        frame.alpha_composite(cimg, (x, y))
    # text
    tx = 110 if s['side'] == 'right' else W - 110
    ty = H // 2 - 40 - (len(names) - 1) * 55
    tk = ease_out((t - BEAT * 0.5) / 0.5)
    ta = tk * (1 - out_k)
    if ta > 0:
        def place_txt(im, y, delay):
            kk = ease_out((t - BEAT * 0.5 - delay) / 0.5)
            a = kk * (1 - out_k)
            off = int(40 * (1 - kk))
            if s['side'] == 'right':
                frame.alpha_composite(with_alpha(im, a), (tx - off, y))
            else:
                frame.alpha_composite(with_alpha(im, a), (tx - im.width + off, y))
        place_txt(role, ty - 70, 0)
        y = ty
        for j, nm in enumerate(names):
            place_txt(nm, y, 0.12 * (j + 1))
            y += nm.height - 40
        if extra:
            place_txt(extra, y + 20, 0.4)
        # accent rule
        rw = int(320 * tk * (1 - out_k))
        if rw > 2:
            ImageDraw.Draw(frame).rectangle(
                (tx, ty - 80, tx + rw, ty - 76) if s['side'] == 'right' else (tx - rw, ty - 80, tx, ty - 76),
                fill=(240, 140, 50, int(255 * ta)))


# ------------------------------------------------------------ particles
rngp = np.random.default_rng(11)
PART = rngp.random((70, 5))  # x, phase, speed, size, hue


def draw_particles(frame, t, amount):
    if amount <= 0:
        return
    d = ImageDraw.Draw(frame)
    for x, ph, sp, sz, hue in PART:
        y = (1.1 - ((t * (0.03 + 0.06 * sp) + ph) % 1.2)) * H
        xx = (x * W + 30 * math.sin(t * 0.7 + ph * 9))
        s = 4 + int(sz * 9)
        c = (120, 255, 120) if hue < 0.5 else (255, 190, 80)
        a = int(amount * 140 * (0.4 + 0.6 * sz))
        d.rectangle((xx, y, xx + s, y + s), fill=c + (a,))


# ------------------------------------------------------------ text-only scenes
T_PRESENTS = [text_img('KORI Network', F_SUB, (225, 225, 235), spacing=6),
              text_img('PRESENTS', F_ROLE_S, (170, 170, 180), spacing=14)]
T_PROD = [text_img('A  DarkZmaj  Production', F_SUB, (225, 225, 235), spacing=4)]
T_FOR = [text_img('for  Obe107  &  samcci', F_SUB, (225, 225, 235), spacing=4)]
T_SUBTITLE = text_img('M I N E C R A F T   S E R V E R', F_SUB, (240, 200, 120))
T_END1 = text_img('KORI Network  ·  DarkZmaj  ·  Obe107  ·  samcci', F_SMALL, (170, 180, 210))
T_DZ_END = text_img('darkzmaj.eu   ·   kori.my', F_SMALL, (140, 150, 190))


def fade_text(frame, ims, t, dur, cy=H // 2, gap=8):
    a = clamp01(t / 0.5) * clamp01((dur - t) / 0.5)
    if a <= 0:
        return
    tot = sum(i.height for i in ims) + gap * (len(ims) - 1)
    y = cy - tot / 2
    drift = 6 * (t / dur)
    for im in ims:
        paste_center(frame, im, W / 2 + drift, y + im.height / 2, a)
        y += im.height + gap


# ------------------------------------------------------------ title
def draw_title_build(frame, t):
    """bars 6-7: letters drop in one at a time. t from bar 6."""
    lx = W / 2 - LOGO.width / 2
    ly = H / 2 - LOGO.height / 2 - 40
    times = [0, 1.5, 3, 4.5, 6, 6.75]
    for i, (a, b) in enumerate(LOGO_BOXES):
        tt = t - times[i] * BEAT
        if tt < 0:
            continue
        k = ease_out(tt / 0.3)
        bounce = math.sin(min(tt / 0.3, 1) * math.pi) * 0 + (1 - k) * -260
        piece = LOGO.crop((a, 0, min(b + 30, LOGO.width), LOGO.height))
        frame.alpha_composite(with_alpha(piece, clamp01(tt / 0.15)), (int(lx + a), int(ly + bounce)))


def draw_title_hold(frame, t, dur):
    """bars 8-9: full title, flash, shake, subtitle."""
    shake = 18 * math.exp(-t * 5) * math.sin(t * 70)
    scale = 1.0 + 0.03 * (t / dur)
    out = clamp01((t - (dur - 0.5)) / 0.5)
    im = LOGO.resize((int(LOGO.width * scale), int(LOGO.height * scale)), Image.BILINEAR)
    paste_center(frame, im, W / 2 + shake, H / 2 - 40 + shake * 0.5, 1 - out)
    sub_k = ease_out((t - BEAT) / 0.6)
    paste_center(frame, T_SUBTITLE, W / 2, H / 2 + LOGO.height / 2 + 10 + 20 * (1 - sub_k), sub_k * (1 - out))


def draw_final(frame, t):
    k = ease_out(t / 0.25)
    shake = 22 * math.exp(-t * 4) * math.sin(t * 60)
    scale = 1.25 - 0.25 * k + 0.02 * t
    im = LOGO.resize((int(LOGO.width * scale), int(LOGO.height * scale)), Image.BILINEAR)
    fade = clamp01((TOTAL - 24 * BAR - t - 0.2) / 1.3)
    paste_center(frame, im, W / 2 + shake, H / 2 - 60, k * fade)
    a2 = ease_out((t - 0.6) / 0.6) * fade
    paste_center(frame, T_SUBTITLE, W / 2, H / 2 + LOGO.height / 2 - 10, a2)
    a3 = ease_out((t - 1.1) / 0.6) * fade
    paste_center(frame, T_END1, W / 2, H / 2 + LOGO.height / 2 + 70, a3)


# ------------------------------------------------------------ outro producers (bars 22-23)
T_P1 = [text_img('PRODUCERS', F_ROLE_S, (190, 180, 165), spacing=10), text_img('Obe107   &   samcci', F_NAME_M)]
T_P2 = [text_img('DEVELOPER', F_ROLE_S, (190, 180, 165), spacing=10), text_img('DarkZmaj', F_NAME_M), T_DZ_END]


# ------------------------------------------------------------ post
yy, xx = np.mgrid[0:H, 0:W]
VIG = (1 - 0.55 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) ** 1.4).clip(0.25, 1)
VIG = VIG[..., None].astype(np.float32)
GRAIN = [np.random.default_rng(i).normal(0, 4, (H // 2, W // 2, 1)).astype(np.float32) for i in range(6)]
del yy, xx


def kick_pulse(t):
    """Brightness pulse synced to the kick in groove sections."""
    if not (8 * BAR <= t < 16 * BAR or 18 * BAR <= t < 22 * BAR):
        return 0.0
    s = ((t % BAR) / (BEAT / 4))
    best = 0
    for k in (0, 6, 10):
        d = (s - k) * BEAT / 4
        if 0 <= d < 0.15:
            best = max(best, 1 - d / 0.15)
    return best


def render(t):
    frame = Image.new('RGBA', (W, H), (0, 0, 0, 255))
    bar = t / BAR
    # particles in groove
    pamt = 0
    if 8 <= bar < 22:
        pamt = 0.5
    elif bar >= 24:
        pamt = 0.8 * clamp01((TOTAL - t) / 2)
    draw_particles(frame, t, pamt)
    if bar < 3:
        b = int(bar)
        img, tint = CARDS[b]
        streak_card(frame, img, t - b * BAR, BAR, tint)
    elif bar < 4:
        fade_text(frame, T_PRESENTS, t - 3 * BAR, BAR)
    elif bar < 5:
        fade_text(frame, T_PROD, t - 4 * BAR, BAR)
    elif bar < 6:
        fade_text(frame, T_FOR, t - 5 * BAR, BAR)
    elif bar < 8:
        draw_title_build(frame, t - 6 * BAR)
    elif bar < 10:
        draw_title_hold(frame, t - 8 * BAR, 2 * BAR)
    elif bar < 22:
        for si, s in enumerate(SECTIONS):
            draw_section(frame, si, t - s['bar'] * BAR)
    elif bar < 23:
        fade_text(frame, T_P1, t - 22 * BAR, BAR, gap=-10)
    elif bar < 24:
        fade_text(frame, T_P2, t - 23 * BAR, BAR, gap=-10)
    else:
        draw_final(frame, t - 24 * BAR)

    a = np.asarray(frame.convert('RGB'), dtype=np.float32)
    # white flash on big hits
    flash = 0.0
    for hit, strength in ((8 * BAR, 0.85), (24 * BAR, 1.0)):
        d = t - hit
        if 0 <= d < 0.5:
            flash = max(flash, strength * (1 - d / 0.5) ** 2)
    for b in range(3):  # logo stabs
        d = t - b * BAR
        if 0 <= d < 0.2:
            flash = max(flash, 0.25 * (1 - d / 0.2))
    pulse = kick_pulse(t) * 0.07
    a = a * (1 + pulse)
    if flash > 0:
        a = a * (1 - flash) + 255 * flash
    a *= VIG
    g = GRAIN[int(t * FPS) % len(GRAIN)]
    a += np.repeat(np.repeat(g, 2, 0), 2, 1)
    return np.clip(a, 0, 255).astype(np.uint8)


def main():
    music, out = sys.argv[1], sys.argv[2]
    if '--preview' in sys.argv:
        ts = [float(x) for x in sys.argv[sys.argv.index('--preview') + 1].split(',')]
        for t in ts:
            Image.fromarray(render(t)).save(f'{out}_{t:06.2f}.png')
        return
    n = int(TOTAL * FPS)
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
           '-i', '-', '-i', music, '-c:v', 'libx264', '-preset', 'medium', '-crf', '24', '-tune', 'grain', '-pix_fmt', 'yuv420p',
           '-c:a', 'aac', '-b:a', '256k', '-shortest', '-movflags', '+faststart', out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n):
        p.stdin.write(render(i / FPS).tobytes())
        if i % 150 == 0:
            print(f'frame {i}/{n}', flush=True)
    p.stdin.close()
    p.wait()
    print('done', out)


if __name__ == '__main__':
    main()
