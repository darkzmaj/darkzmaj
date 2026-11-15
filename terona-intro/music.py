"""Original intro score for the TERONA Minecraft server intro.

Cinematic crime-drama style: timpani + brass stabs, strings, pizzicato
ostinato and a boom-bap beat, with a chiptune lead as the "Minecraft" twist.
E minor, 90 BPM. Pure numpy/scipy synthesis, no samples.

Usage: python3 music.py out.wav
"""
import sys
import wave

import numpy as np
from scipy import signal

SR = 44100
BPM = 90
BEAT = 60 / BPM
BAR = 4 * BEAT
STEP = BEAT / 4          # 16th note
TOTAL = 24 * BAR + 4.5   # 24 bars + final hit tail
N = int(TOTAL * SR)
rng = np.random.default_rng(107)

# stereo buses
dry = np.zeros((N, 2))
verb_send = np.zeros((N, 2))
duck_bus = np.zeros((N, 2))      # pads that get sidechained by the kick
kick_times = []


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def t_of(bar, step=0.0):
    return bar * BAR + step * STEP


def env_adsr(n, a, d, s, r, sr=SR):
    a, d, r = int(a * sr), int(d * sr), int(r * sr)
    a = max(a, 1)
    e = np.full(n, s, dtype=float)
    e[:min(a, n)] = np.linspace(0, 1, a)[:min(a, n)]
    if a < n:
        dd = min(d, n - a)
        e[a:a + dd] = np.linspace(1, s, max(d, 1))[:dd]
    if r > 0:
        rr = min(r, n)
        e[n - rr:] *= np.linspace(1, 0, rr)
    return e


def place(buf, t, x, gain=1.0, pan=0.0):
    i = int(t * SR)
    if i >= len(buf):
        return
    x = x[:len(buf) - i]
    l = np.cos((pan + 1) * np.pi / 4)
    r = np.sin((pan + 1) * np.pi / 4)
    if x.ndim == 1:
        buf[i:i + len(x), 0] += x * gain * l * 1.414
        buf[i:i + len(x), 1] += x * gain * r * 1.414
    else:
        buf[i:i + len(x)] += x * gain


def lp(x, fc, order=2):
    sos = signal.butter(order, min(fc, SR * 0.45), 'low', fs=SR, output='sos')
    return signal.sosfilt(sos, x, axis=0)


def hp(x, fc, order=2):
    sos = signal.butter(order, fc, 'high', fs=SR, output='sos')
    return signal.sosfilt(sos, x, axis=0)


def bp(x, lo, hi, order=2):
    sos = signal.butter(order, [lo, hi], 'band', fs=SR, output='sos')
    return signal.sosfilt(sos, x, axis=0)


def saw(f, n, phase=0.0):
    t = np.arange(n) / SR
    return 2 * ((f * t + phase) % 1.0) - 1


def square(f, n, duty=0.5):
    t = np.arange(n) / SR
    return np.where((f * t) % 1.0 < duty, 1.0, -1.0)


# ---------------------------------------------------------------- instruments
def kick(t, g=1.0):
    n = int(0.5 * SR)
    tt = np.arange(n) / SR
    f = 45 + 95 * np.exp(-tt * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) * np.exp(-tt * 7)
    x += 0.4 * np.exp(-tt * 300) * rng.standard_normal(n) * 0.3
    place(dry, t, np.tanh(x * 1.6), 0.9 * g)
    kick_times.append(t)


def snare(t, g=1.0, verb=0.35):
    n = int(0.35 * SR)
    tt = np.arange(n) / SR
    nz = bp(rng.standard_normal(n), 1200, 7000) * np.exp(-tt * 18)
    tone = np.sin(2 * np.pi * 185 * tt) * np.exp(-tt * 30)
    x = 0.8 * nz + 0.6 * tone
    place(dry, t, x, 0.55 * g)
    place(verb_send, t, x, verb * g)


def hat(t, g=1.0, open_=False, pan=0.25):
    n = int((0.35 if open_ else 0.06) * SR)
    tt = np.arange(n) / SR
    x = hp(rng.standard_normal(n), 7000) * np.exp(-tt * (9 if open_ else 70))
    place(dry, t, x, 0.22 * g, pan)


def timpani(t, m=40, g=1.0, dur=2.5):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    f0 = mtof(m)
    f = f0 * (1 + 0.06 * np.exp(-tt * 12))
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = (np.sin(ph) + 0.5 * np.sin(1.5 * ph) * np.exp(-tt * 3)
         + 0.3 * np.sin(2.0 * ph) * np.exp(-tt * 4)) * np.exp(-tt * 1.6)
    x += lp(rng.standard_normal(n), 900) * np.exp(-tt * 25) * 0.8
    place(dry, t, x, 0.55 * g)
    place(verb_send, t, x, 0.35 * g)


def timp_roll(t0, dur, m=40, g=1.0):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    f0 = mtof(m)
    am = 0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 7.5 * tt))
    x = (np.sin(2 * np.pi * f0 * tt) + 0.4 * np.sin(2 * np.pi * f0 * 1.5 * tt))
    x += lp(rng.standard_normal(n), 400) * 0.6
    sw = (tt / dur) ** 2
    place(dry, t0, x * am * sw, 0.5 * g)
    place(verb_send, t0, x * am * sw, 0.3 * g)


def brass(t, notes, dur=0.32, g=1.0, verb=0.3):
    n = int((dur + 0.25) * SR)
    tt = np.arange(n) / SR
    x = np.zeros(n)
    for m in notes:
        f = mtof(m)
        for det, ph in ((-0.12, 0.1), (0.0, 0.5), (0.12, 0.8)):
            x += saw(f * 2 ** (det / 12), n, ph)
    x /= 3 * len(notes)
    dark, bright = lp(x, 600, 2), lp(x, 4200, 2)
    fe = np.exp(-tt * 9)
    y = dark * (1 - fe) + bright * fe
    e = env_adsr(n, 0.008, 0.12, 0.55, 0.18)
    e[int(dur * SR):] *= np.exp(-np.arange(n - int(dur * SR)) / SR * 14)
    y = np.tanh(y * e * 2.2)
    place(dry, t, y, 0.42 * g)
    place(verb_send, t, y, verb * g)


def strings(t, notes, dur, g=1.0, attack=0.6, bright=2600, to_duck=True):
    n = int((dur + 1.2) * SR)
    tt = np.arange(n) / SR
    L = np.zeros(n)
    R = np.zeros(n)
    for m in notes:
        f = mtof(m)
        for k in range(5):
            det = (k - 2) * 0.07
            vib = 1 + 0.003 * np.sin(2 * np.pi * (5 + k * 0.3) * tt + k)
            ph = np.cumsum(f * 2 ** (det / 12) * vib) / SR + rng.random()
            s = 2 * (ph % 1.0) - 1
            if k % 2:
                L += s
            else:
                R += s
    x = np.stack([L, R], 1) / (5 * len(notes))
    x = lp(x, bright, 2)
    x = hp(x, 120)
    e = env_adsr(n, attack, 0.3, 0.85, 1.1)
    x *= e[:, None]
    place(duck_bus if to_duck else dry, t, x, 0.55 * g)
    place(verb_send, t, x, 0.45 * g)


def tremolo_swell(t, notes, dur, g=1.0):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    x = np.zeros(n)
    for m in notes:
        x += saw(mtof(m), n, rng.random()) + saw(mtof(m) * 1.004, n, rng.random())
    x = lp(x / (2 * len(notes)), 3500)
    trem = 0.55 + 0.45 * np.sin(2 * np.pi * 12 * tt) ** 2
    sw = (tt / dur) ** 1.8
    place(dry, t, x * trem * sw, 0.35 * g, -0.2)
    place(verb_send, t, x * trem * sw, 0.5 * g)


def pizz(t, m, g=1.0, pan=0.0):
    f = mtof(m)
    n = int(0.6 * SR)
    p = max(int(SR / f), 2)
    buf = lp(rng.standard_normal(p), f * 6)
    out = np.zeros(n)
    for i in range(0, n, p):
        seg = buf[:min(p, n - i)]
        out[i:i + len(seg)] = seg
        buf = 0.5 * (buf + np.roll(buf, 1)) * 0.985
    tt = np.arange(n) / SR
    out *= np.exp(-tt * 6)
    out = lp(out, 2800)
    place(dry, t, out, 0.5 * g, pan)
    place(verb_send, t, out, 0.25 * g)


def piano(t, m, dur=1.5, g=1.0, pan=0.1):
    f = mtof(m)
    n = int((dur + 0.6) * SR)
    tt = np.arange(n) / SR
    x = np.zeros(n)
    for h, a in ((1, 1.0), (2, 0.45), (3, 0.25), (4, 0.12), (5, 0.08), (6, 0.04)):
        x += a * np.sin(2 * np.pi * f * h * (1 + 0.0004 * h * h) * tt) * np.exp(-tt * (1.2 + h * 0.9))
    x += 0.15 * hp(rng.standard_normal(n), 2000) * np.exp(-tt * 80)
    e = np.ones(n)
    e[int(dur * SR):] = np.exp(-np.arange(n - int(dur * SR)) / SR * 10)
    x *= e * np.minimum(1, tt / 0.003)
    place(dry, t, x, 0.32 * g, pan)
    place(verb_send, t, x, 0.4 * g)


def bass(t, m, dur, g=1.0):
    f = mtof(m)
    n = int((dur + 0.05) * SR)
    tt = np.arange(n) / SR
    x = saw(f, n) * 0.6 + np.sin(2 * np.pi * f * tt) * 0.9
    x = lp(x, 420 + 900 * 0, 3)
    x *= env_adsr(n, 0.004, 0.15, 0.7, 0.05)
    place(dry, t, np.tanh(x * 1.5), 0.42 * g)


def chip(t, m, dur, g=1.0, duty=0.25, pan=-0.15):
    f = mtof(m)
    n = int(dur * SR)
    tt = np.arange(n) / SR
    vib = 1 + 0.006 * np.sin(2 * np.pi * 6 * tt) * np.minimum(1, tt / 0.15)
    ph = np.cumsum(f * vib) / SR
    x = np.where(ph % 1.0 < duty, 1.0, -1.0)
    x *= env_adsr(n, 0.003, 0.05, 0.8, 0.03)
    x = lp(x, 6000)
    place(dry, t, x, 0.085 * g, pan)
    place(verb_send, t, x, 0.06 * g)
    # echo (dotted 8th) for width
    place(dry, t + 3 * STEP, x, 0.03 * g, -pan * 3)


def riser(t, dur, g=1.0):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    x = rng.standard_normal(n)
    out = np.zeros(n)
    chunks = 48
    for c in range(chunks):
        a, b = c * n // chunks, (c + 1) * n // chunks
        fc = 300 * (40 ** (c / chunks))
        out[a:b] = lp(x[max(0, a - 2000):b], fc)[-(b - a):]
    out *= (tt / dur) ** 2
    place(dry, t, out, 0.25 * g, 0.1)
    place(verb_send, t, out, 0.3 * g)


def impact(t, g=1.0):
    n = int(3.5 * SR)
    tt = np.arange(n) / SR
    f = 30 + 80 * np.exp(-tt * 10)
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 1.3)
    nz = lp(rng.standard_normal(n), 2500) * np.exp(-tt * 5)
    x = np.tanh(1.5 * sub) + 0.5 * nz
    place(dry, t, x, 0.7 * g)
    place(verb_send, t, x, 0.6 * g)


def snare_roll(t0, dur, g=1.0):
    k = 0.0
    while k < dur:
        frac = k / dur
        snare(t0 + k, (0.15 + 0.85 * frac ** 1.5) * g, verb=0.2)
        k += STEP if frac < 0.5 else STEP / 2


# ---------------------------------------------------------------- harmony
EM, C, AM, B = 'Em', 'C', 'Am', 'B'
CHORDS = {
    EM: dict(root=40, pad=[52, 55, 59, 64], stab=[64, 67, 71, 76], tones=[52, 55, 59, 64, 67]),
    C:  dict(root=36, pad=[48, 52, 55, 60], stab=[64, 67, 72, 76], tones=[48, 52, 55, 60, 64]),
    AM: dict(root=33, pad=[45, 48, 52, 57], stab=[64, 69, 72, 76], tones=[45, 48, 52, 57, 60]),
    B:  dict(root=35, pad=[47, 51, 54, 57], stab=[63, 66, 71, 75], tones=[47, 51, 54, 59, 63]),
}
PROG = [EM, C, AM, B]
H = '-'
MEL_A = [
    [76, H, 79, 78, 76, H, 71, H],
    [72, H, 76, 74, 72, H, 67, H],
    [69, 72, 76, H, 74, 72, 71, 69],
    [71, H, 75, H, 78, H, 83, H],
]
MEL_B = [
    [79, H, 78, 76, 74, 76, 71, H],
    [72, 74, 76, 79, H, 76, 74, 72],
    [69, H, 72, 76, 81, H, 79, 76],
    [78, H, 75, H, 71, H, H, H],
]


def melody(bar0, mel, inst, g=1.0, octave=0):
    for bi, row in enumerate(mel):
        for i, m in enumerate(row):
            if m in (H, None):
                continue
            j = i + 1
            while j < len(row) and row[j] == H:
                j += 1
            dur = (j - i) * 2 * STEP
            inst(t_of(bar0 + bi, i * 2), m + octave, dur, g)


def piano_inst(t, m, dur, g):
    piano(t, m, dur * 1.1, g)
    piano(t, m - 12, dur * 1.1, g * 0.35)


def brass_lead(t, m, dur, g):
    brass(t, [m, m - 12], dur * 0.9, g * 0.7, verb=0.35)


def chip_inst(t, m, dur, g):
    chip(t, m, dur * 0.95, g)


def drums_bar(bar, g=1.0, fill=False):
    for s in (0, 6, 10):
        kick(t_of(bar, s), g)
    if fill:
        kick(t_of(bar, 14), 0.7 * g)
    for s in (4, 12):
        snare(t_of(bar, s), g)
    snare(t_of(bar, 7), 0.18 * g, verb=0.05)
    snare(t_of(bar, 15), 0.22 * g, verb=0.05)
    for s in range(0, 16, 2):
        hat(t_of(bar, s), (1.0 if s % 4 == 0 else 0.6) * g, open_=(s == 14))
    for s in (3, 11):
        hat(t_of(bar, s), 0.3 * g, pan=-0.3)


def bass_bar(bar, ch, g=1.0):
    r = CHORDS[ch]['root']
    for s, off, d in ((0, 0, 3), (3, 0, 2), (6, 12, 2), (8, 0, 3), (11, 0, 2), (14, 7, 2)):
        bass(t_of(bar, s), r + off, d * STEP * 0.9, g)


def pizz_bar(bar, ch, g=1.0):
    tn = CHORDS[ch]['tones']
    pat = [0, 2, 4, 2, 1, 2, 3, 2]
    for i, k in enumerate(pat):
        pizz(t_of(bar, i * 2), tn[k] + 12, g * (1.0 if i % 2 == 0 else 0.7), pan=0.35 if i % 2 else -0.35)


def stabs_bar(bar, ch, g=1.0):
    st = CHORDS[ch]['stab']
    brass(t_of(bar, 0), st, 0.3, g)
    brass(t_of(bar, 10), st, 0.18, 0.7 * g)


def chip_arp_bar(bar, ch, g=1.0):
    tn = CHORDS[ch]['tones']
    seq = [0, 1, 2, 3, 4, 3, 2, 1, 0, 2, 4, 2, 3, 4, 3, 2]
    for s, k in enumerate(seq):
        chip(t_of(bar, s), tn[k] + 24, STEP * 0.8, 0.55 * g, duty=0.125, pan=0.3)


# ---------------------------------------------------------------- arrangement
# Bars 0-2: logo hits (timpani + brass stab), drone, swells
for b in range(3):
    timpani(t_of(b), 40, 1.0)
    brass(t_of(b), CHORDS[EM]['stab'], 0.5, 1.0, verb=0.5)
    brass(t_of(b), [40, 47, 52], 0.5, 0.8, verb=0.4)
    brass(t_of(b, 14), [71, 75, 78], 0.12, 0.45)        # pickup stab
strings(t_of(0), [28, 40, 47], 4 * BAR, 0.8, attack=1.5, bright=900, to_duck=False)
tremolo_swell(t_of(0), [64, 71, 76], 3 * BAR, 0.8)

# Bar 3: "presents" - piano motif over Em
for i, m in enumerate([64, 67, 71, 76, 75, 71]):
    piano(t_of(3, i * 2 + 2), m, 1.2, 0.8)
timpani(t_of(3), 40, 0.6)

# Bars 4-5: "a DarkZmaj production" - strings + pizz + piano motif
strings(t_of(4), CHORDS[EM]['pad'], BAR, 0.9, attack=0.4, to_duck=False)
strings(t_of(5), CHORDS[C]['pad'], BAR, 0.9, attack=0.4, to_duck=False)
pizz_bar(4, EM, 0.8)
pizz_bar(5, C, 0.8)
melody(4, MEL_A[:2], piano_inst, 0.9)
bass(t_of(4), 40, BAR * 0.95, 0.6)
bass(t_of(5), 36, BAR * 0.95, 0.6)

# Bars 6-7: build to title
strings(t_of(6), CHORDS[AM]['pad'], BAR, 1.0, attack=0.3, to_duck=False)
strings(t_of(7), CHORDS[B]['pad'] + [63], BAR - BEAT, 1.1, attack=0.3, to_duck=False)
pizz_bar(6, AM, 0.9)
for i in range(6):
    pizz(t_of(7, i * 2), CHORDS[B]['tones'][i % 5] + 12, 1.0)
timp_roll(t_of(6), 2 * BAR - BEAT, 40, 1.0)
snare_roll(t_of(6, 8), 1.5 * BAR - BEAT, 0.9)
riser(t_of(6), 2 * BAR - BEAT * 0.5, 1.0)
for i, m in enumerate([71, 75, 78, 83]):          # rising brass on B
    brass(t_of(7, i * 3), [m, m - 12], 0.25, 0.7)

# Bars 8-15: TITLE drop + credits, full groove
impact(t_of(8), 0.9)
timpani(t_of(8), 40, 1.2)
for b in range(8, 16):
    ch = PROG[(b - 8) % 4]
    drums_bar(b, 1.0, fill=(b % 4 == 3))
    bass_bar(b, ch)
    pizz_bar(b, ch, 0.7)
    stabs_bar(b, ch, 0.9)
    strings(t_of(b), CHORDS[ch]['pad'], BAR, 0.75, attack=0.15)
melody(8, MEL_A, piano_inst, 1.0)
melody(8, MEL_A, brass_lead, 0.55)
melody(12, MEL_B, brass_lead, 0.9)
melody(12, MEL_B, piano_inst, 0.7, octave=-12)
for b in (11, 15):
    for s in (12, 13, 14, 15):                       # tom-ish fill
        timpani(t_of(b, s), 45 - (s - 12) * 2, 0.35, dur=0.6)

# Bars 16-17: break - chiptune twist, no drums
for b in (16, 17):
    ch = [EM, C][b - 16]
    strings(t_of(b), CHORDS[ch]['pad'], BAR, 0.7, attack=0.25, to_duck=False)
    chip_arp_bar(b, ch, 0.9)
    pizz_bar(b, ch, 0.5)
    bass(t_of(b), CHORDS[ch]['root'], BAR * 0.95, 0.5)
melody(16, MEL_A[:2], chip_inst, 1.0, octave=0)
hat(t_of(17, 12), 0.5, True)
riser(t_of(17), BAR, 0.6)
snare_roll(t_of(17, 8), BAR / 2, 0.6)

# Bars 18-21: full again + chiptune counter-melody
for b in range(18, 22):
    ch = PROG[(b - 18 + 2) % 4]        # Am B Em C ... resolves nicer
    drums_bar(b, 1.0, fill=(b == 21))
    bass_bar(b, ch)
    pizz_bar(b, ch, 0.6)
    stabs_bar(b, ch, 0.9)
    strings(t_of(b), CHORDS[ch]['pad'], BAR, 0.8, attack=0.15)
    chip_arp_bar(b, ch, 0.5)
melody(18, MEL_A[2:] + MEL_B[:2], brass_lead, 1.0)
melody(18, MEL_A[2:] + MEL_B[:2], chip_inst, 0.7, octave=12)
timpani(t_of(18), 45, 0.8)

# Bars 22-23: outro on Am -> B, drums out
strings(t_of(22), CHORDS[AM]['pad'] + [69], BAR, 1.0, attack=0.2, to_duck=False)
strings(t_of(23), CHORDS[B]['pad'] + [75], BAR - STEP * 2, 1.1, attack=0.2, to_duck=False)
for i, m in enumerate([69, 72, 76, 81, 79, 76, 72, 69]):
    piano(t_of(22, i * 2), m, 0.8, 0.8)
melody(23, [MEL_A[3]], brass_lead, 0.9)
bass(t_of(22), 33, BAR * 0.95, 0.6)
bass(t_of(23), 35, BAR * 0.95, 0.6)
timp_roll(t_of(23), BAR - STEP, 47, 0.9)
snare_roll(t_of(23, 4), BAR * 0.75 - STEP, 0.8)

# Bar 24: final hit - tutti Em
END = t_of(24)
impact(END, 1.1)
timpani(END, 40, 1.4, dur=4)
brass(END, [52, 59, 64, 67, 71, 76], 1.8, 1.1, verb=0.7)
brass(END, [28, 40, 47], 1.8, 0.9, verb=0.5)
strings(END, [40, 52, 55, 59, 64, 71, 76], 2.0, 1.2, attack=0.02, to_duck=False)
kick(END, 1.0)
piano(END + BEAT, 76, 2.5, 0.6)
piano(END + BEAT, 64, 2.5, 0.4)

# ---------------------------------------------------------------- mix
# sidechain duck for pads
duck = np.ones(N)
dl = int(0.28 * SR)
shape = 1 - 0.55 * np.exp(-np.arange(dl) / SR * 14)
for kt in kick_times:
    i = int(kt * SR)
    seg = duck[i:i + dl]
    duck[i:i + dl] = np.minimum(seg, shape[:len(seg)])
dry += duck_bus * duck[:, None]

# reverb: decaying stereo noise IR
ir_n = int(2.8 * SR)
tt = np.arange(ir_n) / SR
ir = rng.standard_normal((ir_n, 2)) * np.exp(-tt * 2.6)[:, None]
ir = lp(ir, 5000)
ir[:int(0.012 * SR)] = 0
ir /= np.sqrt((ir ** 2).sum(0))
wet = np.stack([signal.fftconvolve(verb_send[:, c], ir[:, c])[:N] for c in range(2)], 1)
wet = hp(wet, 180)
mix = dry + 0.55 * wet

# gentle master: high-pass rumble, glue, soft clip, normalise
mix = hp(mix, 28)
mix /= np.max(np.abs(mix)) + 1e-9
mix = np.tanh(mix * 1.6) / np.tanh(1.6)
fade = int(1.5 * SR)
mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix *= 0.95 / np.max(np.abs(mix))

out = sys.argv[1] if len(sys.argv) > 1 else 'music.wav'
with wave.open(out, 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype('<i2').tobytes())
print(f'wrote {out}: {TOTAL:.2f}s, bar={BAR:.4f}s')
