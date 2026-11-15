# TERONA – server intro

A 68-second cinematic intro for the **Terona** Minecraft server, with an original score.
The style is a comic-panel crime-drama opening (based on a reference clip), redone with pixel-art
Minecraft scenes.

**Credits shown:** KORI Network (presents / plugins & bots) · DarkZmaj (production, lead developer) ·
Obe107 & samcci (creators / founders of Terona).

## Files
- `output/terona-intro.mp4`: final video, 1920×1080, 30 fps, AAC audio
- `output/terona-intro-music.mp3` / `.wav`: the music on its own
- `music.py`: synthesizes the score (E minor, 90 BPM, timpani + brass stabs, strings,
  pizzicato, boom-bap drums and a chiptune lead). Uses numpy and scipy, no samples.
- `video.py`: renders every frame with Pillow, synced to the score, and muxes it with ffmpeg
- `assets/`: KORI art/icon (from darkzmaj.eu), DarkZmaj skin renders (mc-heads.net), OFL fonts
  (Oswald, Cinzel, Press Start 2P, Silkscreen)

## Timeline (1 bar = 2.67 s)
| Bars | Time | Scene |
|---|---|---|
| 0–2 | 0:00–0:08 | Logo hits: KORI Network, DarkZmaj, kori.my |
| 3–5 | 0:08–0:16 | "KORI Network presents", "A DarkZmaj Production", "for Obe107 & samcci" |
| 6–9 | 0:16–0:27 | TERONA block title builds letter by letter, then drops with an impact |
| 10–21 | 0:27–0:59 | Comic-panel credits (Created by Obe107, samcci, Lead Developer, KORI Network, Founders, Starring) |
| 22–23 | 0:59–1:04 | Producers / Developer cards |
| 24 | 1:04–1:08 | Final TERONA logo hit |

## Re-render
```bash
pip install numpy scipy pillow     # ffmpeg must also be installed
python3 music.py music.wav
python3 video.py music.wav terona-intro.mp4
# quick stills: python3 video.py music.wav still --preview 22,30,45
```
Credit text, names and panel art are set in `SECTIONS` in `video.py`. Obe107 and samcci have no
Java-edition skins on mc-heads.net, so their panels use pixel scenes. To show their skins, add their
renders to `assets/` and refer to them in `art_panel()`.
