"""Bandwidth measured on VOICED frames only.

Averaging every frame lets silence and whispers drag the mean spectrum down, which
made the second recording look narrower than the first when it may not be. Gate on
frame energy first: only frames near the loudest parts describe the mic's real reach.
"""
import numpy as np, soundfile as sf

def cutoff(path, pct=75):
    a, sr = sf.read(path, dtype="float32", always_2d=False)
    if a.ndim > 1: a = a.mean(1)
    n = 1 << 14
    frames = np.array([a[i:i+n] for i in range(0, len(a)-n, n)])
    rms = np.sqrt((frames**2).mean(1))
    keep = frames[rms >= np.percentile(rms, pct)]          # loudest quarter = clearly voiced
    mag = np.mean([np.abs(np.fft.rfft(f*np.hanning(n), n)) for f in keep], axis=0)
    freqs = np.fft.rfftfreq(n, 1/sr)
    db = 20*np.log10(mag/(mag.max()+1e-12) + 1e-12)
    above = np.where(db > -40)[0]
    return sr, freqs[above[-1]] if len(above) else 0, len(a)/sr, len(keep)

for p in ["voice_ref3/ref3_48k.wav", "voice_ref2/ref2_48k.wav", "voice_ref/ref_48k.wav"]:
    sr, hi, dur, nf = cutoff(p)
    print(f"{p:30} voiced content to ~{hi/1000:5.1f} kHz | {dur:5.1f}s ({nf} voiced frames)")
