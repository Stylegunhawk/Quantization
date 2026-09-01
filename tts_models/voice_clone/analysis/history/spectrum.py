"""Where does the audio actually stop? Container rate lies; lossy codecs low-pass."""
import numpy as np, soundfile as sf

def cutoff(path):
    a, sr = sf.read(path, dtype="float32", always_2d=False)
    if a.ndim > 1: a = a.mean(1)
    n = 1 << 15
    # average magnitude spectrum over many windows, so one frame's content can't skew it
    frames = [a[i:i+n] for i in range(0, max(len(a)-n, 1), n)][:80]
    mag = np.mean([np.abs(np.fft.rfft(f * np.hanning(len(f)), n)) for f in frames if len(f) == n], axis=0)
    freqs = np.fft.rfftfreq(n, 1/sr)
    db = 20*np.log10(mag/ (mag.max()+1e-12) + 1e-12)
    # highest frequency still within 40dB of the peak = practical bandwidth
    above = np.where(db > -40)[0]
    return sr, freqs[above[-1]] if len(above) else 0, len(a)/sr

for p in ["voice_ref2/ref2_48k.wav", "voice_ref/ref_48k.wav"]:
    sr, hi, dur = cutoff(p)
    print(f"{p:34} container {sr:>5}Hz | real content to ~{hi/1000:5.1f} kHz | {dur:5.1f}s")
