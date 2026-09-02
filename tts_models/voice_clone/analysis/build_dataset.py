#!/usr/bin/env python
"""Cut takes into 24kHz training clips paired with their EXACT script text.

The recording scripts are known verbatim, so ASR is used only to locate words in
time -- never to supply the text. Parakeet transcribes at ~14% WER on this
material ("cachet" for "cache", "six lines" for "six hundred lines"); training a
TTS model on that teaches a wrong text->audio mapping directly. So each passage
is aligned to the audio and the row carries the script's own words.

Run from this directory:  ../../.venv/bin/python build_dataset.py
Self-check:               CHECK=1 ../../.venv/bin/python build_dataset.py
"""
import difflib
import json
import os
import re
import subprocess
from pathlib import Path

import soundfile as sf
from scipy.signal import resample_poly
from mlx_audio.stt.generate import generate_transcription
from mlx_audio.stt.utils import load_model
from num2words import num2words

HERE = Path(__file__).resolve().parent
EXP = HERE.parent

OUT = EXP / "dataset/finetune_data"
SCRIPTS = EXP / "scripts"

# Drop your raw takes here -- one file per recording script, named after it
# (script_1_work.m4a pairs with scripts/script_1_work.txt). Override with
# RECORDINGS_DIR=/path/to/dir. Deliberately NOT recursive: recordings/ also holds
# generated clones (s2_icl_*.wav), and those must never end up in training data.
SRC_DIR = Path(os.environ.get("RECORDINGS_DIR", EXP / "recordings/raw"))
SOURCES = sorted(p for p in SRC_DIR.glob("*") if p.suffix.lower() in (".wav", ".m4a"))

# The 10s ref_audio every row conditions on. This MUST be the same file
# timbre_score.py anchors to -- if the training ref changes between runs, the
# 9.5%-vs-next-run comparison the whole experiment turns on is uncontrolled.
# So it defaults to timbre_score.py's default, and only falls back to the first
# take on a fresh clone where that file does not exist. Override with VOICE_REF.
_SCORING_REF = EXP / "recordings/voice_ref3/ref3_48k.wav"
REF_TAKE = Path(os.environ["VOICE_REF"]) if os.environ.get("VOICE_REF") else (
    _SCORING_REF if _SCORING_REF.exists() else SOURCES[0] if SOURCES else None)

# Parakeet mel-encodes whatever you hand it in one Metal buffer, so a 12.7-min
# take asks for 5.8GB against this machine's 4GB cap and dies. Its own
# chunk_duration= is NOT the way out: the overlap token-merge silently dropped
# 18% of script_1 (136s of speech, one stretch 26.7s) while reporting success.
# Accuracy also decays with length -- measured coverage of one 110s region:
#   20s 94%   30s 92%   45s 93%   60s 74%   110s 55%
# and passages that came back mangled at 40s transcribed perfectly at 10-15s.
# So use short fixed pieces. Cuts land mid-word, which is why they overlap; the
# duplicate words are harmless because the script alignment below, not
# Parakeet's merge, decides what the text is. Level needs no correction:
# normalising this take from -38 to -20 dBFS gave byte-identical transcripts.
PIECE_SEC, OVERLAP_SEC = 15.0, 3.0

# Alignment coverage = fraction of a passage's words found, in order, in the
# audio. It is the deviation detector, not just a quality score: a speaker who
# says something other than the script must NOT get the script text stapled to
# their audio. Above TRUST the difference is ASR noise and the script wins.
# Below REVIEW it is a real deviation -- a self-correction or a skipped clause --
# and neither source can be trusted, so the clip is dropped. In between the clip
# is kept with script text and listed in alignment_review.json to spot-check.
TRUST, REVIEW = 0.90, 0.70

# A stray common word ("the") matching far from its passage would stretch that
# passage's span across half the recording. Matches more than GAP words apart
# belong to different runs; only the densest run is believed.
GAP = 6

# PAD reclaims the breath either side of a clip. MAX_SEC=25 because script_3 #70
# is 44 words -- 18s at 145wpm but 21s at 125. MIN_SEC stays low on purpose:
# script_3 passages 73-76 are one-word reactions ("Yes." "Exactly.") written to
# capture short-utterance breathing, which passage 77 says outright differs from
# long-sentence breathing.
MIN_SEC, MAX_SEC, PAD = 0.4, 25.0, 0.12

# 24kHz is the tokenizer's native rate, NOT an upload-size saving. Verified
# against upstream, not a fork -- reference/upstream_dataset.py, fetched from
# QwenLM/Qwen3-TTS finetuning/dataset.py, the file Colab imports:
#   :45   audio, sr = librosa.load(x, sr=None, mono=True)   <- native rate
#   :105  assert sr == 24000, "Only support 24kHz audio"
# and sft_12hz.py:23 is `from dataset import TTSDataset`, so training runs that
# loader. speech_tokenizer/preprocessor_config.json declares sampling_rate
# 24000 independently. A 48kHz dataset therefore does not train at all: it dies
# on an AssertionError at the first batch, ~15 min in, after the base download
# and tokenisation. Nothing resamples for us -- prepare_data.py hands bare paths
# to encode(), which is what made 48kHz look survivable. It is not.
#
# Nothing is lost. upstream_dataset.py:114 sets mel fmax=12000, so the model
# never looks above 12kHz, and 24kHz sampling puts Nyquist exactly there. The
# source is lossy AAC from Voice Memos with no real content near that ceiling.
#
# resample_poly (not librosa.resample) to match analysis/history/make_24k.py,
# which produced the 24k set behind the 9.5% run -- so the new dataset differs
# from that one by content only, never by resampler. Applied to the whole take
# once, before cutting, so no clip carries polyphase filter transients at its
# edges. Timings stay in seconds and so are unaffected.
TARGET_SR = 24000


def norm(s: str) -> list[str]:
    """Words for comparison only -- never for output. Output text is the script's.

    ASR writes "40" where the scripts spell "forty"; without folding digits to
    words those read as speaker deviations rather than a notation difference.
    """
    s = re.sub(r"\d+", lambda m: " " + num2words(int(m.group())) + " ", s.lower())
    return re.sub(r"[^a-z' ]", " ", s).split()


def load_script(path: Path) -> list[tuple[int, str]]:
    """(passage number, exact text) for every numbered line in a script."""
    return [
        (int(m.group(1)), m.group(2).strip())
        for line in path.read_text().splitlines()
        if (m := re.match(r"^(\d+)\.\s+(.*)", line))
    ]


def to_target(audio, sr: int):
    """Mono at TARGET_SR. A no-op when the take is already there."""
    if audio.ndim > 1:
        audio = audio.mean(1)
    if sr == TARGET_SR:
        return audio, sr
    return resample_poly(audio, TARGET_SR, sr).astype("float32"), TARGET_SR


def decode(src: Path):
    """Samples for a take, converting AAC first if need be.

    libsndfile has no AAC decoder, so the .m4a that Voice Memos hands you -- the
    format the recording scripts tell you to use -- cannot be opened directly.
    macOS ships afconvert, so shell out to that rather than take an ffmpeg
    dependency. Converted every run, not cached: a re-recorded take reusing the
    same filename would otherwise silently train on the old audio.
    """
    if src.suffix.lower() != ".m4a":
        return sf.read(str(src), dtype="float32")
    wav = Path(os.environ.get("TMPDIR", "/tmp")) / f"{src.stem}_48k.wav"
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16", str(src), str(wav)], check=True
    )
    return sf.read(str(wav), dtype="float32")


def transcribe(model, audio, sr: int, tmp: str) -> list[dict]:
    """Words with absolute start/end seconds, read off short overlapping pieces."""
    piece = Path(os.environ.get("TMPDIR", "/tmp")) / "_piece.wav"
    total, lo, words = len(audio) / sr, 0.0, []
    while lo < total:
        hi = min(lo + PIECE_SEC, total)
        sf.write(str(piece), audio[int(lo * sr) : int(hi * sr)], sr)
        result = generate_transcription(
            model=model, audio=str(piece), output_path=tmp, format="txt"
        )
        for sent in result.sentences:
            for tk in sent.tokens:
                start = tk.start + lo
                # re-heard words from the previous piece's overlap
                if words and start < words[-1]["e"] - 0.05:
                    continue
                # subword pieces: a leading space starts a word, 'ing' continues one
                if tk.text.startswith(" ") or not words:
                    words.append({"w": tk.text.strip(), "s": start, "e": tk.end + lo})
                else:
                    words[-1]["w"] += tk.text
                    words[-1]["e"] = tk.end + lo
        if hi >= total:
            break
        lo = hi - OVERLAP_SEC
    return words


def align(passages: list[tuple[int, str]], words: list[dict]) -> tuple[list, list]:
    """Match each passage to a span of audio. Returns (rows, report)."""
    heard = [(norm(w["w"]) or [""])[0] for w in words]
    script, owner = [], []
    for i, (_, text) in enumerate(passages):
        for w in norm(text):
            script.append(w)
            owner.append(i)

    sm = difflib.SequenceMatcher(None, script, heard, autojunk=False)
    hits: dict[int, list[int]] = {}
    for i1, j1, size in sm.get_matching_blocks():
        for k in range(size):
            hits.setdefault(owner[i1 + k], []).append(j1 + k)

    rows, report, prev_end = [], [], -1.0
    for i, (num, text) in enumerate(passages):
        need = sum(1 for o in owner if o == i)
        idx = sorted(hits.get(i, []))
        runs, cur = [], [idx[0]] if idx else []
        for a, b in zip(idx, idx[1:]):
            if b - a <= GAP:
                cur.append(b)
            else:
                runs.append(cur)
                cur = [b]
        if cur:
            runs.append(cur)
        best = max(runs, key=len) if runs else []
        cov = len(best) / need

        said = " ".join(heard[best[0] : best[-1] + 1]) if best else ""
        if cov < REVIEW:
            report.append({"passage": num, "coverage": round(cov, 2),
                           "verdict": "dropped -- spoken words differ from the script",
                           "script": text, "heard": said})
            continue

        start = max(words[best[0]]["s"] - PAD, prev_end)
        end = words[best[-1]]["e"] + PAD
        if not MIN_SEC <= end - start <= MAX_SEC:
            report.append({"passage": num, "coverage": round(cov, 2),
                           "verdict": f"dropped -- {end - start:.1f}s outside "
                                      f"[{MIN_SEC}, {MAX_SEC}]s",
                           "script": text, "heard": said})
            continue
        prev_end = end
        if cov < TRUST:
            report.append({"passage": num, "coverage": round(cov, 2),
                           "verdict": "kept with script text -- listen to confirm",
                           "at": round(start, 1), "script": text, "heard": said})
        rows.append({"num": num, "text": text, "start": start, "end": end, "cov": cov})
    return rows, report


def main() -> None:
    if not SOURCES:
        raise SystemExit(
            f"no .wav/.m4a in {SRC_DIR}\n"
            "The author's recordings are not in the repo. Record the passages in\n"
            "../scripts/*.txt -- one continuous file each -- and put them there."
        )
    (OUT / "wavs").mkdir(parents=True, exist_ok=True)
    for stale in (OUT / "wavs").glob("*.wav"):
        stale.unlink()  # numbering restarts each run; leftovers would be orphans
    model = load_model("mlx-community/parakeet-tdt-0.6b-v3")
    tmp = os.environ.get("TMPDIR", "/tmp") + "/w"

    rows, report, n, total = [], [], 0, 0.0
    for src in SOURCES:
        script_path = SCRIPTS / f"{src.stem}.txt"
        if not script_path.exists():
            raise SystemExit(
                f"{src.name} has no script at {script_path}\n"
                "Clips are paired with exact script text, so each take must be named\n"
                f"after the script it reads: {', '.join(p.name for p in sorted(SCRIPTS.glob('*.txt')))}"
            )
        passages = load_script(script_path)
        audio, sr = decode(src)
        matched, notes = align(passages, transcribe(model, audio, sr, tmp))
        audio, sr = to_target(audio, sr)
        for r in matched:
            n += 1
            total += r["end"] - r["start"]
            path = OUT / "wavs" / f"utt{n:04d}.wav"
            sf.write(str(path), audio[int(r["start"] * sr) : int(r["end"] * sr)], sr)
            rows.append({"audio": f"./wavs/{path.name}", "text": r["text"],
                         "ref_audio": "./ref.wav"})
        report += [{"take": src.name, **x} for x in notes]
        print(f"{src.name}: {len(matched)}/{len(passages)} passages aligned")

    # One consistent ref_audio for every row -- the recipe conditions on it per batch,
    # so varying it would make the speaker embedding a moving target.
    audio, sr = to_target(*decode(REF_TAKE))
    sf.write(str(OUT / "ref.wav"), audio[: int(10 * sr)], sr)
    (OUT / "train_raw.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (OUT / "alignment_review.json").write_text(json.dumps(report, indent=1))

    print(f"\n{n} clips @ {TARGET_SR}Hz, {total:.1f}s ({total / 60:.1f} min) "
          f"-> {OUT}/train_raw.jsonl")
    dropped = [x for x in report if x["verdict"].startswith("dropped")]
    print(f"{len(dropped)} dropped, {len(report) - len(dropped)} kept but flagged "
          f"-> {OUT}/alignment_review.json")
    if total < 600:
        print(f"WARNING: {total / 60:.1f} min. The 1.7-min run scored 9.5%, below a "
              f"generic TTS voice. Target 30-45 min before training again.")


def _check() -> None:
    """Coverage decides whether script text may be trusted, so test that call."""
    passages = [(1, "The cat sat on the mat."), (2, "Rain fell all afternoon.")]
    def fake(seq, t=0.0):
        out = []
        for w in seq:
            out.append({"w": w, "s": t, "e": t + 0.4}); t += 0.5
        return out

    # clean read -> both kept, text is the script's (punctuation and all)
    rows, rep = align(passages, fake("the cat sat on the mat rain fell all afternoon".split()))
    assert len(rows) == 2, rows
    assert rows[0]["text"] == "The cat sat on the mat.", rows[0]
    assert not [x for x in rep if x["verdict"].startswith("dropped")], rep

    # passage 2 spoken differently -> dropped, NOT given the script's words
    rows, rep = align(passages, fake("the cat sat on the mat i forgot my line entirely".split()))
    assert [r["num"] for r in rows] == [1], rows
    assert rep[0]["passage"] == 2 and rep[0]["verdict"].startswith("dropped"), rep

    # spans must not overlap, or two clips would share audio
    rows, _ = align(passages, fake("the cat sat on the mat rain fell all afternoon".split()))
    assert rows[1]["start"] >= rows[0]["end"] - 1e-6, rows
    print("align ok")


if __name__ == "__main__":
    if os.environ.get("CHECK"):
        _check()
    else:
        main()
