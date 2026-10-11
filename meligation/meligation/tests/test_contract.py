"""계약 테스트: python -m pytest tests  (pytest 없으면 python tests/test_contract.py)"""
import csv
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meligation import (CLASSISM_PARAM_HASH, MelExtractor, OnsetDetector, Options,  # noqa: E402
                        Regime, build_session, quantize_int8)
from meligation.wavio import read_segment  # noqa: E402

SR = 48000


def _wav(path, x):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def _click(rng, onset_s, dur=1.0, amp=0.2):
    x = rng.normal(0, 0.004, int(SR * dur)).astype(np.float32)
    k = round(onset_s * SR); m = min(5000, x.size - k); t = np.arange(m) / SR
    x[k:k + m] += (np.sin(2 * np.pi * 3000 * t) + 0.5 * rng.normal(size=m)) * np.minimum(1, t / 5e-4) * np.exp(-t * 300) * amp
    return x


def _make_root(root, onsets, hook_shift=0.0, dur=1.0, session="s0"):
    (root / "labels").mkdir(parents=True); (root / "audio").mkdir()
    rng = np.random.default_rng(1)
    with open(root / "labels" / f"{session}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["clip_id", "wav_onset_s", "onset_source", "jamo"])
        for i, o in enumerate(onsets):
            cid = f"{session}_{i:04d}"
            _wav(root / "audio" / f"{cid}.wav", _click(rng, o, dur))
            w.writerow([cid, o + hook_shift, "keystroke", "ㄱ"])


def test_param_hash_matches_classism():           # B1
    r = Regime()
    assert r.param_hash() == CLASSISM_PARAM_HASH == "915acee6"
    assert set(r.canonical_dict()) == {"sample_rate", "n_fft", "win_samples", "hop_samples", "n_mels",
                                       "fmin", "fmax", "clip_ms", "onset_pre_ms", "frames",
                                       "log_eps", "normalization", "dtype"}


def test_logmel_shape_zscore():
    x = np.random.default_rng(0).normal(size=4800).astype(np.float32)
    lm = MelExtractor().logmel(x)
    assert lm.shape == (64, 48) and lm.dtype == np.float32
    assert abs(lm.mean()) < 1e-4 and abs(lm.std() - 1) < 1e-4


def test_read_segment_matches_full_read(tmp_path):    # P3
    x = np.random.default_rng(0).uniform(-.5, .5, SR).astype(np.float32)
    _wav(tmp_path / "a.wav", x)
    seg, s0, n = read_segment(tmp_path / "a.wav", 1000, 500, SR)
    full, _, _ = read_segment(tmp_path / "a.wav", 0, SR, SR)
    assert s0 == 1000 and n == SR and np.array_equal(seg, full[1000:1500])
    seg, s0, _ = read_segment(tmp_path / "a.wav", -50, 100, SR)   # 파일 범위로 잘림
    assert s0 == 0 and seg.shape[0] == 50


def test_build_skip_idempotent_pack(tmp_path):
    _make_root(tmp_path, [0.5, 0.5, 0.002, 0.995])   # 뒤 2개는 100ms 창 불가
    r = Regime(); ext = MelExtractor(r)
    res = build_session(tmp_path / "labels" / "s0.csv", tmp_path, tmp_path, ext,
                        Options(pack=True))
    assert (res.rows, res.ok, len(res.skipped)) == (4, 2, 2)
    assert (tmp_path / "mel" / "_skipped_s0.csv").exists()
    a = np.load(tmp_path / "mel" / "s0_0000.npy")
    assert a.dtype == np.float32 and a.shape == (64, 48)
    z = np.load(tmp_path / "mel_packed" / "s0.npz")
    assert z["logmel"].shape == (2, 64, 48) and list(z["clip_id"]) == ["s0_0000", "s0_0001"]
    assert str(z["param_hash"]) == "915acee6"
    build_session(tmp_path / "labels" / "s0.csv", tmp_path, tmp_path, ext, Options(pack=True))
    assert np.array_equal(a, np.load(tmp_path / "mel" / "s0_0000.npy"))   # 멱등


def test_onset_detector_accuracy():                    # B3
    rng = np.random.default_rng(3); det = OnsetDetector(SR)
    errs = []
    for t0 in np.linspace(0.30, 0.36, 9):
        x = _click(rng, t0, 0.8, amp=0.1)
        d = det.detect(x)
        assert d.size == 1
        errs.append((d[0] - round(t0 * SR)) * 1000 / SR)
    assert np.abs(errs).max() < 2.0, errs


def test_min_gap_60ms():
    rng = np.random.default_rng(4); x = rng.normal(0, 0.004, SR).astype(np.float32)
    for tt in (0.2, 0.3, 0.34):
        k = round(tt * SR); t = np.arange(2400) / SR
        x[k:k + 2400] += np.sin(2 * np.pi * 3000 * t) * np.exp(-t * 500) * 0.3
    assert OnsetDetector(SR).detect(x).size == 2       # 0.34 는 0.30 에서 40ms → 무시


def test_realign_recovers_late_hook(tmp_path):         # 훅 시각이 8ms 늦어도 소리 시작점에서 자름
    _make_root(tmp_path, [0.5] * 3, hook_shift=0.008)
    ext = MelExtractor()
    res = build_session(tmp_path / "labels" / "s0.csv", tmp_path, tmp_path, ext,
                        Options(realign=True))
    assert res.ok == 3 and res.realigned == 3
    rows = list(csv.DictReader(open(tmp_path / "mel" / "_onset_s0.csv")))
    d = [float(r["delta_ms"]) for r in rows]
    assert all(-10.5 < v < -5.5 for v in d), d          # 훅 대비 소리가 약 8ms 앞


def test_int8_and_cli_hash_guard(tmp_path):
    q = quantize_int8(np.array([[-9., 0., 9.]], np.float32), 0.05, 0)
    assert q.dtype == np.int8 and q.tolist() == [[-128, 0, 127]]
    _make_root(tmp_path, [0.5])
    base = [sys.executable, "-m", "meligation", "--in", str(tmp_path)]
    root = str(Path(__file__).resolve().parents[1])
    ok = subprocess.run(base + ["--expect-hash", "915acee6"], cwd=root, capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert json.load(open(tmp_path / "mel" / "_meta.json"))["param_hash"] == "915acee6"
    bad = subprocess.run(base + ["--expect-hash", "deadbeef"], cwd=root, capture_output=True, text=True)
    assert bad.returncode == 4


if __name__ == "__main__":
    import tempfile
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            kw = {"tmp_path": Path(tempfile.mkdtemp())} if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount] else {}
            fn(**kw); print("PASS", name)
