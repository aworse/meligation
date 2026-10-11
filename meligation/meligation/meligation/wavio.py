"""WAV 입출력. 필요한 구간만 읽는다(wave.setpos) — 1 s WAV 전체를 읽지 않는다 (P3)."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def wav_nframes(path: Path, expect_sr: int) -> int:
    with wave.open(str(path), "rb") as w:
        _check(w, expect_sr)
        return w.getnframes()


def _check(w: wave.Wave_read, expect_sr: int) -> None:
    if w.getsampwidth() != 2:
        raise ValueError(f"PCM16 아님(sampwidth={w.getsampwidth()})")
    if w.getframerate() != expect_sr:
        raise ValueError(f"샘플레이트 {w.getframerate()} != {expect_sr}")


def read_segment(path: Path, start: int, n: int, expect_sr: int):
    """[start, start+n) 를 파일 범위로 잘라 float32 로 읽는다.

    반환: (samples, s0, nframes)  — s0 는 실제로 읽은 첫 샘플의 파일 내 위치.
    """
    with wave.open(str(path), "rb") as w:
        _check(w, expect_sr)
        nframes, ch = w.getnframes(), w.getnchannels()
        s0 = max(0, start)
        s1 = min(nframes, start + n)
        if s1 <= s0:
            return np.zeros(0, np.float32), s0, nframes
        w.setpos(s0)
        raw = w.readframes(s1 - s0)
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, s0, nframes
