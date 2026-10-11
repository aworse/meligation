"""로그멜 변환 — 오프라인(학습)과 온디바이스 레퍼런스가 공유하는 단일 구현. numpy 만 사용.

librosa 기본값과 같은 수식: Slaney 멜 스케일 + 면적 정규화, power=2.0, center=True(반사 패딩),
주기적 Hann(win_samples)을 n_fft 길이 가운데에 패딩.
"""
from __future__ import annotations

import numpy as np

from .regime import Regime


def _hz_to_mel(f):
    f = np.asarray(f, dtype=np.float64)
    f_sp = 200.0 / 3
    min_log_hz, logstep = 1000.0, np.log(6.4) / 27.0
    min_log_mel = min_log_hz / f_sp
    return np.where(f >= min_log_hz,
                    min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep,
                    f / f_sp)


def _mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    f_sp = 200.0 / 3
    min_log_hz, logstep = 1000.0, np.log(6.4) / 27.0
    min_log_mel = min_log_hz / f_sp
    return np.where(m >= min_log_mel,
                    min_log_hz * np.exp(logstep * (m - min_log_mel)),
                    f_sp * m)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """(n_mels, n_fft//2+1) 삼각 필터뱅크."""
    fft_freqs = np.linspace(0, sr / 2, 1 + n_fft // 2)
    hz_pts = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(hz_pts)
    ramps = hz_pts[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    w = np.maximum(0.0, np.minimum(lower, upper))
    w *= (2.0 / (hz_pts[2:n_mels + 2] - hz_pts[:n_mels]))[:, None]
    return w.astype(np.float32)


def fix_frames(a: np.ndarray, frames: int) -> np.ndarray:
    """시간축 고정: 길면 중앙 크롭, 짧으면 가장자리 패딩."""
    t = a.shape[1]
    if t > frames:
        s = (t - frames) // 2
        return a[:, s:s + frames]
    if t < frames:
        extra = frames - t
        left = extra // 2
        return np.pad(a, ((0, 0), (left, extra - left)), mode="edge")
    return a


class MelExtractor:
    """입력: 정확히 clip_samples(100 ms) 길이의 float32 [-1, 1] 창."""

    def __init__(self, regime: Regime = Regime()):
        self.r = regime
        r = regime
        n = np.arange(r.win_samples)
        win = 0.5 - 0.5 * np.cos(2 * np.pi * n / r.win_samples)
        self.window = np.zeros(r.n_fft, dtype=np.float32)
        pad = (r.n_fft - r.win_samples) // 2
        self.window[pad:pad + r.win_samples] = win
        self.fb = mel_filterbank(r.sample_rate, r.n_fft, r.n_mels, r.fmin, r.fmax)

    def logmel(self, clip: np.ndarray) -> np.ndarray:
        """(clip_samples,) → (n_mels, frames) float32, per-clip z-score 완료."""
        r = self.r
        if clip.shape[0] != r.clip_samples:
            raise ValueError(f"clip length {clip.shape[0]} != {r.clip_samples}")
        x = np.pad(clip.astype(np.float32), r.n_fft // 2, mode="reflect")
        n_frames = 1 + (x.shape[0] - r.n_fft) // r.hop_samples
        idx = np.arange(r.n_fft)[None, :] + r.hop_samples * np.arange(n_frames)[:, None]
        spec = np.fft.rfft(x[idx] * self.window[None, :], axis=1)
        power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
        lm = np.log(self.fb @ power.T + r.log_eps)
        lm = fix_frames(lm, r.frames)
        std = lm.std()
        lm = (lm - lm.mean()) / (std if std > 1e-8 else 1.0)
        return lm.astype(np.float32)


def quantize_int8(logmel: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """온디바이스 경로: AFE(deployment.json)가 준 input_scale/zero_point 로 int8 변환."""
    return np.clip(np.round(logmel / scale) + zero_point, -128, 127).astype(np.int8)
