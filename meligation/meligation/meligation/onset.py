"""소리 기반 온셋 검출 (실사용 경로) + 학습 데이터 시작점 재정렬 (B3).

학습: 키보드 훅 시각(wav_onset_s)에서 자름. 실사용: 소리로 찾은 시작점에서 자름.
두 기준이 어긋나면 학습-배포 도메인 시프트가 생기므로, 재정렬은 *실사용과 같은 검출기*
(OnsetDetector)를 훅 시각 ±search_ms 안에서 돌려 시작점을 다시 잡는다.

검출 방식: 400~12000 Hz 대역 에너지 포락선(dB, 1 ms 간격) → 중앙값 잡음 바닥 대비
rise_db 이상 상승 지점 → 국소 피크의 frac 지점까지 역추적 → 해당 프레임 안에서 샘플 단위로
정밀화(|Δx| 가 잡음 σ·6 과 피크·0.1 중 큰 값을 처음 넘는 샘플), 최소 간격 60 ms.
※ 기존 classism/AFE src/segment.py 와 파라미터(대역, 최소간격)는 맞췄지만 구현은 별도다.
   실사용 검출기를 이 클래스로 통일하거나, 그쪽 구현으로 detect() 를 교체해야 B3 가 의미를 갖는다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OnsetConfig:
    band_lo: float = 400.0
    band_hi: float = 12000.0
    min_gap_ms: float = 60.0
    win_ms: float = 5.0
    hop_ms: float = 1.0
    rise_db: float = 10.0        # 잡음 바닥 대비 상승 폭
    frac: float = 0.25           # 시작점 = 피크 상승분의 frac 지점
    peak_ms: float = 20.0        # 피크 탐색 길이
    back_ms: float = 10.0        # 역추적 최대 길이
    bias_ms: float = 0.0         # 샘플 정밀화 실패 시 프레임→시각 보정(파일럿 실측 후 조정)
    sample_sigma: float = 6.0    # 샘플 정밀화: 잡음 σ 의 몇 배 이상
    sample_frac: float = 0.1     # 샘플 정밀화: 국소 피크 |Δx| 의 비율


class OnsetDetector:
    def __init__(self, sample_rate: int = 48000, cfg: OnsetConfig = OnsetConfig()):
        self.sr, self.cfg = sample_rate, cfg
        ms = sample_rate / 1000
        self.win = round(cfg.win_ms * ms)
        self.hop = round(cfg.hop_ms * ms)
        self.nfft = 1 << (self.win - 1).bit_length()
        self.window = np.hanning(self.win).astype(np.float32)
        f = np.fft.rfftfreq(self.nfft, 1 / sample_rate)
        self.mask = (f >= cfg.band_lo) & (f <= cfg.band_hi)

    def envelope_db(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] < self.win:
            return np.zeros(0, np.float32)
        n = 1 + (x.shape[0] - self.win) // self.hop
        idx = np.arange(self.win)[None, :] + self.hop * np.arange(n)[:, None]
        spec = np.fft.rfft(x[idx] * self.window[None, :], self.nfft, axis=1)
        p = (spec.real ** 2 + spec.imag ** 2)[:, self.mask].sum(axis=1)
        return (10 * np.log10(p + 1e-12)).astype(np.float32)

    def detect(self, x: np.ndarray) -> np.ndarray:
        """x 안의 온셋 위치(샘플 인덱스, x 기준)를 시간순으로 반환."""
        c = self.cfg
        env = self.envelope_db(x)
        if env.size == 0:
            return np.zeros(0, np.int64)
        ms = self.sr / 1000
        per_ms = lambda v: max(1, round(v * ms / self.hop))   # noqa: E731
        peak_n, back_n = per_ms(c.peak_ms), per_ms(c.back_ms)
        min_gap = c.min_gap_ms * ms

        dx = np.abs(np.diff(x.astype(np.float32)))             # 샘플 단위 정밀화용
        sigma = float(np.median(dx)) / 0.6745                  # 잡음 진폭 추정(로버스트)
        noise = float(np.median(env))
        above = env >= noise + c.rise_db
        rising = np.flatnonzero(above & ~np.concatenate(([False], above[:-1])))

        out: list[int] = []
        for k in rising:
            peak = float(env[k:k + peak_n].max())
            level = noise + c.frac * (peak - noise)
            lo = max(0, k - back_n)
            hit = np.flatnonzero(env[lo:k + peak_n] >= level)
            j = lo + int(hit[0]) if hit.size else int(k)
            # 프레임 단위(±win) 추정을 샘플 단위로 정밀화: 프레임 안에서 첫 급격한 변화
            a, b = j * self.hop, j * self.hop + self.win + self.hop
            peak_a = float(dx[a:(k + peak_n) * self.hop + self.win].max()) if a < dx.size else 0.0
            thr = max(c.sample_sigma * sigma, c.sample_frac * peak_a)
            h2 = np.flatnonzero(dx[a:b] >= thr)
            pos = int(a + h2[0]) if h2.size else int(a + round(c.bias_ms * ms))
            if out and pos - out[-1] < min_gap:
                continue
            out.append(pos)
        return np.asarray(out, dtype=np.int64)

    def refine(self, x: np.ndarray, hint: int, search_samples: int):
        """hint ±search 안의 가장 가까운 검출 온셋(x 기준 샘플) 또는 None."""
        d = self.detect(x)
        if d.size == 0:
            return None
        k = int(np.argmin(np.abs(d - hint)))
        return int(d[k]) if abs(int(d[k]) - hint) <= search_samples else None
