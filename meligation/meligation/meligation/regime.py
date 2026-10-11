"""추출 규격(Regime) — classism src/melio.py::Regime 과 필드 이름·값·타입이 같아야 한다 (B1).

param_hash 는 canonical_dict() 를 키 정렬·공백 없는 JSON 으로 직렬화해 SHA-256 앞 8자를 취한다.
JSON 직렬화는 타입에 민감하다(200 vs 200.0, 1e-06) — 필드 타입을 바꾸면 해시가 달라진다.
classism 해시(915acee6)는 tests/test_contract.py 에서 고정 검증한다.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

MEL_SPEC_VERSION = "1.0"
CLASSISM_PARAM_HASH = "915acee6"  # classism c6606a0 기준


@dataclass(frozen=True)
class Regime:
    sample_rate: int = 48000
    n_fft: int = 1024
    win_samples: int = 384        # 8 ms @48k
    hop_samples: int = 96         # 2 ms @48k
    n_mels: int = 64
    fmin: float = 200.0
    fmax: float = 20000.0
    clip_ms: int = 100            # onset 앞 onset_pre_ms + 뒤 (clip_ms - onset_pre_ms)
    onset_pre_ms: int = 5
    frames: int = 48
    log_eps: float = 1e-6
    normalization: str = "per_clip_zscore"
    dtype: str = "float32"

    # ---- 파생값 (해시에 포함되지 않음: dataclass 필드가 아님) ----
    @property
    def pre_samples(self) -> int:
        return round(self.onset_pre_ms * self.sample_rate / 1000)

    @property
    def clip_samples(self) -> int:
        return round(self.clip_ms * self.sample_rate / 1000)          # 4800

    @property
    def post_samples(self) -> int:
        return self.clip_samples - self.pre_samples

    def canonical_dict(self) -> dict:
        return asdict(self)

    def param_hash(self) -> str:
        blob = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


def make_meta(regime: Regime) -> dict:
    return {
        "mel_spec_version": MEL_SPEC_VERSION,
        "param_hash": regime.param_hash(),
        "sample_rate": regime.sample_rate,
        "n_mels": regime.n_mels,
        "frames": regime.frames,
        "regime": regime.canonical_dict(),
    }
