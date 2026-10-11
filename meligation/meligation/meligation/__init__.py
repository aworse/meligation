"""meligation — sound-autodaq ↔ classism 로그멜 가교 (v1.0)."""
from .build import Options, SessionResult, build_session, write_meta
from .melspec import MelExtractor, quantize_int8
from .onset import OnsetConfig, OnsetDetector
from .regime import CLASSISM_PARAM_HASH, MEL_SPEC_VERSION, Regime, make_meta

__version__ = "1.0.0"
__all__ = ["Regime", "MelExtractor", "quantize_int8", "OnsetDetector", "OnsetConfig",
           "Options", "SessionResult", "build_session", "write_meta", "make_meta",
           "MEL_SPEC_VERSION", "CLASSISM_PARAM_HASH"]
