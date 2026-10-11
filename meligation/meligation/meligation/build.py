"""세션 단위 빌드: labels/<session>.csv + audio/<clip_id>.wav → mel/<clip_id>.npy (+ 선택: 세션 샤드 .npz)."""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .melspec import MelExtractor
from .onset import OnsetDetector
from .regime import Regime, make_meta
from .wavio import read_segment

CONTEXT_MS = 150      # 재정렬 시 검출기에 줄 주변 문맥(잡음 바닥 추정용)


@dataclass
class Options:
    realign: bool = False            # 훅 시각 ±search_ms 안에서 소리 기반으로 시작점 재정렬 (B3)
    onset_report: bool = False       # 재정렬하지 않아도 훅-소리 시각 차이를 기록 (파일럿 실측용)
    search_ms: float = 30.0
    write_npy: bool = True
    pack: bool = False               # 세션 샤드 <pack_dir>/<session>.npz
    pack_dir: Path | None = None


@dataclass
class SessionResult:
    session: str
    rows: int = 0
    ok: int = 0
    skipped: list = field(default_factory=list)   # [(clip_id, reason)]
    realigned: int = 0


def _atomic_write(path: Path, writer) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        writer(fh)
    os.replace(tmp, path)


def _cut(local: np.ndarray, offset_onset: int, r: Regime):
    """local 안에서 onset 기준 창을 자른다. 못 담으면 (None, 사유)."""
    s = offset_onset - r.pre_samples
    e = s + r.clip_samples
    if s < 0:
        return None, f"앞 부족(start={s})"
    if e > local.shape[0]:
        return None, f"뒤 부족(end={e} > len={local.shape[0]})"
    return local[s:e], None


def build_session(csv_path: Path, root_in: Path, root_out: Path, ext: MelExtractor,
                  opt: Options = Options(), detector: OnsetDetector | None = None) -> SessionResult:
    r = ext.r
    session = csv_path.stem
    mel_dir = root_out / "mel"
    mel_dir.mkdir(parents=True, exist_ok=True)
    res = SessionResult(session)
    use_detector = opt.realign or opt.onset_report
    if use_detector and detector is None:
        detector = OnsetDetector(r.sample_rate)
    search = round(opt.search_ms * r.sample_rate / 1000)
    ctx = round(CONTEXT_MS * r.sample_rate / 1000) if use_detector else 0

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    res.rows = len(rows)

    mels, ids, onset_rows = [], [], []
    for row in rows:
        cid = row["clip_id"]                               # 글자 그대로
        try:
            hint_s = float(row["wav_onset_s"])
        except (KeyError, ValueError, TypeError):
            res.skipped.append((cid, "wav_onset_s 없음/비정상"))
            continue
        wav_path = root_in / "audio" / f"{cid}.wav"
        if not wav_path.is_file():
            res.skipped.append((cid, "wav 파일 없음"))
            continue

        hint = round(hint_s * r.sample_rate)
        seg_start = hint - r.pre_samples - search - ctx
        seg_len = r.clip_samples + 2 * (search + ctx)
        try:
            local, s0, _ = read_segment(wav_path, seg_start, seg_len, r.sample_rate)
        except Exception as e:                              # noqa: BLE001
            res.skipped.append((cid, f"wav 로드 실패: {e}"))
            continue

        onset, status = hint, "hook"
        if use_detector:
            found = detector.refine(local, hint - s0, search)
            if found is None:
                status = "no_detect"
            else:
                status = "detected"
                onset_rows.append((cid, hint_s, (found + s0) / r.sample_rate,
                                   (found + s0 - hint) * 1000 / r.sample_rate, status))
                if opt.realign:
                    onset = found + s0
            if status == "no_detect":
                onset_rows.append((cid, hint_s, "", "", status))

        clip, why = _cut(local, onset - s0, r)
        if clip is None and onset != hint:                  # 재정렬 창이 안 맞으면 훅 시각으로 폴백
            clip, why = _cut(local, hint - s0, r)
            onset = hint
        if clip is None:
            res.skipped.append((cid, why))
            continue
        if opt.realign and onset != hint:
            res.realigned += 1

        lm = ext.logmel(clip)
        assert lm.shape == (r.n_mels, r.frames) and lm.dtype == np.float32
        if opt.write_npy:
            _atomic_write(mel_dir / f"{cid}.npy", lambda fh, a=lm: np.save(fh, a))
        mels.append(lm)
        ids.append(cid)
        res.ok += 1

    # 이번에 못 만든 clip 의 이전 산출물은 제거(오래된 결과가 남지 않게)
    if opt.write_npy:
        for cid, _ in res.skipped:
            (mel_dir / f"{cid}.npy").unlink(missing_ok=True)

    _write_skip_report(mel_dir / f"_skipped_{session}.csv", res.skipped)
    if use_detector:
        _write_onset_report(mel_dir / f"_onset_{session}.csv", onset_rows)

    if opt.pack:
        pack_dir = opt.pack_dir or (root_out / "mel_packed")
        pack_dir.mkdir(parents=True, exist_ok=True)
        stack = np.stack(mels).astype(np.float32) if mels else np.zeros((0, r.n_mels, r.frames), np.float32)
        _atomic_write(pack_dir / f"{session}.npz", lambda fh: np.savez(
            fh, logmel=stack, clip_id=np.asarray(ids, dtype=str),
            param_hash=np.asarray(r_hash(r))))
    return res


def r_hash(r: Regime) -> str:
    return r.param_hash()


def _write_skip_report(path: Path, skipped) -> None:
    if skipped:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["clip_id", "reason"])
            w.writerows(skipped)
    else:
        path.unlink(missing_ok=True)


def _write_onset_report(path: Path, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "hook_onset_s", "sound_onset_s", "delta_ms", "status"])
        w.writerows(rows)


def write_meta(root_out: Path, regime: Regime) -> None:
    mel_dir = root_out / "mel"
    mel_dir.mkdir(parents=True, exist_ok=True)
    data = json.dumps(make_meta(regime), ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    _atomic_write(mel_dir / "_meta.json", lambda fh: fh.write(data))
