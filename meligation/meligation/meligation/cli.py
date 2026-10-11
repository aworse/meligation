from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .build import Options, build_session, write_meta
from .melspec import MelExtractor
from .regime import Regime

EXIT_SKIP_RATIO = 3
EXIT_HASH_MISMATCH = 4


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mel-build",
                                 description="sound-autodaq 산출물 → classism용 로그멜")
    ap.add_argument("--in", dest="inp", required=True, type=Path, help="<root> (labels/, audio/)")
    ap.add_argument("--out", type=Path, default=None, help="<root> (mel/ 생성), 기본=--in")
    ap.add_argument("--session", help="이 세션 CSV 이름(확장자 제외)만 처리")
    ap.add_argument("--realign", action="store_true",
                    help="훅 시각 ±search-ms 안에서 소리 기반으로 시작점 재정렬 (B3)")
    ap.add_argument("--onset-report", action="store_true",
                    help="재정렬 없이 훅-소리 시각 차이만 mel/_onset_<session>.csv 에 기록")
    ap.add_argument("--search-ms", type=float, default=30.0)
    ap.add_argument("--pack", action="store_true", help="세션별 샤드 <pack-dir>/<session>.npz 도 생성")
    ap.add_argument("--pack-only", action="store_true", help="--pack + 클립별 .npy 는 쓰지 않음")
    ap.add_argument("--pack-dir", type=Path, default=None, help="기본 <out>/mel_packed")
    ap.add_argument("--expect-hash", default=None,
                    help="규격 param_hash 가 이 값과 다르면 시작 전에 중단 (예: classism 해시)")
    ap.add_argument("--max-skip-ratio", type=float, default=0.01,
                    help="건너뛴 비율이 넘으면 종료코드 3 (산출물은 그대로 기록). 기본 1%%")
    args = ap.parse_args(argv)

    regime = Regime()
    ph = regime.param_hash()
    if args.expect_hash and args.expect_hash != ph:
        print(f"param_hash 불일치: meligation={ph} expected={args.expect_hash}", file=sys.stderr)
        return EXIT_HASH_MISMATCH

    root_in, root_out = args.inp, args.out or args.inp
    csvs = sorted((root_in / "labels").glob("*.csv"))
    if args.session:
        csvs = [p for p in csvs if p.stem == args.session]
    if not csvs:
        print("labels/*.csv 없음", file=sys.stderr)
        return 2

    opt = Options(realign=args.realign, onset_report=args.onset_report, search_ms=args.search_ms,
                  write_npy=not args.pack_only, pack=args.pack or args.pack_only,
                  pack_dir=args.pack_dir)
    ext = MelExtractor(regime)
    write_meta(root_out, regime)

    total = ok = skipped = 0
    for p in csvs:
        res = build_session(p, root_in, root_out, ext, opt)
        total, ok, skipped = total + res.rows, ok + res.ok, skipped + len(res.skipped)
        extra = f", 재정렬 {res.realigned}" if args.realign else ""
        print(f"[{res.session}] 행 {res.rows} → 생성 {res.ok}, 건너뜀 {len(res.skipped)}{extra}")
        for cid, why in res.skipped:
            print(f"    SKIP {cid}: {why}")

    ratio = skipped / total if total else 0.0
    print(f"완료: 전체 {total}, 생성 {ok}, 건너뜀 {skipped} ({ratio:.2%}), param_hash={ph}")
    if ratio > args.max_skip_ratio:
        print(f"경고: 건너뜀 비율 {ratio:.2%} > {args.max_skip_ratio:.2%}", file=sys.stderr)
        return EXIT_SKIP_RATIO
    return 0


if __name__ == "__main__":
    sys.exit(main())
