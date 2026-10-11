#!/usr/bin/env python3
"""호환용 진입점: python mel_build.py --in <root> --out <root>  (= python -m meligation ...)"""
import sys

from meligation.cli import main

if __name__ == "__main__":
    sys.exit(main())
