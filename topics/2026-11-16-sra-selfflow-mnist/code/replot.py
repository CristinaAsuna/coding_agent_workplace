#!/usr/bin/env python3
"""Regenerate planned figures from recorded metrics and saved samples, CPU only."""
import argparse
import json
from pathlib import Path

import run


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--samples', type=Path, required=True)
    p.add_argument('--figures', type=Path, required=True)
    args = p.parse_args()
    e1 = json.loads((args.results / 'E1.json').read_text())
    e2 = json.loads((args.results / 'E2.json').read_text())
    run.plot_e1(e1['metrics'], e1['baselines'], args.samples, args.figures)
    run.plot_e2(e2['metrics'], args.figures)


if __name__ == '__main__':
    main()
