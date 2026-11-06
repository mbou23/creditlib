# creditlib: notes for Claude Code

Credit derivatives pricing library (Python ≥ 3.9, numpy, scipy). The roadmap is in README.md.

## Commands
- Tests: `.venv/bin/pytest`
- Example: `.venv/bin/python examples/cds_and_bond_walkthrough.py`
- Install (editable): `.venv/bin/pip install -e ".[dev]"`

## Conventions
- Python 3.9 compatible: use `from __future__ import annotations`; no `match`, no runtime `X | Y` types.
- Keep layers separate: `instruments/` holds trade data only, `pricers/` holds models, `curves/` holds market data, `risk/` holds sensitivities.
- Curve time is ACT/365F from the curve reference date; curves accept a `date` or a float time.
- Values are for the protection buyer, scaled by notional (negative notional = seller).
- Every new model needs tests against a closed form, an analytic identity or a limiting case,
  not just regression numbers.
- Use only public methodology (ISDA, O'Kane, Hull, academic papers). No proprietary/employer code or data.
- Never commit secrets; `.env` is git-ignored.
