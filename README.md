# creditlib

A credit derivatives pricing library in Python, built up in stages from single-name
instruments to structured credit:

**bonds → CDS → credit-linked notes → repacks → CDO tranches**

The core models are written from scratch, no QuantLib dependency, with exact
piecewise integration and a test suite that checks results against closed-form
results and analytic identities.

## Status

| Phase | Scope | Status |
|---|---|---|
| 0. Foundations | Day counts (ACT/360, ACT/365F, 30/360, 30E/360, ACT/ACT ISDA), business-day calendars and conventions, bond schedules, standard CDS IMM dates and semi-annual roll | ✅ v0.1 |
| 1. Curves | Piecewise-flat discount and survival curves; discount bootstrap from deposits and swaps | ✅ v0.1 |
| 2. Bonds | Price/yield, accrued, duration, convexity, DV01, Z-spread; risky bond under recovery of par; bond-implied hazard rate | ✅ v0.1 |
| 3. CDS | ISDA-style premium leg (with accrual on default) and protection leg; par spread, upfront, clean/dirty MTM; hazard bootstrap from par spreads or upfront quotes; quoted spread ↔ upfront; CS01, IR01, Recovery01, jump-to-default | ✅ v0.1 |
| 4. CLN | Funded single-name CLN, then issuer and reference two-name default risk | 🔜 next |
| 5. Repacks | SPV with collateral bond + asset swap (par/par and market-value ASW) + embedded CDS; swap counterparty risk | planned |
| 6. Portfolio / CDO | One-factor Gaussian copula, large-pool approximation (Vasicek), exact finite-pool loss recursion, tranche pricing, base correlation, nth-to-default | planned |
| Performance | C++ core for the portfolio loss engine (pybind11) | planned |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python examples/cds_and_bond_walkthrough.py
```

```python
from datetime import date
import creditlib as cl
from creditlib.dates import cds_standard_maturity

today = date(2026, 11, 6)
discount = cl.bootstrap_discount_curve(
    today, deposits=[("3M", 0.0395)], swaps=[("2Y", 0.037), ("5Y", 0.036), ("10Y", 0.038)]
)
quotes = [cl.CDSQuote(cds_standard_maturity(today, n), par_spread=s)
          for n, s in [(1, 0.005), (3, 0.008), (5, 0.012), (10, 0.016)]]
survival = cl.bootstrap_survival_curve(discount, quotes, recovery=0.4)

cds = cl.CDS.standard(today, 5, coupon=0.01, notional=10_000_000)
pricer = cl.CDSPricer(discount, survival, recovery=0.4)
pricer.par_spread(cds), pricer.upfront(cds), pricer.pv(cds)
```

## Design

```
src/creditlib/
  dates/        day counts, calendars, schedules, CDS IMM dates
  curves/       DiscountCurve, SurvivalCurve (shared piecewise-constant-rate base), bootstrap
  instruments/  trade descriptions only (FixedRateBond, CDS); no pricing logic
  pricers/      pricing (bond analytics, CDSPricer), exact default-leg integrals
  risk/         bump-and-rebuild sensitivities
```

* **Instruments, market data and models are kept separate.** A `CDS` is a plain
  description of the trade; curves hold market data; pricers combine the two.
  New products (CLN, repack, tranche) reuse the same curves and integrals.
* **Exact integration.** With flat forwards and flat hazards on every interval,
  the protection leg and accrual-on-default integrals have closed forms (as in the
  ISDA CDS Standard Model). They are evaluated stably for small rates and checked
  against brute-force quadrature in the tests.
* **Risk by re-calibration.** CS01 and Recovery01 bump the market quotes and
  re-bootstrap the curve, as a trading desk would, rather than bumping hazard rates.

## Model conventions and current simplifications

* Curve time is ACT/365F from the valuation date; CDS premium accrues ACT/360.
* Valuation date = trade date `T`; protection and accrual start at step-in `T+1`;
  the T+3 cash-settlement lag is not yet modelled.
* Single-curve discounting (no OIS/IBOR basis); instruments start at the valuation date (no spot lag).
* The risky bond uses recovery of par, conditional on survival to settlement.
* The CDS pricer follows ISDA Standard Model conventions but has not yet been
  reconciled number-for-number against the ISDA reference implementation (planned).

## References

* D. O'Kane, *Modelling Single-name and Multi-name Credit Derivatives*, Wiley (2008)
* ISDA CDS Standard Model, and the 2009 "Big Bang" and 2015 semi-annual roll conventions
* J. Hull, *Options, Futures, and Other Derivatives*: credit chapters
* L. Andersen, J. Sidenius, S. Basu, "All your hedges in one basket", *Risk* (2003), for the planned CDO engine

## License

MIT
