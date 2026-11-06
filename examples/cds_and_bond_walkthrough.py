"""End-to-end walkthrough: build curves, price a risky bond and a standard CDS, compute risk.

Run with:  python examples/cds_and_bond_walkthrough.py
Market data below is illustrative, not real quotes.
"""
from datetime import date

import creditlib as cl
from creditlib.dates import cds_standard_maturity
from creditlib.pricers import bond as bp
from creditlib.risk import cs01, ir01, jump_to_default, recovery01

today = date(2026, 11, 6)
recovery = 0.40

# 1. Discount curve from deposits and swaps
discount = cl.bootstrap_discount_curve(
    today,
    deposits=[("1M", 0.0390), ("3M", 0.0395), ("6M", 0.0400)],
    swaps=[("1Y", 0.0390), ("2Y", 0.0370), ("3Y", 0.0360), ("5Y", 0.0360), ("7Y", 0.0370), ("10Y", 0.0380)],
)

# 2. Survival curve from a CDS par-spread term structure
spreads = {1: 0.0050, 3: 0.0080, 5: 0.0120, 7: 0.0140, 10: 0.0160}
quotes = [cl.CDSQuote(cds_standard_maturity(today, n), par_spread=s) for n, s in spreads.items()]
survival = cl.bootstrap_survival_curve(discount, quotes, recovery)

print("Calibrated hazard rates")
for q, h in zip(quotes, survival.rates):
    print(f"  to {q.maturity}: {h:7.4%}   Q = {survival.q(q.maturity):.4f}")

# 3. A standard 5Y CDS, 100bp coupon, buying 10mm protection
cds = cl.CDS.standard(today, 5, coupon=0.01, notional=10_000_000)
pricer = cl.CDSPricer(discount, survival, recovery)
print(f"\n5Y CDS  maturity {cds.maturity}, accrual start {cds.effective_accrual_start}")
print(f"  par spread      {pricer.par_spread(cds) * 1e4:8.2f} bp")
print(f"  RPV01 (clean)   {pricer.rpv01(cds):8.4f}")
print(f"  points upfront  {pricer.upfront(cds):8.4%}")
print(f"  accrued         {pricer.accrued_premium(cds):12,.0f}")
print(f"  dirty MTM       {pricer.pv(cds):12,.0f}")
print(f"  CS01            {cs01(cds, discount, quotes, recovery):12,.0f}")
print(f"  IR01            {ir01(cds, discount, quotes, recovery):12,.0f}")
print(f"  Recovery01      {recovery01(cds, discount, quotes, recovery):12,.0f}")
print(f"  Jump-to-default {jump_to_default(cds, discount, quotes, recovery):12,.0f}")

# 4. A risky bond of the same issuer, priced off the CDS curve
bond = cl.FixedRateBond(issue_date=date(2025, 3, 15), maturity=date(2031, 3, 15), coupon=0.05)
settle = date(2026, 11, 10)
clean = bp.risky_clean_price(bond, settle, discount, survival, recovery)
ytm = bp.yield_from_clean_price(bond, settle, clean)
print(f"\n5% bond maturing {bond.maturity}, settling {settle}")
print(f"  CDS-implied clean price {clean:9.4f}")
print(f"  yield to maturity       {ytm:9.4%}")
print(f"  Z-spread                {bp.z_spread(bond, settle, clean, discount) * 1e4:9.2f} bp")
print(f"  modified duration       {bp.modified_duration(bond, settle, ytm):9.4f}")
print(f"  bond-implied flat hazard {bp.implied_hazard_rate(bond, settle, clean, discount, recovery):8.4%}")
