# Well Economics Calculator

> Full-cycle E&P financial model: NPV, IRR, payback period, monthly cashflow,
> waterfall revenue breakdown, and tornado sensitivity analysis.

## Features

- **NPV** at user-defined discount rate (monthly discounting)
- **IRR** via Newton-Raphson iteration (annualized)
- **Payback period** from cumulative cashflow
- **Production forecast** using Arps hyperbolic decline
- **Revenue waterfall**: Gross → NRI → Severance Tax → OPEX → Net CF
- **Tornado chart**: NPV sensitivity to ±20% on 6 key parameters
- **Monthly cashflow table** with export to CSV
- Full **WI / NRI / Severance Tax** framework

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| WI | Working Interest — your % share of costs |
| NRI | Net Revenue Interest — your % share of revenue |
| Severance Tax | State production tax (e.g. Texas: 4.6%) |
| CAPEX | Drilling + completion cost |
| OPEX Fixed | Monthly fixed operating costs |
| OPEX Variable | Per-barrel lifting cost |

## Financial Methods

- **NPV**: `Σ CF_t / (1 + r_monthly)^t`
- **IRR**: Newton-Raphson root finding on monthly NPV = 0
- **Decline**: Arps hyperbolic `q(t) = qi / (1 + b·di·t)^(1/b)`

## Tech Stack

`Python` · `Streamlit` · `Plotly` · `NumPy` · `pandas` · `petropt`
