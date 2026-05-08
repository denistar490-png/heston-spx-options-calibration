"""
Heston Model Calibration on SPX Options

Calibrating the Heston stochastic volatility model to real SPX option data
and comparing the fitted volatility surface against the market surface.

Important choices:
- The very short maturity T = 0.008 years is excluded from calibration
  because ultra-short-dated options are often very noisy and difficult for the
  standard constant-parameter Heston model to fit.
- Calibration is performed on implied volatility, not directly on prices.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution, brentq
from scipy.stats import norm
from scipy.interpolate import griddata


# SETTINGS

DATA_FILE = "options_clean.csv"
S0 = 7165.08
RISK_FREE_RATE = 0.0359
DIVIDEND_YIELD = 0.0

# Exclude ultra-short maturities.
# 0.02 years is about 7.3 calendar days.
MIN_MATURITY = 0.02

# Integration parameters.
INTEGRATION_UPPER_LIMIT = 100.0
INTEGRATION_STEPS = 300


# 1. BLACK-SCHOLES HELPERS

def black_scholes_price(S0, K, T, r, vol, option_type="call", q=0.0):
    """Black-Scholes price for a European call or put."""
    if T <= 0 or vol <= 0:
        return max(S0 - K, 0.0) if option_type == "call" else max(K - S0, 0.0)

    d1 = (np.log(S0 / K) + (r - q + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)

    if option_type == "call":
        return (
            S0 * np.exp(-q * T) * norm.cdf(d1)
            - K * np.exp(-r * T) * norm.cdf(d2)
        )

    return (
        K * np.exp(-r * T) * norm.cdf(-d2)
        - S0 * np.exp(-q * T) * norm.cdf(-d1)
    )


def implied_volatility(price, S0, K, T, r, option_type="call", q=0.0):
    """
    Invert Black-Scholes numerically to obtain implied volatility.
    Returns NaN if the price is not compatible with Black-Scholes bounds.
    """
    if not np.isfinite(price) or price <= 0 or T <= 0:
        return np.nan

    if option_type == "call":
        intrinsic = max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        upper = S0 * np.exp(-q * T)
    else:
        intrinsic = max(K * np.exp(-r * T) - S0 * np.exp(-q * T), 0.0)
        upper = K * np.exp(-r * T)

    if price <= intrinsic + 1e-10 or price >= upper:
        return np.nan

    def objective(vol):
        return black_scholes_price(S0, K, T, r, vol, option_type, q) - price

    try:
        return brentq(objective, 1e-4, 5.0, maxiter=100)
    except ValueError:
        return np.nan


# 2. HESTON CHARACTERISTIC FUNCTION

def heston_charfunc(phi, S0, v0, kappa, theta, sigma, rho, T, r, q=0.0):
    """
    Heston characteristic function of log(S_T) under the risk-neutral measure Q.

    Risk-neutral dynamics:
        dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW_t^S
        dv_t = kappa(theta - v_t) dt + sigma sqrt(v_t) dW_t^v
        corr(dW^S, dW^v) = rho
    """
    a = kappa * theta
    b = kappa
    i = 1j

    d = np.sqrt((rho * sigma * phi * i - b) ** 2 + sigma**2 * (phi * i + phi**2))
    g = (b - rho * sigma * phi * i + d) / (b - rho * sigma * phi * i - d)

    exp1 = np.exp(i * phi * (np.log(S0) + (r - q) * T))
    exp2 = ((1 - g * np.exp(d * T)) / (1 - g)) ** (-2 * a / sigma**2)
    exp3 = np.exp(
        a * T * (b - rho * sigma * phi * i + d) / sigma**2
        + v0
        * (b - rho * sigma * phi * i + d)
        * (1 - np.exp(d * T))
        / (sigma**2 * (1 - g * np.exp(d * T)))
    )

    return exp1 * exp2 * exp3


# 3. HESTON PRICER

def heston_call_price(
    S0,
    K,
    T,
    r,
    v0,
    kappa,
    theta,
    sigma,
    rho,
    q=0.0,
    umax=INTEGRATION_UPPER_LIMIT,
    N=INTEGRATION_STEPS,
):
    """
    European call price via midpoint integration of the Heston characteristic function.
    K and T may be scalars or numpy arrays.
    """
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)

    total = np.zeros_like(K, dtype=complex)
    dphi = umax / N

    for j in range(N):
        phi = (j + 0.5) * dphi

        cf1 = heston_charfunc(phi - 1j, S0, v0, kappa, theta, sigma, rho, T, r, q)
        cf2 = heston_charfunc(phi, S0, v0, kappa, theta, sigma, rho, T, r, q)

        numerator = np.exp((r - q) * T) * cf1 - K * cf2
        denominator = 1j * phi * K ** (1j * phi)

        total += dphi * numerator / denominator

    call = np.real((S0 * np.exp(-q * T) - K * np.exp(-r * T)) / 2 + total / np.pi)
    return np.maximum(call, 0.0)


def heston_option_price(S0, K, T, r, params, option_type="call", q=0.0):
    """Heston call or put price. Puts are obtained by put-call parity."""
    v0, kappa, theta, sigma, rho = params

    call = heston_call_price(S0, K, T, r, v0, kappa, theta, sigma, rho, q)

    if option_type == "call":
        return call

    put = call - S0 * np.exp(-q * np.asarray(T)) + np.asarray(K) * np.exp(-r * np.asarray(T))
    return np.maximum(put, 0.0)


# 4. NUMERICAL VALIDATION TESTS

def validate_heston_pricer():
    """
    Numerical checks before calibration:
    1. No-arbitrage bounds
    2. Put-call parity
    3. Integration convergence
    4. Implied volatility sanity check
    """
    S_test = 100.0
    K_test = 100.0
    T_test = 1.0
    r_test = 0.03
    q_test = 0.00

    params_test = [0.04, 2.00, 0.04, 0.30, -0.70]

    call_price = float(
        heston_option_price(
            S_test,
            K_test,
            T_test,
            r_test,
            params_test,
            option_type="call",
            q=q_test,
        )
    )

    put_price = float(
        heston_option_price(
            S_test,
            K_test,
            T_test,
            r_test,
            params_test,
            option_type="put",
            q=q_test,
        )
    )

    print(f"Heston call price: {call_price:.6f}")
    print(f"Heston put price:  {put_price:.6f}")

    call_lower = max(S_test * np.exp(-q_test * T_test) - K_test * np.exp(-r_test * T_test), 0.0)
    call_upper = S_test * np.exp(-q_test * T_test)

    put_lower = max(K_test * np.exp(-r_test * T_test) - S_test * np.exp(-q_test * T_test), 0.0)
    put_upper = K_test * np.exp(-r_test * T_test)

    assert call_lower <= call_price <= call_upper, "Call price violates no-arbitrage bounds"
    assert put_lower <= put_price <= put_upper, "Put price violates no-arbitrage bounds"
    print("No-arbitrage bounds test passed.")

    lhs = call_price - put_price
    rhs = S_test * np.exp(-q_test * T_test) - K_test * np.exp(-r_test * T_test)
    parity_error = abs(lhs - rhs)

    print(f"Put-call parity error: {parity_error:.10f}")
    assert parity_error < 1e-6, "Put-call parity error is too large"
    print("Put-call parity test passed.")

    price_N200 = float(heston_call_price(S_test, K_test, T_test, r_test, *params_test, q=q_test, N=200))
    price_N500 = float(heston_call_price(S_test, K_test, T_test, r_test, *params_test, q=q_test, N=500))
    price_N1000 = float(heston_call_price(S_test, K_test, T_test, r_test, *params_test, q=q_test, N=1000))

    print("\nIntegration convergence:")
    print(f"N = 200:  {price_N200:.6f}")
    print(f"N = 500:  {price_N500:.6f}")
    print(f"N = 1000: {price_N1000:.6f}")

    assert abs(price_N500 - price_N1000) < 0.05, "Integration does not seem stable enough"
    print("Integration convergence test passed.")

    iv_test = implied_volatility(call_price, S_test, K_test, T_test, r_test, "call", q_test)
    print(f"Implied volatility from test Heston price: {iv_test:.6f}")

    assert np.isfinite(iv_test), "Implied volatility is NaN or infinite"
    assert 0.01 < iv_test < 3.0, "Implied volatility is outside a reasonable range"

    print("Implied volatility test passed.")
    print("All validation tests passed.\n")


# 5. DATA PREPARATION

def load_calibration_data(path=DATA_FILE, option_type="put"):
    """
    Load cleaned option data and filter for calibration.

    We use puts because SPX put options usually capture the left skew more clearly.
    We exclude ultra-short maturity options with T < MIN_MATURITY because the
    standard Heston model often struggles with very short-dated market noise.
    """
    df = pd.read_csv(path)

    df = df.dropna(subset=["type", "strike", "T", "mid", "IV"])
    df = df[(df["T"] >= MIN_MATURITY) & (df["mid"] > 0) & (df["IV"] > 0)]

    df = df[df["type"].str.lower() == option_type].copy()
    df["moneyness"] = df["strike"] / S0

    df = df[(df["moneyness"] >= 0.80) & (df["moneyness"] <= 1.20)]
    df = df.sort_values(["T", "strike"]).reset_index(drop=True)

    print(f"Calibration data: {len(df)} {option_type} options")
    print(f"Excluded maturities below: T < {MIN_MATURITY}")
    print(f"Moneyness range:  {df['moneyness'].min():.3f} — {df['moneyness'].max():.3f}")
    print(f"Maturities (yrs): {np.round(np.sort(df['T'].unique()), 4)}")

    return df


# 6. CALIBRATION ON IMPLIED VOLATILITY

def calibrate_heston(df, r=RISK_FREE_RATE, q=DIVIDEND_YIELD, option_type="put"):
    """
    Calibrate Heston parameters by minimising the squared error between:
        market implied volatility
        Heston implied volatility
    """
    K = df["strike"].to_numpy(float)
    T = df["T"].to_numpy(float)
    market_iv = df["IV"].to_numpy(float)
    moneyness = K / S0

    bounds = [
        (1e-4, 0.50),    # v0
        (0.05, 10.0),    # kappa
        (1e-4, 0.50),    # theta
        (1e-3, 3.00),    # sigma
        (-0.95, -0.20),  # rho
    ]

    def objective(x):
        v0, kappa, theta, sigma, rho = x

        if v0 <= 0 or theta <= 0 or kappa <= 0 or sigma <= 0 or not (-1 < rho < 1):
            return 1e10

        try:
            model_price = heston_option_price(S0, K, T, r, x, option_type, q)

            model_iv = np.array(
                [
                    implied_volatility(price, S0, strike, tau, r, option_type, q)
                    for price, strike, tau in zip(model_price, K, T)
                ]
            )

            valid = np.isfinite(model_iv) & np.isfinite(market_iv)

            if valid.sum() < 0.80 * len(K):
                return 1e8 + (len(K) - valid.sum()) * 1e5

            weights = 1.0 / (np.abs(moneyness[valid] - 1.0) + 0.05)
            error = np.mean(weights * (model_iv[valid] - market_iv[valid]) ** 2)

            if not np.isfinite(error):
                return 1e10

            return float(error)

        except Exception:
            return 1e10

    print("\nCalibration in progress: minimising implied-volatility error...")
    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=42,
        maxiter=50,
        popsize=6,
        tol=1e-6,
        disp=True,
        workers=1,
        mutation=(0.5, 1.5),
        recombination=0.7,
        polish=True,
    )

    v0, kappa, theta, sigma, rho = result.x
    feller = 2 * kappa * theta - sigma**2

    print("\n── Calibrated parameters ──")
    print(f"v0 = {v0:.6f}  -> initial vol   = {np.sqrt(v0) * 100:.2f}%")
    print(f"kappa = {kappa:.6f}")
    print(f"theta = {theta:.6f}  -> long-run vol  = {np.sqrt(theta) * 100:.2f}%")
    print(f"sigma = {sigma:.6f}")
    print(f"rho = {rho:.6f}")
    print(f"Feller: 2κθ − σ² = {feller:.6f} ({'satisfied' if feller >= 0 else 'violated'})")
    print(f"Objective value: {result.fun:.8f}")

    with open("heston_params.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "v0": float(v0),
                "kappa": float(kappa),
                "theta": float(theta),
                "sigma": float(sigma),
                "rho": float(rho),
                "objective": float(result.fun),
                "feller": float(feller),
                "min_maturity": float(MIN_MATURITY),
            },
            f,
            indent=4,
        )

    print("Parameters saved to heston_params.json")
    return result.x


# 7. HESTON IMPLIED VOL SURFACE

def heston_iv_surface(S0, moneyness_grid, T_grid, r, params, option_type="put", q=0.0):
    """Create the Heston implied-volatility surface on a regular grid."""
    iv_mesh = np.full((len(T_grid), len(moneyness_grid)), np.nan)
    K_grid = moneyness_grid * S0

    for i, T in enumerate(T_grid):
        prices = heston_option_price(S0, K_grid, T, r, params, option_type, q)
        ivs = [
            implied_volatility(p, S0, K, T, r, option_type, q)
            for p, K in zip(prices, K_grid)
        ]
        iv_mesh[i, :] = ivs

    return iv_mesh


# 8. PLOTS

def plot_smiles(df, params, r=RISK_FREE_RATE, q=DIVIDEND_YIELD, option_type="put"):
    maturities = np.sort(df["T"].unique())

    fig, axes = plt.subplots(
        1,
        len(maturities),
        figsize=(4.5 * len(maturities), 4.5),
        squeeze=False,
    )

    for ax, T in zip(axes.ravel(), maturities):
        subset = df[df["T"] == T].sort_values("moneyness")

        K_vals = subset["strike"].to_numpy(float)
        prices = heston_option_price(S0, K_vals, T, r, params, option_type, q)

        iv_heston = np.array(
            [
                implied_volatility(p, S0, K, T, r, option_type, q)
                for p, K in zip(prices, K_vals)
            ]
        )

        ax.plot(subset["moneyness"], subset["IV"], "o-", markersize=3, label="Market IV")
        ax.plot(subset["moneyness"], iv_heston, "s--", markersize=3, label="Heston IV")
        ax.set_title(f"T = {T:.3f} years")
        ax.set_xlabel("Moneyness K/S₀")
        ax.set_ylabel("Implied volatility")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.suptitle("Market Implied Volatility vs Calibrated Heston Implied Volatility", fontsize=13)
    plt.tight_layout()
    plt.savefig("heston_smile_fit.png", dpi=150)
    plt.show()
    print("Saved heston_smile_fit.png")


def plot_vol_surface(df, params, r=RISK_FREE_RATE, q=DIVIDEND_YIELD, option_type="put"):
    data = df[df["type"].str.lower() == option_type].dropna(subset=["IV"]).copy()
    data["moneyness"] = data["strike"] / S0

    m_grid = np.linspace(data["moneyness"].min(), data["moneyness"].max(), 40)
    t_grid = np.linspace(data["T"].min(), data["T"].max(), 40)
    M, TT = np.meshgrid(m_grid, t_grid)

    market_mesh = griddata(
        points=(data["moneyness"], data["T"]),
        values=data["IV"],
        xi=(M, TT),
        method="linear",
    )

    print("\nComputing Heston vol surface")
    heston_mesh = heston_iv_surface(S0, m_grid, t_grid, r, params, option_type, q)

    error_mesh = np.abs(heston_mesh - market_mesh)
    valid = np.isfinite(market_mesh) & np.isfinite(heston_mesh)

    rmse = np.sqrt(np.nanmean(error_mesh[valid] ** 2))
    print(f"Vol surface RMSE: {rmse * 100:.2f}%")

    fig = plt.figure(figsize=(17, 6))

    ax1 = fig.add_subplot(131, projection="3d")
    surf1 = ax1.plot_surface(M, TT, market_mesh, cmap="Blues", alpha=0.85, edgecolor="none")
    ax1.set_title("Market Vol Surface", fontsize=11)
    ax1.set_xlabel("Moneyness K/S₀")
    ax1.set_ylabel("Maturity (yrs)")
    ax1.set_zlabel("IV")
    fig.colorbar(surf1, ax=ax1, shrink=0.4)

    ax2 = fig.add_subplot(132, projection="3d")
    surf2 = ax2.plot_surface(M, TT, heston_mesh, cmap="Oranges", alpha=0.85, edgecolor="none")
    ax2.set_title("Heston Vol Surface", fontsize=11)
    ax2.set_xlabel("Moneyness K/S₀")
    ax2.set_ylabel("Maturity (yrs)")
    ax2.set_zlabel("IV")
    fig.colorbar(surf2, ax=ax2, shrink=0.4)

    ax3 = fig.add_subplot(133, projection="3d")
    surf3 = ax3.plot_surface(M, TT, error_mesh, cmap="RdYlGn_r", alpha=0.85, edgecolor="none")
    ax3.set_title(f"Absolute Error  (RMSE = {rmse * 100:.2f}%)", fontsize=11)
    ax3.set_xlabel("Moneyness K/S₀")
    ax3.set_ylabel("Maturity (yrs)")
    ax3.set_zlabel("|IV error|")
    fig.colorbar(surf3, ax=ax3, shrink=0.4)

    plt.suptitle("Volatility Surface: Market vs Heston, excluding T < 0.02", fontsize=13)
    plt.tight_layout()
    plt.savefig("vol_surface_comparison.png", dpi=150)
    plt.show()
    print("Saved vol_surface_comparison.png")


# MAIN

if __name__ == "__main__":
    validate_heston_pricer()

    df = load_calibration_data(option_type="put")

    params = calibrate_heston(df, option_type="put")

    plot_smiles(df, params, option_type="put")

    plot_vol_surface(df, params, option_type="put")

    print("\nDone. Output files:")
    print("  heston_params.json")
    print("  heston_smile_fit.png")
    print("  vol_surface_comparison.png")
