"""
Heston Model Calibration on SPX Options

Calibrating the Heston stochastic volatility model to real SPX option data
and comparing the fitted volatility surface against the market surface.

Important choices:
- The very short maturity T = 0.008 years is excluded from calibration
because ultra-short-dated options are often very noisy and difficult for the
standard constant-parameter Heston model to fit.
- Calibration is performed on implied volatility, not directly on prices.

Structure:
1. Load cleaned option data (options_clean.csv)
2. Black-Scholes helper functions
3. Heston characteristic function and semi-analytical pricer
4. Numerical validation tests
5. Calibration via differential_evolution
6. Volatility smile plots
7. Volatility surface comparison
"""

import json #only for saving the parametrized parameters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution, brentq #global optimizer, we use brentq to find the implied volatility
from scipy.stats import norm
from scipy.interpolate import griddata #for drawing the volatility surface in 3D


# SETTINGS


DATA_FILE = "options_clean.csv"
S0 = 7165.08
RISK_FREE_RATE = 0.0359
DIVIDEND_YIELD = 0.0 #for simplicity

# Exclude ultra-short maturities.
# 0.02 years is about 7.3 calendar days.
MIN_MATURITY = 0.02

# Integration parameters.
#n the Heston model, the option price is not calculated using a simple formula
# like Black-Scholes. It is calculated through an integral.
INTEGRATION_UPPER_LIMIT = 100.0 #we integrate from 0 to 100
INTEGRATION_STEPS = 300 #we are discretizing the integral in 300 subintervals that we will sum
#The integral is theoretically defined over [0,∞). In the implementation it is truncated at 100. This introduces a truncation error, but numerical convergence checks showed that increasing the integration range further had negligible impact on option prices

# 1. BLACK-SCHOLES HELPERS
#The inputs are:
#S0: current price of the underlying asset;
#K: strike price;
#T: time to maturity;
#r: risk-free rate;
#vol: volatility;
#option_type: "call" or "put";
#q: dividend yield.
def black_scholes_price(S0, K, T, r, vol, option_type="call", q=0.0):
    """Black-Scholes price for a European call or put."""
    if T <= 0 or vol <= 0: #If the time to maturity is zero or the volatility is zero, we cannot use the standard Black-Scholes formula because we would have divisions by zero. Therefore, we return the intrinsic value of the option.

        return max(S0 - K, 0.0) if option_type == "call" else max(K - S0, 0.0)

    d1 = (np.log(S0 / K) + (r - q + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)

    if option_type == "call": #Call price under Black-Scholes
        return S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)#put price under Black-Scholes


def implied_volatility(price, S0, K, T, r, option_type="call", q=0.0): #'price' is simply the option price that I want to convert into IV.
#we use this function to transform prices in implied volatility
    """
    Invert Black-Scholes numerically to obtain implied volatility.
    Returns NaN if the price is not compatible with Black-Scholes bounds.
    """
    if not np.isfinite(price) or price <= 0 or T <= 0: #If the price is not valid, or it is negative/zero, or the maturity is not valid, it returns NaN.

        return np.nan
#lower bound call = max(S0 e^{-qT} - K e^{-rT}, 0)
#upper bound call= S0 e^{-qT}
#lower bound put= max(K e^{-rT} - S0 e^{-qT}, 0)
#upper bound put= K e^{-rT}
    if option_type == "call":
        intrinsic = max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        upper = S0 * np.exp(-q * T)
    else:
        intrinsic = max(K * np.exp(-r * T) - S0 * np.exp(-q * T), 0.0)
        upper = K * np.exp(-r * T)
#If the price is outside the theoretical no-arbitrage bounds,
# I do not even try to calculate the implied volatility,
# because that price is not financially valid or cannot be meaningfully inverted using Black-Scholes.

    if price <= intrinsic + 1e-10 or price >= upper: #i add 1e-10 because it is used to avoid rounding issues and numerically unstable cases when the price is too close to the option’s theoretical minimum.

        return np.nan

    def objective(vol):
        return black_scholes_price(S0, K, T, r, vol, option_type, q) - price #we want to find the vol such as black_scholes_price(vol) - price = 0, so that black_scholes_price(vol) = price
#I pass the objective function to `brentq`;
# then `brentq` itself will try different volatility values between 0.0001 and 5.0 until it finds the one that makes `objective(vol) = 0`.

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
 #The inputs are:
#phi: complex/real integration variable;
#S0: initial price;
#v0: initial variance;
#kappa: speed of mean reversion;
#theta: long-term variance;
#sigma: volatility of volatility;
#rho: correlation;
#T: maturity;
#r: risk-free rate;
#q: dividend yield.

    a = kappa * theta
    b = kappa
    i = 1j #imaginary unit in Python

    d = np.sqrt((rho * sigma * phi * i - b)**2 + sigma**2 * (phi * i + phi**2))
    g = (b - rho * sigma * phi * i + d) / (b - rho * sigma * phi * i - d)

    exp1 = np.exp(i * phi * (np.log(S0) + (r - q) * T))
    exp2 = ((1 - g * np.exp(d * T)) / (1 - g))**(-2 * a / sigma**2)
    exp3 = np.exp(
        a * T * (b - rho * sigma * phi * i + d) / sigma**2
        + v0 * (b - rho * sigma * phi * i + d) * (1 - np.exp(d * T))
        / (sigma**2 * (1 - g * np.exp(d * T)))
    )

    return exp1 * exp2 * exp3 #characteristic function

# 3. HESTON PRICER

def heston_call_price(S0, K, T, r, v0, kappa, theta, sigma, rho, q=0.0,
                      umax=INTEGRATION_UPPER_LIMIT, N=INTEGRATION_STEPS):# we are going to calculate the price of the call
    """
    European call price via rectangular/midpoint integration of the Heston
    characteristic function. K and T may be scalars or numpy arrays.
    """
    K = np.asarray(K, dtype=float)#in order to have a more efficient code we convert K and T in array numpy, in this way the function can calculate multiple strikes/maturities together, not just one at a time
    T = np.asarray(T, dtype=float)

    total = np.zeros_like(K, dtype=complex)#Create the empty container where we accumulate, for each strike, the numerical sum of the Heston integral(the sum of each rectangular area). We use `complex` because the integral uses complex numbers, even though the final option price will be real.

    dphi = umax / N #lenght of each subinterval of integration

    # midpoint rule: j starts from 0.
    for j in range(N):
        phi = (j + 0.5) * dphi

        cf1 = heston_charfunc(phi - 1j, S0, v0, kappa, theta, sigma, rho, T, r, q)
        cf2 = heston_charfunc(phi,      S0, v0, kappa, theta, sigma, rho, T, r, q)

        numerator = np.exp((r - q) * T) * cf1 - K * cf2
        denominator = 1j * phi * K**(1j * phi)

        total += dphi * numerator / denominator

    call = np.real((S0 * np.exp(-q * T) - K * np.exp(-r * T)) / 2 + total / np.pi)
    return np.maximum(call, 0.0)


def heston_option_price(S0, K, T, r, params, option_type="call", q=0.0): #`params` is the vector of calibrated Heston parameters.
#"call" is just the default value.
#That is: if you do not specify option_type when calling the function, Python automatically uses "call".
    """Heston call or put price. Puts are obtained by put-call parity."""
    v0, kappa, theta, sigma, rho = params #Take the 5 values contained in `params` and assign them, in order, to 5 different variables.

    call = heston_call_price(S0, K, T, r, v0, kappa, theta, sigma, rho, q)

    if option_type == "call":
        return call

    put = call - S0 * np.exp(-q * np.asarray(T)) + np.asarray(K) * np.exp(-r * np.asarray(T))#put-call parity, P = C - S0 e^{-qT} + K e^{-rT}
    return np.maximum(put, 0.0)
#The code always computes the Heston call price first. If the requested option is a call, it returns the call price. If instead the requested option is a put, it uses the call price just computed and transforms it into a put price using put-call parity.


# 4. NUMERICAL VALIDATION TESTS

def validate_heston_pricer():
    """
    Numerical checks before calibration:
    1. No-arbitrage bounds
    2. Put-call parity
    3. Integration convergence
    4. Implied volatility sanity check
    """
#Here we are testing an ATM call/put with:
#underlying price 100;
#strike 100;
#maturity 1 year;
#interest rate 3%;
#initial variance 0.04, meaning initial volatility 20%;
#kappa 2;
#theta 0.04;
#sigma 0.30;
#rho -0.70.

    S_test = 100.0
    K_test = 100.0
    T_test = 1.0
    r_test = 0.03
    q_test = 0.00

    params_test = [0.04, 2.00, 0.04, 0.30, -0.70]

    call_price = float(heston_option_price(
        S_test, K_test, T_test, r_test, params_test, option_type="call", q=q_test
    ))

    put_price = float(heston_option_price(
        S_test, K_test, T_test, r_test, params_test, option_type="put", q=q_test
    ))

    print(f"Heston call price: {call_price:.6f}")
    print(f"Heston put price:  {put_price:.6f}")
#lower and upper bounds call and put calculated with heston
    call_lower = max(S_test * np.exp(-q_test * T_test) - K_test * np.exp(-r_test * T_test), 0.0)
    call_upper = S_test * np.exp(-q_test * T_test)

    put_lower = max(K_test * np.exp(-r_test * T_test) - S_test * np.exp(-q_test * T_test), 0.0)
    put_upper = K_test * np.exp(-r_test * T_test)

    assert call_lower <= call_price <= call_upper, "Call price violates no-arbitrage bounds"
    assert put_lower <= put_price <= put_upper, "Put price violates no-arbitrage bounds"
    print("No-arbitrage bounds test passed.")
#test put-call parity
    lhs = call_price - put_price
    rhs = S_test * np.exp(-q_test * T_test) - K_test * np.exp(-r_test * T_test)
    parity_error = abs(lhs - rhs)

    print(f"Put-call parity error: {parity_error:.10f}")
    assert parity_error < 1e-6, "Put-call parity error is too large"
    print("Put-call parity test passed.")
#Compute the same price using 200, 500, and 1000 integration steps. If the price changes very little between 500 and 1000, then the integration is stable.

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
#The IV must be computable and within a reasonable range, 3 is 300%.

    print("Implied volatility test passed.")
    print("All validation tests passed.\n")



# 5. DATA PREPARATION


def load_calibration_data(path=DATA_FILE, option_type="put"):#in this way we could use also other csv files
    """
    Load cleaned option data and filter for calibration.

    We use puts because SPX put options usually capture the left skew more clearly.
    We exclude ultra-short maturity options with T < MIN_MATURITY because the
    standard Heston model often struggles with very short-dated market noise.
    """
    df = pd.read_csv(path)

    df = df.dropna(subset=["type", "strike", "T", "mid", "IV"])#Remove rows with missing data in the key columns.
    df = df[(df["T"] >= MIN_MATURITY) & (df["mid"] > 0) & (df["IV"] > 0)]
#Keep only options with:
#maturity of at least 0.02;
#positive mid_price;
#positive IV.

    df = df[df["type"].str.lower() == option_type].copy()#Keep only the puts, because option_type="put"
    df["moneyness"] = df["strike"] / S0#If `K/S0 < 1`, for a put it means it is OTM. For example, `0.85` means the strike is at 85% of the current index level.


    # Keep a reasonable smile region.
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

    This is more suitable than pure price-error calibration when the goal is to
    fit the volatility smile/surface.
    """
#We extract the strike, maturity, and implied volatility.
    K = df["strike"].to_numpy(float)#with to_numpy we transform the pandas series in NumPy arrays
    T = df["T"].to_numpy(float)
    market_iv = df["IV"].to_numpy(float)
    moneyness = K / S0
#let's define the bounds on the parameters
    bounds = [
        (1e-4, 0.50),    # v0
        (0.05, 10.0),    # kappa
        (1e-4, 0.50),    # theta
        (1e-3, 3.00),    # sigma
        (-0.95, -0.20),  # rho: typically negative for equity index options because usually, when the market goes down, volatility goes up.
    ]
#The parameter `ρ` measures the correlation between these two shocks: dWs(shock of the price) and dWv (shock of the volatility)

    def objective(x): #We do not pass `x` to `calibrate_heston`, because `x` is exactly what we want to find.It is exactly like with `brentq` before: we did not pass `vol`, because `vol` was the unknown that `brentq` was searching for.
        v0, kappa, theta, sigma, rho = x

        if v0 <= 0 or theta <= 0 or kappa <= 0 or sigma <= 0 or not (-1 < rho < 1):#If the parameters are not valid, it returns a huge error. This tells the optimizer that that region is not good.
            return 1e10

        try:#The problem is that some parameters tested by the optimizer may generate numerical errors. Without `try/except`, at the first error the calibration would stop completely.
            model_price = heston_option_price(S0, K, T, r, x, option_type, q)#array of prices
#`model_iv` is a vector/array containing all the implied volatilities generated by the Heston model, one for each option in the dataset.
            model_iv = np.array([
                implied_volatility(price, S0, strike, tau, r, option_type, q)
                for price, strike, tau in zip(model_price, K, T)
            ])#`zip` takes the three arrays(price,strike,tau) and matches them element by element.
#We take one element at a time from `model_price`, one from `K`, and one from `T`. We temporarily call these three elements `price`, `strike`, and `tau`.
            valid = np.isfinite(model_iv) & np.isfinite(market_iv) #Keep only the observations where both the model IV and the market IV are finite and valid, so we create valid that is an array or True and False

            # Penalise parameter sets that fail to produce enough valid implied vols.
            if valid.sum() < 0.80 * len(K):#valid.sum() count the number of True in valid, so the number of options for which we've been able to calculate a valid implied volatility. In particular with the if condition we are imposing the condition:If the number of valid options is less than 80% of the total, then these parameters are too problematic.
                return 1e8 + (len(K) - valid.sum()) * 1e5#len(K) is the total number of options in the calibration dataset. With this return we return a huge error to the objective function.

            # More weight around ATM, while still keeping OTM information.
            weights = 1.0 / (np.abs(moneyness[valid] - 1.0) + 0.05)#This is the weight function. If moneyness is close to 1, the denominator is small, so the weight is high. Therefore, we give more importance to near-the-money options.

            error = np.mean(weights * (model_iv[valid] - market_iv[valid])**2)#loss to minimize,Weighted average of (Heston IV - market IV)^2.


            if not np.isfinite(error):
                return 1e10

            return float(error)#Returns the error to the optimizer.

        except Exception:
            return 1e10#If something goes wrong, it returns a huge error instead of stopping everything.
#parameters in differential _evolution:
#`objective` is the function to minimize.
#`bounds` are the limits on the parameters.
#`seed=42` is used for reproducibility.
#`maxiter=50` means it can go up to 50 generations.
#`popsize=6` means a relatively small population, so the code does not become too slow.
#`disp=True` prints the steps.
#`polish=True` means that at the end it performs a local refinement to improve the result.
#Then it extracts the final parameters (`result.x`).
#Then it computes the Feller condition.

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
    )#Find the vector `x` within these bounds that minimizes `objective(x)`.

    v0, kappa, theta, sigma, rho = result.x
    feller = 2 * kappa * theta - sigma**2

    print("\n── Calibrated parameters ──")
    print(f"v0 = {v0:.6f}  -> initial vol   = {np.sqrt(v0)*100:.2f}%")#`v0` and `theta` are variances, so to interpret them as volatilities we take the square root
    print(f"kappa = {kappa:.6f}")
    print(f"theta = {theta:.6f}  -> long-run vol  = {np.sqrt(theta)*100:.2f}%")
    print(f"sigma = {sigma:.6f}")
    print(f"rho = {rho:.6f}")
    print(f"Feller: 2κθ − σ² = {feller:.6f} ({'satisfied' if feller >= 0 else 'violated'})")
    print(f"Objective value: {result.fun:.8f}")

    with open("heston_params.json", "w", encoding="utf-8") as f:#i save the calibrated parameters, in particular pyhton create the file heston_params.json. `f` is the temporary name the code uses to refer to the opened file.
        json.dump(#This function takes Python data and writes it into a file in JSON format.
            {#we use the dictionary format
                "v0": float(v0),
                "kappa": float(kappa),
                "theta": float(theta),
                "sigma": float(sigma),
                "rho": float(rho),
                "objective": float(result.fun),#minimum error found by the optimizer
                "feller": float(feller),
                "min_maturity": float(MIN_MATURITY),
            },
            f,
            indent=4,
        )

    print("parameters saved")
    return result.x



# 7. HESTON IMPLIED VOL SURFACE

#This function creates the Heston volatility surface on a grid.
#Inputs:
#moneyness grid;
#maturity grid;
#calibrated parameters.

def heston_iv_surface(S0, moneyness_grid, T_grid, r, params, option_type="put", q=0.0):
    iv_mesh = np.full((len(T_grid), len(moneyness_grid)), np.nan)#create an empty matrix with len(T_grid) rows and len(moneyness_grid) columns, so one rows for every maturity and opne column for every moneyness value
    K_grid = moneyness_grid * S0 #we transform the moneyness in strike: K = moneyness × S0

    for i, T in enumerate(T_grid):
        prices = heston_option_price(S0, K_grid, T, r, params, option_type, q)
        ivs = [
            implied_volatility(p, S0, K, T, r, option_type, q)
            for p, K in zip(prices, K_grid)
        ]
        iv_mesh[i, :] = ivs#It replaces one row of the matrix with the Heston IVs computed for that maturity.

    return iv_mesh#returns the heston surface



# 8. PLOTS


def plot_smiles(df, params, r=RISK_FREE_RATE, q=DIVIDEND_YIELD, option_type="put"):
    maturities = np.sort(df["T"].unique())

    fig, axes = plt.subplots(
        1, len(maturities),
        figsize=(4.5 * len(maturities), 4.5),
        squeeze=False
    )

    for ax, T in zip(axes.ravel(), maturities):
        subset = df[df["T"] == T].sort_values("moneyness")

        K_vals = subset["strike"].to_numpy(float)
        prices = heston_option_price(S0, K_vals, T, r, params, option_type, q)

        iv_heston = np.array([
            implied_volatility(p, S0, K, T, r, option_type, q)
            for p, K in zip(prices, K_vals)
        ])

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

    rmse = np.sqrt(np.nanmean(error_mesh[valid]**2))
    print(f"Vol surface RMSE: {rmse*100:.2f}%")

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
    ax3.set_title(f"Absolute Error  (RMSE = {rmse*100:.2f}%)", fontsize=11)
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

    validate_heston_pricer()#First, it checks that the Heston pricer works properly. It performs the numerical tests.

    df = load_calibration_data(option_type="put")

    params = calibrate_heston(df, option_type="put")

    plot_smiles(df, params, option_type="put")

    plot_vol_surface(df, params, option_type="put")

    print("\nDone. Output files:")
    print("  heston_params.json")
    print("  heston_smile_fit.png")
    print("  vol_surface_comparison.png")
