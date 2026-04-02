# Multi-Asset Black–Scholes Pricing Model

This project extends the classical Black–Scholes framework to price derivatives dependent on multiple underlying assets. Unlike the standard single-asset formulation, this model incorporates correlations between asset price movements using a multivariate stochastic process.

## Overview

In real financial markets, many derivatives depend on more than one asset (e.g., basket options, spread options). To model this, we simulate correlated asset paths using geometric Brownian motion and estimate option prices via Monte Carlo methods.

## Key Features

* Multi-asset pricing using correlated stochastic processes
* Monte Carlo simulation for numerical estimation
* Sensitivity analysis across volatility, correlation, and maturity
* Modular implementation for experimenting with different parameters

## Methodology

1. Model each asset using geometric Brownian motion
2. Introduce correlation via Cholesky decomposition of the covariance matrix
3. Simulate multiple price paths
4. Compute expected discounted payoff to estimate option price

## Tech Stack

* Python
* NumPy, SciPy
* Matplotlib (for visualization)

## Motivation

The standard Black–Scholes model assumes a single underlying asset, which limits its applicability. This project explores a more realistic setting where multiple assets interact, providing better insight into complex financial instruments.

## Future Work

* Variance reduction techniques (e.g., antithetic variates)
* Closed-form approximations for specific multi-asset options
* GPU acceleration for large-scale simulations
