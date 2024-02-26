# Reformulated Problem III: Analyzing Predictive Regressions

1. **Understanding the Model:**
   - Analyze the model: $y_t = \beta x_{t-1} + u_t$ and $x_t = \rho x_{t-1} + v_t$, where $u_t$ and $v_t$ are error terms with specified covariance.

2. **Simulation and Analysis:**
   - **Task 1:** Simulate the model using Stambaugh (1999) parameters. For various scenarios, simulate 1000 samples and compute the OLS estimator of $\beta$.
   - **Task 2:** Repeat for two-step and four-step forecasts. Evaluate the OLS slope estimator's standard deviation using naive and Newey-West methods versus the true formula.
   - **Task 3:** Use real-world data from "Irrational Exuberance". Conduct predictive regressions with real prices, guided by Hodrick (1992).

**Guidelines for Answering:**

- **Simulation Setup:** Learn to simulate time series data with the model's parameters.
- **Parameter Variation:** Observe how changes in $\rho$, $\sigma_{uv}$, and sample size affect OLS estimates of $\beta$.
- **Data Analysis:** For real-world data, select variables as suggested, focusing on real prices.
- **Interpretation of Results:** Interpret simulation outcomes and the implications for forecasting.

# Question 1 example code

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

def simulate_series(T, rho, beta, sigma_u, sigma_v, sigma_uv):
    np.random.seed(0) # For reproducibility
    x = np.zeros(T+1)
    u = np.random.normal(0, sigma_u, T+1)
    v = np.random.normal(0, sigma_v, T+1)
    # Generate correlated u and v
    u += sigma_uv / sigma_v * v
    
    for t in range(1, T+1):
        x[t] = rho * x[t-1] + v[t]
        
    y = beta * x[:-1] + u[1:]
    return y, x[:-1]

def ols_estimator(x, y):
    x = sm.add_constant(x) # Adds a constant term to the predictor
    model = sm.OLS(y, x)
    results = model.fit()
    return results.params[1] # Returns the beta coefficient

# Simulation parameters from Stambaugh (1999)
T = 100 # Sample size
rho = 0.9
beta = 0.5
sigma_u = 1
sigma_v = 1
sigma_uv = 0.5

# Simulate series
y, x = simulate_series(T, rho, beta, sigma_u, sigma_v, sigma_uv)

# Compute OLS estimator
beta_hat = ols_estimator(x, y)

print(f"Estimated beta: {beta_hat}")
```
