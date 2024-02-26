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

# Question 2 example code

```python
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.sandbox.regression.gmm import NeweyWest

# Function to simulate the model
def simulate_model(n, beta, rho, sigma_u, sigma_v, sigma_uv, steps=1):
    np.random.seed(0)  # For reproducibility
    x = np.zeros(n)
    y = np.zeros(n)
    u = np.random.normal(0, sigma_u, n)
    v = np.random.normal(0, sigma_v, n)
    cov_uv = sigma_uv * sigma_u * sigma_v
    u += cov_uv / sigma_u * v  # Add covariance

    for t in range(1, n):
        x[t] = rho * x[t-1] + v[t]
        if t >= steps:
            y[t] = beta * x[t-steps] + u[t]

    return y, x

# Function to compute OLS and its standard deviation
def compute_ols_std(y, x, lags=1):
    model = OLS(y, sm.add_constant(x)).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    return model.params[1], model.bse[1]

# Parameters (example values, adjust based on the problem)
n = 100  # Sample size
beta = 0.5
rho = 0.9
sigma_u = 1
sigma_v = 1
sigma_uv = 0.5

# Simulate and compute OLS for two-step and four-step ahead forecasts
for steps in [2, 4]:
    y, x = simulate_model(n, beta, rho, sigma_u, sigma_v, sigma_uv, steps=steps)
    beta_est, std_est = compute_ols_std(y[steps:], x[steps:], lags=steps)
    print(f"Steps ahead: {steps}, OLS estimate of beta: {beta_est:.4f}, Standard deviation: {std_est:.4f}")

# Comparison with true analytical formula needs to be conducted based on specific model details
```
# Question 3 example code

```python
import pandas as pd
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv('data/shiller_data_clean.csv')

# Selecting the necessary columns for the predictive regression
# Assuming we're predicting 'Real price' based on the model suggested in Problem III
# Adjust the 'X' variable as per the specific requirements of the question
X = df[['Real Dividend', 'Long_interest_rate']]  # Example predictors
y = df['Real price']

# Adding a constant for the intercept
X = sm.add_constant(X)

# Splitting the data into training and testing sets if necessary
# Here, we use the entire dataset for simplicity
# For a time series analysis, consider chronological splits

# Perform the OLS regression
model = sm.OLS(y, X).fit()

# Print out the summary of the regression
print(model.summary())

# Interpretation of the results
# This should include an analysis of the coefficients, R-squared, and any statistical tests or diagnostics relevant to the model's assumptions.
```
