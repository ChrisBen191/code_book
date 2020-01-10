# DEPENDENCIES



# MODIFIER COMMANDS

converts a string index of date/time to a datetime format
```python
df.index = pd.to_datetime(df.index)
```

correlation provides insight between the compatibility between two series; typically used with '% Change'
```python
# '% Change' converts the series in the df to provide a more accurate correlation
pct_df = df.pct_change()

correlation = pct_df['Column_Name_1'].corr(pct_df['Column_Name_2'])
```
preform a linear regression using **pandas**
```python
pd.ols(y, x)
```

preform a linear regression using **statsmodels**
```python
import statsmodels.api as sm
# regression boilerplate = sm.OLS(y, x).fit()

# compute % changes in the series
pct_df = df.pct_change()

# add a constant to the df for the regression intercept
pct_df = sm.add_constant(df)

# delete the row of NaN (created when % change is calculated)
pct_df = pct_df.dropna()

# run the regression
results = sm.OLS(pct_df['Column_Name_1'], df[['const', 'Column_Name_2']]).fit()

# 'summary' provides an in depth analysis; review the 'coef' and 'R-squared' information 
print(results.summary())
```

**autocorrelation** is the correleation of a single times series with a lagged copy of itself
```python
pct_df = df.pct_change()

# autocorrelation also known as 'serial correlation' or 'lag-one' autocorrelation
autocorrelation = df['Column_Name'].autocorr()

# negative autocorrelation = mean reverting
# positive autocorrelation = trend-following or momentum
```

provides **autocorrelation function (ACF)** numerical values instead of the plot
```python
from statsmodels.tsa.stattools import acf

# x= series or array
acf_values = acf(x)
```

calculates and plots the **sample autocorrelation function (ACF)**, which shows different lags (not just 'lag-one')
```python
from statsmodels.graphics.tsaplots import plot_acf

# x= series or array
# lags= # of lags the autocorrelation function will be plotted 
# aplha= width of confidence interval (0-1), alpha=1 removes confidence interval
plot_acf(x, lags=num, alpha=0.05)

# any significant non-zero autocorrelations imply that the series can be forecast from the past
plt.show()
```



