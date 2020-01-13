# DEPENDENCIES
```python
import numpy as np
```

# NUMPY METHODS
|                                |                                                                              |
| ------------------------------ | ---------------------------------------------------------------------------- |
| `np.sqrt(value)`               | Calcaulates the **square root** of the value specified                       |
| `np.power(power_value, value)` | Takes the **value** specified and raises it by the value of **power_value**. |
| `np.sin(value)`                | Calculates the **sine** of the value specified                               |
| `np.cov(x, y)`                 | Computes the 2D **covariance matrix** for the x and y arrays specified       |
|`np.corrcoef(x, y)`                          |Computes the **Pearson Correlation coefficient** (condsidered easier to interpret than covariance) for the x and y arrays specified|
|`np.zeroes( num_of_rows , num_of_columns)`|Creates a matrix of **zeroes** of the shape **num_rows** and **num_columns** specified.|


 Can use **np.ones()** to produce a matrix of **ones**.




Creates **arr_list**,  a list of arrays.
```python
arr_list = [[1,2,3], [4,5,6], [7,8,9]]
```

Creates a **matrix** from a **arr_list**, a list of arrays. 
```python
my_matrix = np.array(arr_list)
```

Displays the **shape** of the matrix (row and column count).
```python
my_matrix.shape
```

Arranges the values in the array specified; uses a **start** and **end** value, as well as an option **interval** amount. **End** value is not included in calculation.
```python
np.arange( start_int, end_int, interval_int)
```

Calculates the **percentiles** of the specified data
```python
# the second parameter takes no an array of the percentiles to calculate
np.percentile(data, [25, 50,75])
```

Returns **linearly spaced** numbers over the specified interval; the **start** and **end** are included in the calculation.
```python
np.linspace( start_int, end_int, amt_of_nums )
```

Creates an **identity matrix** (ones across the diagonal matrix) with the number of rows/columns specified by **matrix_num**.
```python
identity_matrix = np.eye( matrix_num )
```

Creates **random_arr**, an array containing the **num_of_values** of random numbers specified
```python
random_arr = np.random.rand( num_of_values )
```

Creates **rando_arr**, an array containing the **num_of_values** of random numbers specified; **randn** however returns samples from the **standard normal distribution** (0 to 1) so values closer to zero are more likely to appear.
```python
random_arr = np.random.randn( num_of_values )
```
