# DEPENDENCIES

        import numpy as np


# NUMPY COMMANDS

Stores the **square root** of the value **x** and stores it as a variable.

        square_root = np.sqrt(x)

Takes the **value** and raises it by the power of the **power_value**.

        power_of_num = np.power(power_value, value)

Takes the **sine** of the value specified and stores it as a variable.

        sine_of_num = np.sin(value)

Creates **array_list**,  a list of arrays.

        array_list = [[1,2,3], [4,5,6], [7,8,9]]

Creates a **matrix** from a **array_list**, a list of arrays. 

        my_matrix = np.array(array_list)

Displays the **shape** of the matrix (row and column count).

        my_matrix.shape

Arranges the values in the array specified; uses a **start** and **end** value, as well as an option **interval** amount. **End** value is not included in calculation.

        np.arrange( start_int, end_int, interval_int)

Creates a matrix of **zeroes** of the shape (rows, columns) specified. Can use **np.ones()** to produce a matrix of **ones**.

        np.zeroes(10,10)

Returns evenly spaced numbers over the specified interval; the **start** and **end** are included in the calculation.

        np.linspace( start_int, end_int, amt_of_nums)