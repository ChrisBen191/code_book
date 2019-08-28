# DEPENDENCIES

        import numpy as np


# NUMPY METHODS

Stores the **square root** of the value **x** and stores it as a variable.

        square_root = np.sqrt(x)

Takes the **value** and raises it by the power of the **power_value**.

        power_of_num = np.power(power_value, value)

Takes the **sine** of the value specified and stores it as a variable.

        sine_of_num = np.sin(value)

Creates **arr_list**,  a list of arrays.

        arr_list = [[1,2,3], [4,5,6], [7,8,9]]

Creates a **matrix** from a **arr_list**, a list of arrays. 

        my_matrix = np.array(arr_list)

Displays the **shape** of the matrix (row and column count).

        my_matrix.shape

Arranges the values in the array specified; uses a **start** and **end** value, as well as an option **interval** amount. **End** value is not included in calculation.

        np.arange( start_int, end_int, interval_int)

Creates a matrix of **zeroes** of the shape **num_rows** and **num_columns** specified. Can use **np.ones()** to produce a matrix of **ones**.

        np.zeroes( num_of_rows , num_of_columns)

Returns **linearly spaced** numbers over the specified interval; the **start** and **end** are included in the calculation.

        np.linspace( start_int, end_int, amt_of_nums )

Creates an **identity matrix** (ones across the diagonal matrix) with the number of rows/columns specified by **matrix_num**.

        identity_matrix = np.eye( matrix_num )

Creates **random_arr**, an array containing the **num_of_values** of random numbers specified

        random_arr = np.random.rand( num_of_values )

Creates **rando_arr**, an array containing the **num_of_values** of random numbers specified; **randn** however returns samples from the **standard normal distribution** (0 to 1) so values closer to zero are more likely to appear.

        random_arr = np.random.randn( num_of_values )

