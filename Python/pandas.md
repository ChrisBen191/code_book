```python
import pandas as pd
import numpy as np

pd.set_option("display.notebook_repr_html", False)
```


```python
df = pd.DataFrame({'state' : ['Texas', 'Texas', 'Kansas', 'Texas', 'Missouri', 'Missouri', 'North Carolina'],
                   'city' : ['Austin', 'San Antonio', 'Kansas City', 'Llano', 'Kansas City', 'St. Louis', 'Raleigh'],
                   'bbq_rating' : [8, 9, 7, 10, 7, 6, 5],
                   'outdoor_rating' : [8, 7, 5, 3, 5, 7, 8]})

df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>city</th>
      <th>bbq_rating</th>
      <th>outdoor_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Texas</td>
      <td>Austin</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Texas</td>
      <td>San Antonio</td>
      <td>9</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Kansas</td>
      <td>Kansas City</td>
      <td>7</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Texas</td>
      <td>Llano</td>
      <td>10</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Missouri</td>
      <td>Kansas City</td>
      <td>7</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Missouri</td>
      <td>St. Louis</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>6</th>
      <td>North Carolina</td>
      <td>Raleigh</td>
      <td>5</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



# Groupby


```python
# groupby object is a pd.DataFrame if a list or array is passed to be computed
grouped_df = df.groupby('state')[['bbq_rating']]
grouped_df
```




    <pandas.core.groupby.generic.DataFrameGroupBy object at 0x7fc461866400>




```python
# groupby object is a pd.Series if only a single column is passed to be computed
grouped_series = df.groupby('state')['bbq_rating']
grouped_series
```




    <pandas.core.groupby.generic.SeriesGroupBy object at 0x7fc461866550>




```python
# creates a GROUPBY object that can be iterated over to compile aggregrates
grouped = df.groupby(df['state'])
grouped
```




    <pandas.core.groupby.generic.DataFrameGroupBy object at 0x7fc461866610>




```python
# provides a statistical overview of the groupby object
grouped.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">bbq_rating</th>
      <th colspan="8" halign="left">outdoor_rating</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>state</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Kansas</th>
      <td>1.0</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>7.00</td>
      <td>7.0</td>
      <td>7.00</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Missouri</th>
      <td>2.0</td>
      <td>6.5</td>
      <td>0.707107</td>
      <td>6.0</td>
      <td>6.25</td>
      <td>6.5</td>
      <td>6.75</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>1.414214</td>
      <td>5.0</td>
      <td>5.5</td>
      <td>6.0</td>
      <td>6.5</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>North Carolina</th>
      <td>1.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>5.00</td>
      <td>5.0</td>
      <td>5.00</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>3.0</td>
      <td>9.0</td>
      <td>1.000000</td>
      <td>8.0</td>
      <td>8.50</td>
      <td>9.0</td>
      <td>9.50</td>
      <td>10.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>2.645751</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>7.5</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# applying mean() method to groupby object grouped
grouped.mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bbq_rating</th>
      <th>outdoor_rating</th>
    </tr>
    <tr>
      <th>state</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Kansas</th>
      <td>7.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Missouri</th>
      <td>6.5</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>North Carolina</th>
      <td>5.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>9.0</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# grouping by 'outdoor_rating' field, passing multiple group keys, finding average; creates a pd.Series
means = df['outdoor_rating'].groupby([df['state'], df['city']]).mean()
means
```




    state           city       
    Kansas          Kansas City    5
    Missouri        Kansas City    5
                    St. Louis      7
    North Carolina  Raleigh        8
    Texas           Austin         8
                    Llano          3
                    San Antonio    7
    Name: outdoor_rating, dtype: int64




```python
# displays the 'size' or count of each group
df.groupby(['state']).size()
```




    state
    Kansas            1
    Missouri          2
    North Carolina    1
    Texas             3
    dtype: int64



## Groupby agg


```python
# using 'agg' with groupby object to aggregrate multi columns with multiple aggregrations
agg = {
    'city': 'nunique',
    'bbq_rating': ['mean', 'max'],
    'outdoor_rating': ['mean', 'max']
      }

# grouping by state, passing agg dict into agg() method
agg_df = df.groupby('state').agg(agg)
agg_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>city</th>
      <th colspan="2" halign="left">bbq_rating</th>
      <th colspan="2" halign="left">outdoor_rating</th>
    </tr>
    <tr>
      <th></th>
      <th>nunique</th>
      <th>mean</th>
      <th>max</th>
      <th>mean</th>
      <th>max</th>
    </tr>
    <tr>
      <th>state</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Kansas</th>
      <td>1</td>
      <td>7.0</td>
      <td>7</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Missouri</th>
      <td>2</td>
      <td>6.5</td>
      <td>7</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>North Carolina</th>
      <td>1</td>
      <td>5.0</td>
      <td>5</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>3</td>
      <td>9.0</td>
      <td>10</td>
      <td>6</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
# list comphrension that creates new column names with column_aggregration notation; reset_index() flattens the summary
agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
agg_df.reset_index()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>city_nunique</th>
      <th>bbq_rating_mean</th>
      <th>bbq_rating_max</th>
      <th>outdoor_rating_mean</th>
      <th>outdoor_rating_max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Kansas</td>
      <td>1</td>
      <td>7.0</td>
      <td>7</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Missouri</td>
      <td>2</td>
      <td>6.5</td>
      <td>7</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>North Carolina</td>
      <td>1</td>
      <td>5.0</td>
      <td>5</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Texas</td>
      <td>3</td>
      <td>9.0</td>
      <td>10</td>
      <td>6</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



### Iterating over Groups


```python
# groupby object supports iteration; generates a tuple of group key name, associated data
for state, group in df.groupby('state'):
    print(f'State: {state}')
    print(group, '\n')
```

    State: Kansas
        state         city  bbq_rating  outdoor_rating
    2  Kansas  Kansas City           7               5 
    
    State: Missouri
          state         city  bbq_rating  outdoor_rating
    4  Missouri  Kansas City           7               5
    5  Missouri    St. Louis           6               7 
    
    State: North Carolina
                state     city  bbq_rating  outdoor_rating
    6  North Carolina  Raleigh           5               8 
    
    State: Texas
       state         city  bbq_rating  outdoor_rating
    0  Texas       Austin           8               8
    1  Texas  San Antonio           9               7
    3  Texas        Llano          10               3 
    



```python
# iterating over groupby object with multiple passed group keys
for (state, city), group in df.groupby(['state', 'city']):
    print(f'State: {state}')
    print(f'Specific City: {city}')
    print(group, '\n')
```

    State: Kansas
    Specific City: Kansas City
        state         city  bbq_rating  outdoor_rating
    2  Kansas  Kansas City           7               5 
    
    State: Missouri
    Specific City: Kansas City
          state         city  bbq_rating  outdoor_rating
    4  Missouri  Kansas City           7               5 
    
    State: Missouri
    Specific City: St. Louis
          state       city  bbq_rating  outdoor_rating
    5  Missouri  St. Louis           6               7 
    
    State: North Carolina
    Specific City: Raleigh
                state     city  bbq_rating  outdoor_rating
    6  North Carolina  Raleigh           5               8 
    
    State: Texas
    Specific City: Austin
       state    city  bbq_rating  outdoor_rating
    0  Texas  Austin           8               8 
    
    State: Texas
    Specific City: LLano
       state   city  bbq_rating  outdoor_rating
    3  Texas  LLano          10               3 
    
    State: Texas
    Specific City: San Antonio
       state         city  bbq_rating  outdoor_rating
    1  Texas  San Antonio           9               7 
    


### Creating Groupby Dict


```python
# computing a dict of the data pieces
pieces = dict(list(df.groupby('state')))

pieces['Texas']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>city</th>
      <th>bbq_rating</th>
      <th>outdoor_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Texas</td>
      <td>Austin</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Texas</td>
      <td>San Antonio</td>
      <td>9</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Texas</td>
      <td>LLano</td>
      <td>10</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# groupby object is a pd.DataFrame if a list or array is passed to be computed
grouped_df = df.groupby('state')[['bbq_rating']]
grouped_df
```




    <pandas.core.groupby.generic.DataFrameGroupBy object at 0x7fc6319e4610>




```python
# groupby object is a pd.Series if only a single column is passed to be computed
grouped_series = df.groupby('state')['bbq_rating']
grouped_series
```




    <pandas.core.groupby.generic.SeriesGroupBy object at 0x7fc6319f7d90>




```python

df.groupby('state').quantile(0.9)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bbq_rating</th>
      <th>outdoor_rating</th>
    </tr>
    <tr>
      <th>state</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Kansas</th>
      <td>7.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Missouri</th>
      <td>6.9</td>
      <td>6.8</td>
    </tr>
    <tr>
      <th>North Carolina</th>
      <td>5.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>9.8</td>
      <td>7.8</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
