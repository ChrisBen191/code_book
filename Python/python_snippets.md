# Python

## Importing Data
---

### Glob


```python
import glob

# list stating the differences in the csvs to be imported
csv_list = ['northern', 'southern', 'eastern', 'western']

# initalizing an empty list to store the df's created
df_list = []

# looping over each list_element to create 'file_name'
for list_element in csv_list:
    
    file_name = f'../Path/To/File/{list_element}US_popular_diners'

    # using glob to wildcard search the appropriate csv file; file type added at end of string
    for file in glob.glob(file_name + '*.csv'):

        # saving each df generated to 'df_list'
        df_list.append(pd.read_csv(file, index_col=None))

# concatinating the stored dfs together, row-wise or union-style
complete_df = pd.concat(df_list, axis=0)
```

## Iterators
---

### Range


```python
# increments are specified using STEP
num_range = range(0, 16, 2)

# converts RANGE OBJECT to a list
nums_list = list(num_range)
print(nums_list)
```

    [0, 2, 4, 6, 8, 10, 12, 14]


### Enumerate


```python
# indexing starting at START value specified
enumerate_values = enumerate(['cat', 'dog', 'monkey', 'lemur'], start = 10)

# converts the ENUMERATE OBJECT to a list
indexed_list = list(enumerate_values)

print(indexed_list)
```

    [(10, 'cat'), (11, 'dog'), (12, 'monkey'), (13, 'lemur')]


### Map


```python
# MAP can be passed an aggregration to be applied to the second parameter, list of digits
rounded_values = map(round, [12.3, 14.4, 17.8])

# converts the MAP object to a list
rounded_list = list(rounded_values)
print(f'Rounded Values: {rounded_list}\n')

# MAP can be passed a parsing or string modifier to be applied to the second parameter, list of strings
titled_values = map(str.title, ['denver', 'longbeach', 'hobbs'])

# converts the MAP object to a list
titled_list = list(titled_values)
print(f'Titled Values: {titled_list}\n')

# MAP can be used with lambda to pass anonymous functions without looping
squared_values = map(lambda x: x**2, [6, 8, 9, 12, 14])

# converts the MAP object to a list
squares_list = list(squared_values)
print(f'Squared Values: {squares_list}\n')
```

    Rounded Values: [12, 14, 18]
    
    Titled Values: ['Denver', 'Longbeach', 'Hobbs']
    
    Squared Values: [36, 64, 81, 144, 196]
    


## Collections Module
---

### Counter


```python
from collections import Counter

pet_list = ['dog', 'dog', 'cat', 'parrot', 'monkey', 'dog', 'frog', 'cat', 'parrot', 'horse',
            'cat', 'dog','hamster', 'horse','snake', 'cat', 'parrot', 'frog', 'hamster', 'snake']

# collects the COUNT of values in pet_list; stores as COUNTER object saved as counter_dict
counter_dict = Counter(pet_list)
print(f'COUNTER Object (dict)\n {counter_dict}\n')

# capturing top value counts (top 3) and saving LIST OF TUPLES as top_3
top_3 = counter_dict.most_common(3)
print(f'Top 3 Popular Pets\n {top_3}')
```

    COUNTER Object (dict)
     Counter({'dog': 4, 'cat': 4, 'parrot': 3, 'frog': 2, 'horse': 2, 'hamster': 2, 'snake': 2, 'monkey': 1})
    
    Top 3 Popular Pets
     [('dog', 4), ('cat', 4), ('parrot', 3)]


### defaultdict


```python
from collections import defaultdict


pet_colors = {
    'dog': ['golden', 'brindle'], 
    'cat': ['black', 'calico'], 
    'horse': ['chocolate', 'spotted']
}

# creating a new dict, with the default value of an empty list
new_dict = defaultdict(list)

# iterating through pet_colors dict, storing k:v pairs to the DEFAULTDICT named new_dict 
for key, value_list in pet_colors.items():
    new_dict[key].append(value_list)

    
# the DEFAULTDICT named new_dict does not have frog key, but by default returns an empty dict
print(f'Frog value does not exist yet: {new_dict["frog"]}\n')

# frog key now exists in new_dict, with an empty list as it's value by default
print(f'Frog value defaulted to empty list due to DEFAULTDICT\n {new_dict}')
```

    Frog value does not exist yet: []
    
    Frog value defaulted to empty list due to DEFAULTDICT
     defaultdict(<class 'list'>, {'dog': [['golden', 'brindle']], 'cat': [['black', 'calico']], 'horse': [['chocolate', 'spotted']], 'frog': []})


### namedtuple


```python
from collections import namedtuple

# list of tuples with bbq joint information, stored as bbq_tuples
bbq_tuples = [
    ('Jack Jordans', 'Odessa', 'brisket', 12),
    ('Russels', 'Denver', 'brisket burt ends', 25),
    ('Franklins', 'Austin', 'chopped sandwich', 15)
]

# creating the BARBEQUE nametuple and passing list of attribute names
Barbeque = namedtuple('Barbeque', ['name', 'location', 'special', 'special_price'])

# initializing a list to hold the list of namedtuples
bbq_joints = []

# looping over bbq_tuples, identifying each attribute
for joint, location, special, special_price in bbq_tuples:
    
    # creating NAMEDTUPLE object as details, appending to bbq_joints list
    details = Barbeque(joint, location, special, special_price)
    bbq_joints.append(details)
    
bbq_joints
```




    [Barbeque(name='Jack Jordans', location='Odessa', special='brisket', special_price=12),
     Barbeque(name='Russels', location='Denver', special='brisket burt ends', special_price=25),
     Barbeque(name='Franklins', location='Austin', special='chopped sandwich', special_price=15)]




```python
# nametuples' attributes are now easily accessible
for joint in bbq_joints:
    print(f'{joint.name} in {joint.location} has a great {joint.special} special!')
```

    Jack Jordans in Odessa has a great brisket special!
    Russels in Denver has a great brisket burt ends special!
    Franklins in Austin has a great chopped sandwich special!


## Itertools Module
---

### Combinations


```python
from itertools import combinations

colors = ['red', 'blue', 'green']

# '2' indicates NUM OF COMBINATIONS, '*' unpacks the COMBO OBJECT to a list
color_mixes = [ *combinations(colors, 2) ]

color_mixes
```




    [('red', 'blue'), ('red', 'green'), ('blue', 'green')]



## Sets
---


```python
tx_favorites = ['steak', 'brisket', 'burritos', 'chili', 'bbq', 'hatch chilies', 'queso', 'mediterranean', 'wings']
co_favorites = ['steak', 'green chilies', 'chili', 'sushi', 'pizza', 'mediterranean', 'wings']

# converting lists into SET objects which only store unique values (no duplicates)
tx_set = set(tx_favorites)
co_set = set(co_favorites)
```

### intersection


```python
# INTERSECTION collects values shared between the two sets
both_sets = tx_set.intersection(co_set)
print(both_sets)
```

### difference


```python
#DIFFERENCE collects values only in the set specified
tx_set_only = tx_set.difference(co_set)
co_set_only = co_set.difference(tx_set)

print(f'Only a Texas favorite\n {tx_set_only}\n')
print(f'Only a Colorado favorite\n {co_set_only}')
```

    Only a Texas favorite
     {'hatch chilies', 'bbq', 'brisket', 'burritos', 'queso'}
    
    Only a Colorado favorite
     {'pizza', 'sushi', 'green chilies'}


### symmetric difference


```python
# SYMMETRIC_DIFFERENCE collects values that exist in ONE of the sets, but NOT BOTH
differences = tx_set.symmetric_difference(co_set)
print(f'Either Texas/Colorado Favorite, BUT NOT BOTH\n {differences}\n')
```

    Either Texas/Colorado Favorite, BUT NOT BOTH
     {'pizza', 'hatch chilies', 'bbq', 'sushi', 'green chilies', 'brisket', 'burritos', 'queso'}
    


### union


```python
# UNION collects all values from each set (no duplicates)
union_set = tx_set.union(co_set)
print(f'All Favorites (both states), NO DUPLICATES\n {union_set}\n')
```

    All Favorites (both states), NO DUPLICATES
     {'pizza', 'hatch chilies', 'chili', 'wings', 'bbq', 'sushi', 'green chilies', 'mediterranean', 'brisket', 'burritos', 'queso', 'steak'}
    


### set modifiers


```python
# add a single value to the set only if the value is unique
tx_set.add('burger')

# merges a list of values into the set, only if the values are unique
co_set.update(['buffalo burger', 'breakfast skillet', 'salads'])

# removes an element from the set
co_set.discard('salads')

print(tx_set)
print(co_set)
```

    {'hatch chilies', 'chili', 'wings', 'burger', 'bbq', 'mediterranean', 'brisket', 'burritos', 'queso', 'steak'}
    {'breakfast skillet', 'pizza', 'chili', 'wings', 'mediterranean', 'sushi', 'green chilies', 'buffalo burger', 'steak'}



```python
# check to see if a POSSIBLE VALUE is a member of a set; faster than list looping
if 'burger' in tx_set:
    print('The set contains the value!')
```

    The set contains the value!


## Lists
---

|            Command             | Behavior                                                                                             |
| :---------------------------: | -------------------------------------------------------------------------------------------- |
|    `list.append('value')`     | adds the value specified to the **end** of list                                              |
| `list.insert(index, 'value')` | **inserts** the value into the list at the index specified                                   |
|    `list.remove('value')`     | **removes** the element with the corresponding 'value' from the list specified               |
|     `list.pop(position)`      | **removes** or "pops" the element with the corresponding index value from the list specified |
|        `list.upper()`         | **uppercases** each string element in the list specified                                     |
|        `list.lower()`         | **lowercases** each string element in the list specified                                     |
|        `list.title()`         | **uppercases the initial letter** in each element in the list specified                      |
|  `list_one.extend(list_two)`  | **appends** list_two to list_one; similar to .append()                                       |
|     `list.index('value')`     | provides the **position or index** of an item in a list                                      |
|        `sorted(list)`         | provides the items in a list sorted in **alphabetic/numeric order**, ascending               |
|     `del list[index_num]`     | deletes the element in the list with the corresponding index_num                             |
|         `list[:int]`          | slices list from the beginning element of the list to element with **int-1** index           |
|         `list[int:]`          | slices list from the element with **int** index to the last element in the list              |
|   `list[start_int:end_int]`   | slices list from element with index **start_num** to the element with **end_num-1** index    |

### Zip

Combines lists into tuples the length of lists passed. Tuples created equals the smallest list passed.


```python
breakfast = ['cereal', 'scrambled eggs', 'yogurt']
dinner = ['steak', 'cereal', 'burger', 'pasta', 'salmon']

zipped_list = [*zip(breakfast, dinner)]

print(zipped_list)
```

    [('cereal', 'steak'), ('scrambled eggs', 'cereal'), ('yogurt', 'burger')]


## List Comprehensions
---


```python
price_list = [15.99, 7.95, 34.50, 20.20, 11.59]

# determing 15% discount for each item in price list w/list comprehension
discounts = [price * (0.85) for price in price_list]

# formatting each price into $$.$$ format w/list comprehension
discounts = [round(discount, 2) for discount in discounts]
print(f'Price List Discounts:\n {discounts}')
```

    Price List Discounts:
     [13.59, 6.76, 29.32, 17.17, 9.85]



```python
# filters out elements not meeting conditional clause
cheap = [price for price in discounts if price>=10]
print(f'Cheapest Prices:\n {cheap}')
```

    Cheapest Prices:
     [13.59, 29.32, 17.17]



```python
# if/else providing string values cooresponding if conditional is met 
price_levels = ['expensive' if price>10 else 'cheap' for price in discounts]

# assume $10 gift card can be applied to only $15 and more...
after_gc = [price-10 if price>=15 else price for price in discounts]
print(after_gc)
```

    [13.59, 6.76, 19.32, 7.170000000000002, 9.85]


## Dictionaries
---

|           Command            | Behavior                                                                                                                  |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `dict[new_key] = new_value` | updates the dict with a new k:v pair denoted by passed key and value                                              |
| `dict.update(second_dict)`  | updates the dict with a passed dictionary, tuple(s), etc.                                                         |
|      `dict.pop(value)`      | safely removes the value from the dict if it exists                                                               |
|   `dict.get(value, 'NA')`   | safely retrieves the value from the dict if it exists, else it returns secondary value (default None)             |
|        `key in dict`        | boolean statement for if key exists in specified dict; commonly used in conditional statements when parsing dicts |
