# Python

## Importing Data
---

### Glob


```python
import glob
import pandas as pd

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
    ('Jack Jordans', 'Odessa', 'smoked sausage', 12),
    ('Russels', 'Denver', 'brisket burnt ends', 25),
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
    
# diplays the Barbeque nametuples
bbq_joints
```




    [Barbeque(name='Jack Jordans', location='Odessa', special='smoked sausage', special_price=12),
     Barbeque(name='Russels', location='Denver', special='brisket burnt ends', special_price=25),
     Barbeque(name='Franklins', location='Austin', special='chopped sandwich', special_price=15)]




```python
# nametuples' attributes are now easily accessible
for joint in bbq_joints:
    print(f'{joint.name} BBQ in {joint.location} has a great {joint.special} special!')
```

    Jack Jordans BBQ in Odessa has a great smoked sausage special!
    Russels BBQ in Denver has a great brisket burnt ends special!
    Franklins BBQ in Austin has a great chopped sandwich special!


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

    {'mediterranean', 'chili', 'steak', 'wings'}


### difference


```python
#DIFFERENCE collects values only in the set specified
tx_set_only = tx_set.difference(co_set)
co_set_only = co_set.difference(tx_set)

print(f'Only a Texas favorite\n {tx_set_only}\n')
print(f'Only a Colorado favorite\n {co_set_only}')
```

    Only a Texas favorite
     {'bbq', 'brisket', 'burritos', 'hatch chilies', 'queso'}
    
    Only a Colorado favorite
     {'sushi', 'green chilies', 'pizza'}


### symmetric difference


```python
# SYMMETRIC_DIFFERENCE collects values that exist in ONE of the sets, but NOT BOTH
differences = tx_set.symmetric_difference(co_set)
print(f'Either Texas/Colorado Favorite, BUT NOT BOTH\n {differences}\n')
```

    Either Texas/Colorado Favorite, BUT NOT BOTH
     {'sushi', 'brisket', 'pizza', 'bbq', 'green chilies', 'burritos', 'hatch chilies', 'queso'}
    


### union


```python
# UNION collects all values from each set (no duplicates)
union_set = tx_set.union(co_set)
print(f'All Favorites (both states), NO DUPLICATES\n {union_set}\n')
```

    All Favorites (both states), NO DUPLICATES
     {'pizza', 'bbq', 'sushi', 'wings', 'brisket', 'green chilies', 'steak', 'burritos', 'mediterranean', 'hatch chilies', 'queso', 'chili'}
    


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

    {'bbq', 'burger', 'wings', 'brisket', 'steak', 'burritos', 'mediterranean', 'hatch chilies', 'queso', 'chili'}
    {'pizza', 'buffalo burger', 'sushi', 'green chilies', 'wings', 'breakfast skillet', 'steak', 'mediterranean', 'chili'}



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

print(f'Tuple Count equals length of smallest passed list:\n {zipped_list}')
```

    Tuple Count equals length of smallest passed list:
     [('cereal', 'steak'), ('scrambled eggs', 'cereal'), ('yogurt', 'burger')]


## List Comprehensions
---


```python
price_list = [15.99, 7.95, 34.50, 20.20, 11.59]
print(f'Starting Prices:\n {price_list}')
```

    Starting Prices:
     [15.99, 7.95, 34.5, 20.2, 11.59]



```python
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
print(f'Prices after GiftCard applied:\n {after_gc}')
```

    Prices after GiftCard applied:
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

### dictionary


```python
# creates a dictionary of key:value pairs
meal_dict = {
    'breakfast': 'scrambled eggs',
    'lunch': 'sandwich',
    'dinner': 'steak'
}

# dictionary is accessed by providing the key in brackets
lunch = meal_dict['lunch']
print(f'Lunch Meal: {lunch}')
```

    Lunch Meal: sandwich


### List of dictionaries


```python
# creates a list of dictionaries of key:value pairs
list_of_dicts = [
    {'breakfast': 'scrambled eggs'},
    {'lunch': 'sandwich'},
    {'dinner': 'steak'}
]

# dictionary is accessed by providing index of dict in list, then providing the key for that dict
lunch = list_of_dicts[1]['lunch']
print(f'Lunch Meal: {lunch}')
```

    Lunch Meal: sandwich



```python
snack = {'snack': 'cheezits'}

# appends adds the snack dict the list_of_dicts
list_of_dicts.append(snack)
print(list_of_dicts)
```

    [{'breakfast': 'scrambled eggs'}, {'lunch': 'sandwich'}, {'dinner': 'steak'}, {'snack': 'cheezits'}]


### Dict of dictionaries


```python
dict_of_dicts = {
    'breakfast': {'light': 'yogurt', 'medium': 'breakfast sandwich', 'heavy': 'hangover burger'},
    'lunch': {'light': 'sandwich', 'medium': 'pizza', 'heavy': 'fried chicken'},
    'dinner': {'light': 'steak salad', 'medium': 'spagetti', 'heavy': 'steak'}
}

# dictionary is access by providing the key to the nested dict, then providing the key for that dict
lunch = dict_of_dicts['lunch']['light']
print(f'Lunch Meal: {lunch}')
```

    Lunch Meal: sandwich



```python
snacks = {'snacks': {'light': 'cheezeits', 'medium': 'potato chips', 'heavy': 'chocolate cake'}}

# updates dict_of_dicts with snacks dictionary
dict_of_dicts.update(snacks)

for meal_key in dict_of_dicts:
    print(f'Meals for {meal_key}:\n {dict_of_dicts[meal_key]}\n')
```

    Meals for breakfast:
     {'light': 'yogurt', 'medium': 'breakfast sandwich', 'heavy': 'hangover burger'}
    
    Meals for lunch:
     {'light': 'sandwich', 'medium': 'pizza', 'heavy': 'fried chicken'}
    
    Meals for dinner:
     {'light': 'steak salad', 'medium': 'spagetti', 'heavy': 'steak'}
    
    Meals for snacks:
     {'light': 'cheezeits', 'medium': 'potato chips', 'heavy': 'chocolate cake'}
    


### Dictionary Modifiers


```python
sport_dict = {
    'basketball': ['lakers', 'spurs', 'nets'],
    'football': ['cowboys', 'chiefs', 'seahawks'],
}

# conditional to determine if element is inside of football list in sport_dict
if 'cowboys' in sport_dict['football']:
    print('The team is inside the dictionary!')
```

    The team is inside the dictionary!



```python
# loops through sport_dict to display the sport keys
print('List of Sports: (keys representing each sport)')
for sport in sport_dict.keys():
    print(sport)
```

    List of Sports: (keys representing each sport)
    basketball
    football



```python
# loops through sport_dict to display the list of teams
print('\nList of Teams: (values for each sport key)')
for teams in sport_dict.values():
    print(teams)
```

    
    List of Teams: (values for each sport key)
    ['lakers', 'spurs', 'nets']
    ['cowboys', 'chiefs', 'seahawks']



```python
# loops through sport_dict to display the sport keys and corresponding team values
for sport, teams in sport_dict.items():
    print(f'\n{sport} Teams: \n {teams}')
```

    
    basketball Teams: 
     ['lakers', 'spurs', 'nets']
    
    football Teams: 
     ['cowboys', 'chiefs', 'seahawks']


## Functions
---

### docstring example


```python
def function(arg_1, arg_2=42):
	"""Description of what the function does. Google Docstring Style
	Args:
    arg_1 (str): Description of arg_1 that can break onto the next line if needed.
    arg_2 (int, optional): Write optional when an argument has a default value.

    Returns:
    bool: Optional description of the return value Extra lines are not indented.

    Raises:
    ValueError: Include any error types that the function intentionally raises.

    Notes:
    See https://www.datacamp.com/community/tutorials/docstrings-python for more info.
	"""
```

### unpacking arguments


```python
def multiply(*args):
    """
    Using * lets the function know to unpack the arguments; allows for any number
    of arguments to be passed into the function.
    """
    print(args)
    
    total = 1
    for arg in args:
        # multiplies the unpacked arguements
        total = total * arg
        
    return total
    
# runs the multiply function with several sets of values
print(multiply(1,3,5), '\n')
print(multiply(-1), '\n')
print(multiply(12,3,44,-4))
```

    (1, 3, 5)
    15 
    
    (-1,)
    -1 
    
    (12, 3, 44, -4)
    -6336


### destructuring arguments into parameters


```python
def add(x,y):
    """
    Using * when calling the add function breaks down the parameter into the x,y variables;
    requires the amount of desctructured parameters and function variables are the same.
    """
    return x + y


# using '*' allows the list passed to be deconstructed
nums = [3,5]
print(add(*nums), '\n')

# using '**' allows the dictionary passed to be deconstructed
nums_dict = {'x':10, 'y':10}
print(add(**nums_dict))
```

    8 
    
    20


## keyword arguments


```python
def named(**kwargs):
    """
    Specifying **kwargs will process any number of passed keyword parameters inside dict
    """
    print(kwargs,'\n')
    
named(name='bob', age=25)
```

    {'name': 'bob', 'age': 25} 
    



```python
def print_nicely(**kwargs):
    """
    Passing the kwargs and iterating over any number of parameters
    """
    named(**kwargs)
    
    for arg, value in kwargs.items():
        print(f'{arg}: {value}')
    
# dict containing kw arguments to be passed into print_nicely function
details = {'name':'bob', 'age':25}
print_nicely(**details)
```

    {'name': 'bob', 'age': 25} 
    
    name: bob
    age: 25



```python
def both(*args, **kwargs):
    """
    this setup is typically used so all arguments can be passed into another function
    """
    print(args, '\n')
    print(kwargs)
    
both(1,2,3,12, name='bob', age=25)
```

    (1, 2, 3, 12) 
    
    {'name': 'bob', 'age': 25}

