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

### Enumerate


```python
# indexing starting at START value specified
enumerate_values = enumerate(['cat', 'dog', 'monkey', 'lemur'], start = 10)

# converts the ENUMERATE OBJECT to a list
indexed_list = list(enumerate_values)

print(indexed_list)
```

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

    {'wings', 'steak', 'mediterranean', 'chili'}


### difference


```python
#DIFFERENCE collects values only in the set specified
tx_set_only = tx_set.difference(co_set)
co_set_only = co_set.difference(tx_set)

print(f'Only a Texas favorite\n {tx_set_only}\n')
print(f'Only a Colorado favorite\n {co_set_only}')
```

    Only a Texas favorite
     {'bbq', 'burritos', 'queso', 'hatch chilies', 'brisket'}
    
    Only a Colorado favorite
     {'green chilies', 'sushi', 'pizza'}


### symmetric difference


```python
# SYMMETRIC_DIFFERENCE collects values that exist in ONE of the sets, but NOT BOTH
differences = tx_set.symmetric_difference(co_set)
print(f'Either Texas/Colorado Favorite, BUT NOT BOTH\n {differences}\n')
```

    Either Texas/Colorado Favorite, BUT NOT BOTH
     {'green chilies', 'bbq', 'pizza', 'burritos', 'queso', 'hatch chilies', 'brisket', 'sushi'}
    


### union


```python
# UNION collects all values from each set (no duplicates)
union_set = tx_set.union(co_set)
print(f'All Favorites (both states), NO DUPLICATES\n {union_set}\n')
```

    All Favorites (both states), NO DUPLICATES
     {'wings', 'green chilies', 'bbq', 'burritos', 'queso', 'chili', 'hatch chilies', 'pizza', 'steak', 'brisket', 'mediterranean', 'sushi'}
    


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

    {'wings', 'bbq', 'burritos', 'queso', 'chili', 'hatch chilies', 'burger', 'steak', 'brisket', 'mediterranean'}
    {'wings', 'green chilies', 'breakfast skillet', 'chili', 'buffalo burger', 'pizza', 'steak', 'sushi', 'mediterranean'}



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


## Classes (Object Oriented Programming)
---


```python
class Student:
    
    # init will run when class is called
    def __init__(self, name, grades):
        self.name = name
        self.grades = grades
        
        
    # function inside of class is called a 'method'
    def average(self):
        '''
        average function uses the created student (aka 'self')
        '''
        return sum(self.grades) / len(self.grades)
        
        
# calling the Student class, providing parameters for the class
student = Student('Anakin', (80, 82, 85, 78, 92))
print(student.name)
print(student.grades)
```

    Anakin
    (80, 82, 85, 78, 92)



```python
# calling Student class with new passed parameters, saved as student2
student2 = Student('Obi Wan', (90, 90, 93, 78, 90))
print(student2.name)

# average is a method in the class and requires ()
print(student2.average())
```

    Obi Wan
    88.2


### magic methods


```python
class Person:
    
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
        
    # magic method called automatically when turning the object into a string
    def __str__(self):
        return f'Person: {self.name}, {self.age} years old.\n'
    
    # magic method designed to return a string that can reconstruct the original object
    def __repr__(self):
        return f'<Person({self.name}, {self.age})>'
        
vader = Person('Anakin', 35)

# displays a string representation of the object if __str__ not defined, else return defined __str__
print(vader)

# prints the unambiguous representation of the object
print(vader.__repr__())
```

    Person: 'Anakin', 35 years old.
    
    <Person(Anakin, 35)>


### @classmethod and @staticmethod


```python
class ClassTest:
    
    # instance methods are methods that use the class object self
    def instance_method(self):
        print(f'Called instance_method of {self}')
        
    # classmethod are methods that use the cls object, or the Class object itself
    @classmethod
    def class_method(cls):
        print(f'Called class_method of {cls}')
        
    # static methods are methods that do not have passed parameters self/cls;  
    @staticmethod
    def static_method():
        print('Called static_method.')
```


```python
# saving instance of ClassTest and displaying instance method
test = ClassTest()
test.instance_method()

# used for actions, uses data inside the object created, either manipulating or modifying
```

    Called instance_method of <__main__.ClassTest object at 0x7fcb510245b0>



```python
# python will know to pass ClassTest due to @classmethod decorator
ClassTest.class_method()

# often used to create 'factories'
```

    Called class_method of <class '__main__.ClassTest'>



```python
# essentially a function in the class does not use any parameters
ClassTest.static_method()

# used to place a method inside a class, belongs with the class logically
```

    Called static_method.



```python
class Book:
    # class variables that make sense to go with this class
    TYPES = ('hardcover', 'paperback')
    
    def __init__(self, name, book_type, weight):
        self.name = name 
        self.book_type = book_type
        self.weight = weight
        
    def __str__(self):
        return f'Book: {self.name}, {self.book_type}...weighing {self.weight}g.'
    
    def __repr__(self):
        return f'<Book({self.name}, {self.book_type}, {self.weight})>'

    @classmethod
    def hardcover(cls, name, page_weight):
        
        # creates new Book instance, using first type (hardcover), increasing page weight by 100
        return cls(name, cls.TYPES[0], page_weight + 100)
    
    @classmethod
    def paperback(cls, name, page_weight):
        
        # creates new Book instance, using second type (paperback), keeping original page weight
        return cls(name, cls.TYPES[1], page_weight)
```


```python
print(Book.TYPES)
```

    ('hardcover', 'paperback')



```python
# creating Book instance as book
book = Book('Harry Potter', 'hardcover', 1500)

# displaying the defined __str__ method and __repr__ method strings
print(book)
print(book.__repr__())

# when class instance is called by itself, the __repr__ method defined string will be displayed
book
```

    Book: Harry Potter, hardcover...weighing 1500g.
    <Book(Harry Potter, hardcover, 1500)>





    <Book(Harry Potter, hardcover, 1500)>




```python
# using the classmethod hardcover, creating hardcover instance of HP 
hc_book = Book.hardcover('Harry Potter', 1500)
print(hc_book)
```

    Book: Harry Potter, hardcover...weighing 1600g.



```python
# using the classmethod paperback, creating paperback instance of HP
pb_book = Book.paperback('Harry Potter', 1500)
print(pb_book)
```

    Book: Harry Potter, paperback...weighing 1500g.


# Pandas
---


```python
import pandas as pd

# defines a df
df = pd.DataFrame({'num_legs': [2, 4, 4, 6], 'num_wings': [2, 0, 0, 0]},
                  index=['falcon', 'dog', 'cat', 'ant'])

# stores the value counts of the values in num_legs column
series_frequency = df['num_legs'].value_counts()

# looping over series_frequency, providing a way to display value counts in reports w/ addition info
for value, count in series_frequency.items():
    print(f'There are {count} animal(s) with {value} legs')
```

    There are 2 animal(s) with 4 legs
    There are 1 animal(s) with 6 legs
    There are 1 animal(s) with 2 legs


### Inheritance


```python
class Device:
    
    def __init__(self, name, connected_by):
        self.name = name
        self.connected_by = connected_by
        self.connected = True
        
    def __str__(self):
        # !r is a shorthand way to include quotes around value
        return f'Device {self.name!r} ({self.connected_by})'
    
    def disconnect(self):
        self.connected = False
        print('Disconnected.')
```


```python
printer = Device("Printer", "USB")
print(printer)
printer.disconnect()
```

    Device 'Printer' (USB)
    Disconnected.



```python
# inheritance is assigned by passing the class as a parameter
class Printer(Device):
    
    def __init__(self, name, connected_by, capacity):
        
        # super() calls the init class of the parent class, Device
        super().__init__(name, connected_by)
        self.capacity = capacity
        self.remaining_pages = capacity

    def __str__(self):
        return f'{super().__str__()} ({self.remaining_pages} pages remaining)'
    
    def print(self, pages):
        '''Prints out the # of pages passed, subtracts pages from remaining_pages value.'''
        
        if not self.connected:
            print('Your printer is not connected!')
            return
        
        print(f'Printing {pages} pages.\n')
        self.remaining_pages -= pages
```


```python
# creating instance of PRINTER class, displaying __str__ value
printer = Printer("Printer", "USB", 500)
print(printer)

# running print function in Printer class, again displaying the modified __str__ value 
printer.print(20)
print(printer)

# calling the disconnect function from Device, testing if not self.connected clause
printer.disconnect()
printer.print(30)
```

    Device 'Printer' (USB) (500 pages remaining)
    Printing 20 pages.
    
    Device 'Printer' (USB) (480 pages remaining)
    Disconnected.
    Your printer is not connected!


### Composition - More common than Inheritance


```python
class BookShelf:
    # *books expects to have any number of Book objects passed
    def __init__(self, *books):
        self.books = books
        
    def __str__(self):
        return f'BookShelf with {len(self.books)} books.'
    
class Book:
    
    def __init__(self, name):
        self.name = name
        
    def __str__(self):
        return f'Book {self.name}'
```


```python
# creating two instances of Book Class
book = Book('Harry Potter')
book2 = Book('The Dark Tower')

# with inheritance, the book objects were passed to define the BookShelf
shelf = BookShelf(book, book2)
print(shelf)
```

    BookShelf with 2 books.


## decorators


```python
user = {'username': 'chris', 'access_level': 'guest'}


def get_admin_password():
    return '1234'

def make_secure(func):
    def secure_function():
        if user['access_level'] == 'admin':
            return func()
        else:
            return f"No admin permissions for {user['username']}."
        
    return secure_function
    
get_admin_password = make_secure(get_admin_password)
print(get_admin_password())
```

    No admin permissions for chris.


## decorators; @ syntax


```python
import functools

user = {'username': 'chris', 'access_level': 'admin'}


def make_secure(func):
    
    # the functools.wraps tells secure_fuction it is a wrapper for get_admin_password, and prevents
    # get_admin_password function name/documentation from being replaced w/secure_function name/documentation
    @functools.wraps(func)
    def secure_function():
        if user['access_level'] == 'admin':
            return func()
        else:
            return f"No admin permissions for {user['username']}."
        
    return secure_function

# this @make_secure decorator call will create the function and then pass it through make_secure
@make_secure
def get_admin_password():
    return '1234'

# w/out the @functools.wraps(func) call, this print would display 'secure_function', not 'get_admin_password'
print(get_admin_password.__name__)

print(get_admin_password())
```

    get_admin_password
    1234


### decorating functions w/parameters


```python
import functools
user = {'username': 'chris', 'access_level': 'billing'}

def make_secure(func):
    
    @functools.wraps(func)
    # by providing (*args, **kwargs) the function can take any number of parameters from any function passed to decorator
    def secure_function(*args, **kwargs):

        # if access_level == 'admin' or 'billing', then returns the func(get_password) with parameters  
        if (user['access_level'] == 'admin') | (user['access_level'] == 'billing'):
            return func(*args, **kwargs)

        else:
            return f"No admin permissions for {user['username']}."
        
    return secure_function

@make_secure
def get_password(panel):
    # after running through make_secure, appropriate password is returned
    if panel == 'admin':
        return '1234'
    elif panel == 'billing':
        return 'super_secure_password'

get_password('billing')
```




    'super_secure_password'



## decorators w/parameters


```python
import functools

# make_secure has now access_level parameter passed into function
def make_secure(access_level):
    
    # decorator function does the access level testing; if an access level is passed that @make_secure decorator
    # has as a parameter, continue w/passed function, ELSE print string information.
    def decorator(func):
        
        @functools.wraps(func)
        def secure_function(*args, **kwargs):
            if user['access_level'] == access_level:
                return func(*args, **kwargs)
            else:
                return f"No {access_level} permissions for {user['username']}."

        # secure function is returned by decorator when run
        return secure_function
    # decorator function is returned by make_secure when run
    return decorator
    
# including 'admin' parameter in @make_secure decorator
@make_secure('admin')
def get_admin_password():
    return 'admin: 1234'

# including 'user' parameter in @make_secure decorator
@make_secure('user')
def get_dashboard_password():
    return 'user: user_password'
```


```python
user = {'username': 'Bruce Wayne', 'access_level': 'user'}
print(get_admin_password())
print(get_dashboard_password())
```

    No admin permissions for Bruce Wayne.
    user: user_password



```python
user = {'username': 'Batman', 'access_level': 'admin'}

print(get_admin_password())
print(get_dashboard_password())
```

    admin: 1234
    No user permissions for Batman.


## mutability


```python
a = []
b = a
print(id(b))
print(id(a))
```

    140305482601856
    140305482601856



```python
 
```
