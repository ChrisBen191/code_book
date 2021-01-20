#!/usr/bin/env python
# coding: utf-8

# # Python

# ## Importing Data
# ---

# ### Glob

# In[37]:


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


# ## Iterators
# ---

# ### Range

# In[7]:


# increments are specified using STEP
num_range = range(0, 16, 2)

# converts RANGE OBJECT to a list
nums_list = list(num_range)
print(nums_list)


# ### Enumerate

# In[8]:


# indexing starting at START value specified
enumerate_values = enumerate(['cat', 'dog', 'monkey', 'lemur'], start = 10)

# converts the ENUMERATE OBJECT to a list
indexed_list = list(enumerate_values)

print(indexed_list)


# ### Map

# In[9]:


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


# ## Collections Module
# ---

# ### Counter

# In[10]:


from collections import Counter

pet_list = ['dog', 'dog', 'cat', 'parrot', 'monkey', 'dog', 'frog', 'cat', 'parrot', 'horse',
            'cat', 'dog','hamster', 'horse','snake', 'cat', 'parrot', 'frog', 'hamster', 'snake']

# collects the COUNT of values in pet_list; stores as COUNTER object saved as counter_dict
counter_dict = Counter(pet_list)
print(f'COUNTER Object (dict)\n {counter_dict}\n')

# capturing top value counts (top 3) and saving LIST OF TUPLES as top_3
top_3 = counter_dict.most_common(3)
print(f'Top 3 Popular Pets\n {top_3}')


# ### defaultdict

# In[11]:


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


# ### namedtuple

# In[12]:


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


# In[13]:


# nametuples' attributes are now easily accessible
for joint in bbq_joints:
    print(f'{joint.name} BBQ in {joint.location} has a great {joint.special} special!')


# ## Itertools Module
# ---

# ### Combinations

# In[14]:


from itertools import combinations

colors = ['red', 'blue', 'green']

# '2' indicates NUM OF COMBINATIONS, '*' unpacks the COMBO OBJECT to a list
color_mixes = [ *combinations(colors, 2) ]

color_mixes


# ## Sets
# ---

# In[15]:


tx_favorites = ['steak', 'brisket', 'burritos', 'chili', 'bbq', 'hatch chilies', 'queso', 'mediterranean', 'wings']
co_favorites = ['steak', 'green chilies', 'chili', 'sushi', 'pizza', 'mediterranean', 'wings']

# converting lists into SET objects which only store unique values (no duplicates)
tx_set = set(tx_favorites)
co_set = set(co_favorites)


# ### intersection

# In[16]:


# INTERSECTION collects values shared between the two sets
both_sets = tx_set.intersection(co_set)
print(both_sets)


# ### difference

# In[17]:


#DIFFERENCE collects values only in the set specified
tx_set_only = tx_set.difference(co_set)
co_set_only = co_set.difference(tx_set)

print(f'Only a Texas favorite\n {tx_set_only}\n')
print(f'Only a Colorado favorite\n {co_set_only}')


# ### symmetric difference

# In[18]:


# SYMMETRIC_DIFFERENCE collects values that exist in ONE of the sets, but NOT BOTH
differences = tx_set.symmetric_difference(co_set)
print(f'Either Texas/Colorado Favorite, BUT NOT BOTH\n {differences}\n')


# ### union

# In[19]:


# UNION collects all values from each set (no duplicates)
union_set = tx_set.union(co_set)
print(f'All Favorites (both states), NO DUPLICATES\n {union_set}\n')


# ### set modifiers

# In[20]:


# add a single value to the set only if the value is unique
tx_set.add('burger')

# merges a list of values into the set, only if the values are unique
co_set.update(['buffalo burger', 'breakfast skillet', 'salads'])

# removes an element from the set
co_set.discard('salads')

print(tx_set)
print(co_set)


# In[21]:


# check to see if a POSSIBLE VALUE is a member of a set; faster than list looping
if 'burger' in tx_set:
    print('The set contains the value!')


# ## Lists
# ---

# |            Command             | Behavior                                                                                             |
# | :---------------------------: | -------------------------------------------------------------------------------------------- |
# |    `list.append('value')`     | adds the value specified to the **end** of list                                              |
# | `list.insert(index, 'value')` | **inserts** the value into the list at the index specified                                   |
# |    `list.remove('value')`     | **removes** the element with the corresponding 'value' from the list specified               |
# |     `list.pop(position)`      | **removes** or "pops" the element with the corresponding index value from the list specified |
# |        `list.upper()`         | **uppercases** each string element in the list specified                                     |
# |        `list.lower()`         | **lowercases** each string element in the list specified                                     |
# |        `list.title()`         | **uppercases the initial letter** in each element in the list specified                      |
# |  `list_one.extend(list_two)`  | **appends** list_two to list_one; similar to .append()                                       |
# |     `list.index('value')`     | provides the **position or index** of an item in a list                                      |
# |        `sorted(list)`         | provides the items in a list sorted in **alphabetic/numeric order**, ascending               |
# |     `del list[index_num]`     | deletes the element in the list with the corresponding index_num                             |
# |         `list[:int]`          | slices list from the beginning element of the list to element with **int-1** index           |
# |         `list[int:]`          | slices list from the element with **int** index to the last element in the list              |
# |   `list[start_int:end_int]`   | slices list from element with index **start_num** to the element with **end_num-1** index    |

# ### Zip
# 
# Combines lists into tuples the length of lists passed. Tuples created equals the smallest list passed.

# In[22]:


breakfast = ['cereal', 'scrambled eggs', 'yogurt']
dinner = ['steak', 'cereal', 'burger', 'pasta', 'salmon']

zipped_list = [*zip(breakfast, dinner)]

print(f'Tuple Count equals length of smallest passed list:\n {zipped_list}')


# ## List Comprehensions
# ---

# In[23]:


price_list = [15.99, 7.95, 34.50, 20.20, 11.59]
print(f'Starting Prices:\n {price_list}')


# In[24]:


# determing 15% discount for each item in price list w/list comprehension
discounts = [price * (0.85) for price in price_list]

# formatting each price into $$.$$ format w/list comprehension
discounts = [round(discount, 2) for discount in discounts]
print(f'Price List Discounts:\n {discounts}')


# In[25]:


# filters out elements not meeting conditional clause
cheap = [price for price in discounts if price>=10]
print(f'Cheapest Prices:\n {cheap}')


# In[26]:


# if/else providing string values cooresponding if conditional is met 
price_levels = ['expensive' if price>10 else 'cheap' for price in discounts]

# assume $10 gift card can be applied to only $15 and more...
after_gc = [price-10 if price>=15 else price for price in discounts]
print(f'Prices after GiftCard applied:\n {after_gc}')


# ## Dictionaries
# ---

# |           Command            | Behavior                                                                                                                  |
# | ------------------------- | ----------------------------------------------------------------------------------------------------------------- |
# | `dict[new_key] = new_value` | updates the dict with a new k:v pair denoted by passed key and value                                              |
# | `dict.update(second_dict)`  | updates the dict with a passed dictionary, tuple(s), etc.                                                         |
# |      `dict.pop(value)`      | safely removes the value from the dict if it exists                                                               |
# |   `dict.get(value, 'NA')`   | safely retrieves the value from the dict if it exists, else it returns secondary value (default None)             |
# |        `key in dict`        | boolean statement for if key exists in specified dict; commonly used in conditional statements when parsing dicts |

# ### dictionary

# In[27]:


# creates a dictionary of key:value pairs
meal_dict = {
    'breakfast': 'scrambled eggs',
    'lunch': 'sandwich',
    'dinner': 'steak'
}

# dictionary is accessed by providing the key in brackets
lunch = meal_dict['lunch']
print(f'Lunch Meal: {lunch}')


# ### List of dictionaries

# In[28]:


# creates a list of dictionaries of key:value pairs
list_of_dicts = [
    {'breakfast': 'scrambled eggs'},
    {'lunch': 'sandwich'},
    {'dinner': 'steak'}
]

# dictionary is accessed by providing index of dict in list, then providing the key for that dict
lunch = list_of_dicts[1]['lunch']
print(f'Lunch Meal: {lunch}')


# In[29]:


snack = {'snack': 'cheezits'}

# appends adds the snack dict the list_of_dicts
list_of_dicts.append(snack)
print(list_of_dicts)


# ### Dict of dictionaries

# In[30]:


dict_of_dicts = {
    'breakfast': {'light': 'yogurt', 'medium': 'breakfast sandwich', 'heavy': 'hangover burger'},
    'lunch': {'light': 'sandwich', 'medium': 'pizza', 'heavy': 'fried chicken'},
    'dinner': {'light': 'steak salad', 'medium': 'spagetti', 'heavy': 'steak'}
}

# dictionary is access by providing the key to the nested dict, then providing the key for that dict
lunch = dict_of_dicts['lunch']['light']
print(f'Lunch Meal: {lunch}')


# In[31]:


snacks = {'snacks': {'light': 'cheezeits', 'medium': 'potato chips', 'heavy': 'chocolate cake'}}

# updates dict_of_dicts with snacks dictionary
dict_of_dicts.update(snacks)

for meal_key in dict_of_dicts:
    print(f'Meals for {meal_key}:\n {dict_of_dicts[meal_key]}\n')


# ### Dictionary Modifiers

# In[32]:


sport_dict = {
    'basketball': ['lakers', 'spurs', 'nets'],
    'football': ['cowboys', 'chiefs', 'seahawks'],
}

# conditional to determine if element is inside of football list in sport_dict
if 'cowboys' in sport_dict['football']:
    print('The team is inside the dictionary!')


# In[33]:


# loops through sport_dict to display the sport keys
print('List of Sports: (keys representing each sport)')
for sport in sport_dict.keys():
    print(sport)


# In[34]:


# loops through sport_dict to display the list of teams
print('\nList of Teams: (values for each sport key)')
for teams in sport_dict.values():
    print(teams)


# In[35]:


# loops through sport_dict to display the sport keys and corresponding team values
for sport, teams in sport_dict.items():
    print(f'\n{sport} Teams: \n {teams}')


# ## Functions
# ---

# ### docstring example

# In[38]:


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


# ### unpacking arguments

# In[45]:


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


# ### destructuring arguments into parameters

# In[52]:


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


# ## keyword arguments

# In[65]:


def named(**kwargs):
    """
    Specifying **kwargs will process any number of passed keyword parameters inside dict
    """
    print(kwargs,'\n')
    
named(name='bob', age=25)


# In[67]:


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


# In[72]:


def both(*args, **kwargs):
    """
    this setup is typically used so all arguments can be passed into another function
    """
    print(args, '\n')
    print(kwargs)
    
both(1,2,3,12, name='bob', age=25)


# ## Classes (Object Oriented Programming)
# ---

# In[20]:


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


# In[18]:


# calling Student class with new passed parameters, saved as student2
student2 = Student('Obi Wan', (90, 90, 93, 78, 90))
print(student2.name)

# average is a method in the class and requires ()
print(student2.average())


# ### magic methods

# In[100]:


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


# ### @classmethod and @staticmethod

# In[56]:


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


# In[57]:


# saving instance of ClassTest and displaying instance method
test = ClassTest()
test.instance_method()

# used for actions, uses data inside the object created, either manipulating or modifying


# In[58]:


# python will know to pass ClassTest due to @classmethod decorator
ClassTest.class_method()

# often used to create 'factories'


# In[59]:


# essentially a function in the class does not use any parameters
ClassTest.static_method()

# used to place a method inside a class, belongs with the class logically


# In[121]:


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


# In[122]:


print(Book.TYPES)


# In[123]:


# creating Book instance as book
book = Book('Harry Potter', 'hardcover', 1500)

# displaying the defined __str__ method and __repr__ method strings
print(book)
print(book.__repr__())

# when class instance is called by itself, the __repr__ method defined string will be displayed
book


# In[124]:


# using the classmethod hardcover, creating hardcover instance of HP 
hc_book = Book.hardcover('Harry Potter', 1500)
print(hc_book)


# In[125]:


# using the classmethod paperback, creating paperback instance of HP
pb_book = Book.paperback('Harry Potter', 1500)
print(pb_book)


# In[ ]:




