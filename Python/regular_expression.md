| Character Classes / Regex Symbols | Represents                                                                                                           |
| --------------------------------: | :------------------------------------------------------------------------------------------------------------------- |
|                              `\d` | Any numeric digit from 0 to 9.                                                                                       |
|                              `\D` | Any character that is not a numeric digit from 0 to 9.                                                               |
|                              `\w` | Any letter, numeric digit, or the underscore character. (Think of this as matching “word” characters.)               |
|                              `\W` | Any character that is not a letter, numeric digit, or the underscore character.                                      |
|                              `\s` | Any space, tab, or newline character. (Think of this as matching “space” characters.)                                |
|                              `\S` | Any character that is not a space, tab, or newline.                                                                  |
|                               `?` | matches zero or one of the preceding group.                                                                          |
|                               `*` | matches zero or more of the preceding group.                                                                         |
|                               `+` | matches one or more of the preceding group.                                                                          |
|                             `{n}` | matches exactly n of the preceding group.                                                                            |
|                            `{n,}` | matches n or more of the preceding group.                                                                            |
|                            `{,m}` | matches 0 to m of the preceding group.                                                                               |
|                           `{n,m}` | matches at least n and at most m of the preceding group.                                                             |
|              `{n,m}? or *? or +?` | performs a non-greedy match of the preceding group.                                                                  |
|                           `^spam` | means the string must begin with spam.                                                                               |
|                           `spam$` | means the string must end with spam.                                                                                 |
|                               `.` | matches any character, except newline characters.                                                                    |
|   `[abc] or [a-z] or [a-zA-Z0-9]` | matches any character between the brackets (such as a, b, or c).                                                     |
|                          `[^abc]` | matches any character that isn’t between the brackets.                                                               |
|         `re.IGNORECASE` or `re.I` | including in re.compile() as second argument will make regex object case-insensitive                                 |
|                      `re.VERBOSE` | including in re.compile() as second argument will make regex ignore whitespace and comments inside the match pattern |


```python
import re
```


```python
# creates a REGEX object that defines a xxx-xxx-xxxx phone pattern
phone_regex = re.compile(r'\d\d\d-\d\d\d-\d\d\d\d')
```


```python
# using the search() method to search passed string for any matches to phone_regex; stored as match object
match = phone_regex.search('My phone number is 123-867-5309')

print(f'Phone number found as : {mo.group()}')
```

    Phone number found as : 123-867-5309



```python
# including parentheses will create groups w/in the regex object
phone_regex_groups = re.compile(r'(\d\d\d)-(\d\d\d-\d\d\d\d)')

# using search method() w/phone_regex_groups
match = phone_regex_groups.search('My phone number is 123-867-5309')

# using the group() method displays the distinct groups w/in the match object; default passes all groups in match object
print(f'Area Code: {match.group(1)}')
print(f'Phone #: {match.group(2)}')

print(f'Complete Phone #: {match.group()}')
```

    Area Code: 123
    Phone #: 867-5309
    Complete Phone #: 123-867-5309



```python
# the groups() method returns all groups in the match object
area_code, main_number = match.groups()

# using the group() method displays the distinct groups w/in the match object; default passes all groups in match object
print(f'Area Code: {area_code}')
print(f'Phone #: {main_number}')
```

    Area Code: 123
    Phone #: 867-5309



```python
# using a PIPE in regex object allows matching of multiple groups
person_regex = re.compile(r'Spider-man|Peter Parker')

match_1 = person_regex.search('Is Spider-man possibly Peter Parker?')

# the match object will retrieve the first occurence if both values exist in the string
print(match_1.group())
```

    Spider-man



```python
# using parentheses using PIPES will allow several patterns for matching
hero_regex = re.compile(r'(Super|Spider-|Bat|Wonderwo)man')

hero_match = hero_regex.search('Batman is not a real superhero in certain schools of thought.')

# displaying the returned group of the matched text using .group(1); displaying entire match object
print(hero_match.group(1))
print(hero_match.group())
```

    Bat
    Batman



```python
# regex object is greedy; or will choose the longest string possible (ie. 5 Ha's, not 3)
greedy_regex = re.compile(r'(Ha){3,5}')

# the match object returns the longest string
greedy = greedy_regex.search('HaHaHaHaHa')
print(greedy.group())
```

    HaHaHaHaHa



```python
# regex object is non-greedy by including '?' in match pattern  
notgreedy_regex = re.compile(r'(Ha){3,5}?')

# the match object does not always bring the longest possible pattern
notgreedy = notgreedy_regex.search('HaHaHaHaHa')
print(notgreedy.group())
```

    HaHaHa



```python
phones_regex = re.compile(r'\d{3}-\d{3}-\d{4}')

# findall() method returns every match in the searched strings
phones_match = phones_regex.findall('Cell: 123-867-5309 & Work: 912-386-7530')

# findall() returns a list of strings when there are no groups in regex object
phones_match
```




    ['123-867-5309', '912-386-7530']




```python
# same regex object as above but grouped
phones_regex_groups = re.compile(r'(\d{3})-(\d{3})-(\d{4})')

# findall() method returns every match in the searched strings
phones_match = phones_regex_groups.findall('Cell: 123-867-5309 & Work: 912-386-7530')

# findall() returns a list of tuples when groups in regex object
phones_match
```




    [('123', '867', '5309'), ('912', '386', '7530')]




```python
# substituting strings with the sub() method
names_regex = re.compile(r'Agent \w+')

names_regex.sub('CENSORED', 'Agent Alice passed along the documents to Agent Bob')
```




    'CENSORED passed along the documents to CENSORED'




```python
# regex object will obtain first initial using (\w) in match pattern
agent_names_regex = re.compile(f'Agent (\w)\w*')

# including \1, \2, \3, etc. in .sub() will replace string with what is stored in group(1) 
agent_names_regex.sub(r'Agent \1****', 'Agent Alice passed along the documents to Agent Bob')
```




    'Agent A**** passed along the documents to Agent B****'




```python
# using re.VERBOSE to ignore white spaces/comments to better code complex match patterns
phone_regex = re.compile(r'''(
(\d{3}) # area code
(-)     # separator
(\d{3}) # first 3 numbers
(-)     # separator
(\d{4}) # last 4 numbers
)''', re.VERBOSE)

phone_regex.findall('Cell: 123-867-5309 & Work: 912-386-7530')
```




    [('123-867-5309', '123', '-', '867', '-', '5309'),
     ('912-386-7530', '912', '-', '386', '-', '7530')]


