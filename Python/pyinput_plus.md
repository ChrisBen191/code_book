# PyInputPlus Module

|           Method           | Behavior                                                                                                          |
| -------------------------:   | :----------------------------------------------------------------------------------------------------------------- |
| `inputStr()` | similar to built-in input() function, but has additional features and can pass a custom validation function into it|
| `inputNum()`  | ensures user enters a number and returns int, float |
| `inputChoice()`  | ensures the users enters one of the provided choices|
| `inputMenu()`  | similar to inputChoice(), but provides menu w/numbered or lettered options |
| `inputDatetime()`  | ensures the user enters a date and time |
| `inputYesNo()`  | ensure the user enters a yes/no response|
| `inputBool()`  | similar to inputYesNo(), but takes a true/false response and returns a Boolean value|
| `inputEmail()`  | ensures the user enters a valid email address|
| `inpuFilepath()`  | ensures the user enters a valid file path and filename; can validate a file with that name exists|
| `inputPassword()`  | like built-in input() function, but displays * characters as the user types so the password isn't displayed|


```python
import pyinputplus as pyip
```


```python
# including min, max, greaterThan, lessThan keywords w/can specify a range of valid values 
response = pyip.inputNum('Enter number:', min=4)
```

    Enter number:

     3


    Number must be at minimum 4.
    Enter number:

     5



```python
#### using blank=True allows no responce to be recorded; blank=False by default
pyip.inputStr('Enter name:', blank=True)
```

    Enter name:

     





    ''




```python
# using limit keyword limits the attempts before raising RetryLimitException error
pyip.inputNum('Enter number:', limit=2)
```

    Enter number:

     hello


    'hello' is not a number.
    Enter number:

     1





    1




```python
# using timeout keyword limits the seconds to input before raising TimeoutException error;
# using default keyword provides a value instead of raising RetryLimitException error
pyip.inputStr('Enter name:', timeout=10, default='N/A')
```

    Enter name:

     ole yeller





    'ole yeller'




```python
# allowRegexes keyword take a list of regex strings to determine what is valid input
response = pyip.inputNum('Enter number:', allowRegexes=[r'(I|V|X|L|C|D|M)', r'zero'])
```

    Enter number:

     zero



```python
# allowRegexes keyword take a list of regex strings to determine what is valid input;
# if using both allowRegexes and blockRegexes the ALLOW list overrides the BLOCK list
response = pyip.inputNum('Enter number:', blockRegexes=[r'[02468]$'])
```

    Enter number:

     78


    This response is invalid.
    Enter number:

     77



```python
pyip.inputMenu(['Red', 'Blue', 'Green', 'Purple'], lettered=True)
```

    Please select one of the following:
    A. Red
    B. Blue
    C. Green
    D. Purple


     Purple





    'Purple'




```python
# custom function taking values adding up to 10
def addsUpToTen(numbers):
    
    numbersList = list(numbers)
    
    for i, digit in enumerate(numbersList):
        numbersList[i] = int(digit)
        
    if sum(numbersList) != 10:
    
        raise Exception(f'The digits must add up to 10, not {sum(numbersList)}\n')
        
    return int(numbers)


# running the inputCustom() method w/custom function addsUpToTen passed
response = pyip.inputCustom(addsUpToTen, prompt='Input numbers adding up to 10:')
```

    Input numbers adding up to 10:

     123


    The digits must add up to 10, not 6
    
    Input numbers adding up to 10:

     1234

