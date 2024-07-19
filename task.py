#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1
class MaxHeap:
    def __init__(self):
        self.heap = []

    def insert(self, val):
        self.heap.append(val)
        self._heapify_up(len(self.heap) - 1)

    def delete(self):
        if len(self.heap) == 0:
            return None
        if len(self.heap) == 1:
            return self.heap.pop()
        root = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down(0)
        return root

    def get_max(self):
        if len(self.heap) == 0:
            return None
        return self.heap[0]

    def _heapify_up(self, index):
        parent_index = (index - 1) // 2
        if parent_index >= 0 and self.heap[index] > self.heap[parent_index]:
            self.heap[index], self.heap[parent_index] = self.heap[parent_index], self.heap[index]
            self._heapify_up(parent_index)

    def _heapify_down(self, index):
        left_child_index = 2 * index + 1
        right_child_index = 2 * index + 2
        largest = index

        if left_child_index < len(self.heap) and self.heap[left_child_index] > self.heap[largest]:
            largest = left_child_index

        if right_child_index < len(self.heap) and self.heap[right_child_index] > self.heap[largest]:
            largest = right_child_index

        if largest != index:
            self.heap[index], self.heap[largest] = self.heap[largest], self.heap[index]
            self._heapify_down(largest)


heap = MaxHeap()
heap.insert(10)
heap.insert(20)
heap.insert(15)

print(heap.get_max())  

print(heap.delete()) 
print(heap.get_max())  


# In[2]:


#10
def transpose_matrix(matrix):

    if not matrix:
        return []

    
    rows = len(matrix)
    columns = len(matrix[0])
    
    
    transposed = []
    
    
    for c in range(columns):
    
        new_row = []
        for r in range(rows):
        
            new_row.append(matrix[r][c])
        
        transposed.append(new_row)
    
    return transposed


original_matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

transposed_matrix = transpose_matrix(original_matrix)
print(transposed_matrix)


# In[3]:


#9
import random
import string

def generate_random_password(length=12):
    
    lowercase = string.ascii_lowercase
    uppercase = string.ascii_uppercase
    digits = string.digits
    special_characters = string.punctuation
    
    
    password = [
        random.choice(lowercase),
        random.choice(uppercase),
        random.choice(digits),
        random.choice(special_characters)
    ]
    
    
    if length > 4:
        all_characters = lowercase + uppercase + digits + special_characters
        password += random.choices(all_characters, k=length-4)
    
    
    random.shuffle(password)
    
    
    return ''.join(password)


password = generate_random_password(12)
print("Generated Password:", password)


# In[4]:


#8
def calculate(num1, num2, operator):
    """
    Perform arithmetic operation based on the given operator.
    
    Parameters:
    num1 (float): The first number.
    num2 (float): The second number.
    operator (str): The operator, which can be '+', '-', '*', or '/'.
    
    Returns:
    float: The result of the arithmetic operation.
    str: An error message if the operator is invalid or division by zero occurs.
    """
    
    if operator == '+':
        return num1 + num2
    elif operator == '-':
        return num1 - num2
    elif operator == '*':
        return num1 * num2
    elif operator == '/':
        if num2 == 0:
            return "Error: Division by zero is not allowed."
        return num1 / num2
    else:
        return "Error: Invalid operator. Please use '+', '-', '*', or '/'."


result1 = calculate(10, 5, '+')
print("10 + 5 =", result1)

result2 = calculate(10, 5, '-')
print("10 - 5 =", result2)

result3 = calculate(10, 5, '*')
print("10 * 5 =", result3)

result4 = calculate(10, 5, '/')
print("10 / 5 =", result4)

result5 = calculate(10, 0, '/')
print("10 / 0 =", result5)

result6 = calculate(10, 5, '%')
print("10 % 5 =", result6)


# In[5]:


#6
def safe_divide(dividend, divisor):
    """
    Divide two numbers and handle division by zero.
    
    Parameters:
    dividend (float): The number to be divided.
    divisor (float): The number by which to divide.
    
    Returns:
    float: The result of the division if no error occurs.
    str: A custom error message if division by zero occurs.
    """
    try:
        
        result = dividend / divisor
    except ZeroDivisionError:
        
        return "Error: Division by zero is not allowed."
    else:
        
        return result


result1 = safe_divide(10, 2)
print("10 / 2 =", result1)

result2 = safe_divide(10, 0)
print("10 / 0 =", result2)


# In[6]:


#5
def fibonacci(n):
    """
    Compute the nth Fibonacci number using recursion.
    
    Parameters:
    n (int): The position in the Fibonacci sequence (0-based).
    
    Returns:
    int: The nth Fibonacci number.
    """
    
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        
        return fibonacci(n - 1) + fibonacci(n - 2)


n = 10
result = fibonacci(n)
print(f"The {n}th Fibonacci number is: {result}")


# In[8]:


#4
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def clean_and_preprocess(df):
    """
    Clean and preprocess the given DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to be cleaned and preprocessed.
    
    Returns:
    pd.DataFrame: The cleaned and preprocessed DataFrame.
    """
    
    df = df.dropna()
    

    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    
    scaler = MinMaxScaler()
    
    
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    
    encoder = OneHotEncoder(sparse=False, drop='first')
    
    
    encoded_features = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))
    
    
    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, encoded_df], axis=1)
    
    return df


data = {
    'age': [25, 30, 45, None, 22],
    'income': [50000, 60000, 80000, 120000, None],
    'gender': ['Male', 'Female', 'Female', 'Male', 'Female'],
    'occupation': ['Engineer', 'Doctor', 'Artist', 'Engineer', 'Artist']
}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

cleaned_df = clean_and_preprocess(df)
print("\nCleaned and Preprocessed DataFrame:")
print(cleaned_df)


# In[12]:


#3
from sklearn.datasets import fetch_california_housing
import pandas as pd


california = fetch_california_housing()


california_df = pd.DataFrame(california.data, columns=california.feature_names)
california_df['target'] = california.target


print(california_df.head())
import pandas as pd


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"


column_names = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", 
    "PTRATIO", "B", "LSTAT", "MEDV"
]


boston_df = pd.read_csv(url, delim_whitespace=True, names=column_names)


print(boston_df.head())


# In[14]:


#7
import time
import functools

def measure_execution_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        
        start_time = time.time()
        
        
        result = func(*args, **kwargs)
        
        
        end_time = time.time()
        
        
        elapsed_time = end_time - start_time
        
        
        print(f"Execution of '{func.__name__}' took {elapsed_time:.4f} seconds")
        
        return result
    
    return wrapper

@measure_execution_time
def perform_computation():
    
    total = 0
    for i in range(1, 10000000):
        total += i
    return total


result = perform_computation()


print("Result:", result)


# In[16]:


#2
import requests
from requests.exceptions import HTTPError, Timeout, RequestException

def download_content(urls):
    """
    Download content from a list of URLs, retrying up to 3 times if an error occurs.
    
    Parameters:
    urls (list): List of URLs to download content from.
    
    Returns:
    dict: A dictionary with URLs as keys and their content or error messages as values.
    """
    results = {}

    for url in urls:
        attempts = 0
        success = False
        
        while attempts < 3 and not success:
            try:
                response = requests.get(url, timeout=5)
                response.raise_for_status()  
                results[url] = response.text
                success = True
            except HTTPError as http_err:
                results[url] = f"HTTP error occurred: {http_err}"
            except Timeout as timeout_err:
                results[url] = f"Timeout error occurred: {timeout_err}"
            except RequestException as req_err:
                results[url] = f"Request exception occurred: {req_err}"
            except Exception as err:
                results[url] = f"An error occurred: {err}"
            
            if not success:
                attempts += 1
                if attempts < 3:
                    print(f"Retrying ({attempts}/3) for URL: {url}")
                else:
                    print(f"Failed to download content from URL: {url} after 3 attempts")

    return results


urls = [
    "https://jsonplaceholder.typicode.com/posts",
    "https://jsonplaceholder.typicode.com/comments",
    "https://jsonplaceholder.typicode.com/albums",
    "https://jsonplaceholder.typicode.com/photos",
    "https://jsonplaceholder.typicode.com/todos",
    "https://jsonplaceholder.typicode.com/users",
    "https://invalid-url.com"
]


results = download_content(urls)


for url, content in results.items():
    print(f"URL: {url}\nContent: {content[:100]}...\n")


# In[ ]:





# In[ ]:




