+++
title = "How to cache your functions the right way?"
date = "2024-08-04"
category = "treatise"
tags = [
    "python", "CS", 
]
+++

"Benchmarking"—a word that either kicks off an experiment or follows a bold new proposal. But when you're dealing with massive datasets, the benchmarking and evaluation functions can take their sweet time. What's a reasonable person to do? Simple: hit run, grab a coffee, lunch, or, if you're really daring, take a nap. Yet, the horror of returning to find your entire experiment crashed because one row was invalid, an API hit its rate limit, or some sneaky corner case reared its ugly head is all too real.

Once bitten, twice shy—we scramble for makeshift solutions, like saving outputs to files and reloading them. Sure, it works, but doing it for each new experiment or API, each with its unique data quirks, quickly turns into a chore. And let's not even get started on the rampant chaos when dealing with asynchronous functions and multiple coroutines.

Here's where things get interesting: in many of these scenarios, both inputs and outputs are immutable, opening the door for a smarter, sleeker way to generalize our caching, no matter the data type. Enter Python's decorators, they allow us to add caching functionality to our functions without mucking around with their insides in an elegant way.

In this short write up, we'll build a caching decorator from scratch, gradually improving it to handle more complex scenarios.

## Version 1: Basic Caching with a Dictionary

Let's start with a simple caching decorator that stores results in a dictionary:

```python
from functools import wraps

def cache(func):
    cache_dict = {}
    
    @wraps(func)
    def wrapper(*args):
        if args in cache_dict:
            return cache_dict[args]
        result = func(*args)
        cache_dict[args] = result
        return result
    
    return wrapper

# Usage example
@cache
def expensive_function(x, y):
    # Simulate an expensive operation
    import time
    time.sleep(2)
    return x + y

print(expensive_function(2, 3))  # Takes 2 seconds
print(expensive_function(2, 3))  # Returns immediately
```

This basic version works well for simple cases, but it has some limitations:

1. It only works with positional arguments.
2. The cache persists only for the lifetime of the program.
3. There's no way to bypass the cache if needed.
4. It doesn't support asynchronous functions.

Let's address these issues one by one.

## Version 2: Supporting Both Args and Kwargs

To support both positional and keyword arguments, we need to create a cache key that includes both:

```python
import json

def cache(func):
    cache_dict = {}
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create a cache key from both args and kwargs
        key = json.dumps((args, sorted(kwargs.items())), sort_keys=True)
        
        if key in cache_dict:
            return cache_dict[key]
        
        result = func(*args, **kwargs)
        cache_dict[key] = result
        return result
    
    return wrapper
```

Now our decorator supports both types of arguments. However, the cache still doesn't persist between program runs, and we might want to limit the number of cached entries to prevent memory issues.

## Version 3: Persistent Cache with Entry Limit

Let's improve our decorator to store the cache in a file and limit the number of entries:

```python
import json
import os

def cache(entries_limit=5):
    def decorator(func):
        cache_file = f"{func.__name__}_cache.json"
        cache_dict = {}
        new_entries = 0

        # Load cache from file if it exists
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache_dict = json.load(f)

        def save_cache():
            with open(cache_file, 'w') as f:
                json.dump(cache_dict, f)

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal new_entries
            
            key = json.dumps((args, sorted(kwargs.items())), sort_keys=True)
            
            if key in cache_dict:
                return cache_dict[key]
            
            result = func(*args, **kwargs)
            cache_dict[key] = result
            new_entries += 1
            
            # Save cache every entries_limit new entries
            if new_entries >= entries_limit:
                save_cache()
                new_entries = 0
            
            return result

        return wrapper
    return decorator
```

This version addresses the persistence issue and adds an entry limit. However, there might be cases where we want to bypass the cache and force a recalculation.

## Version 4: Allowing Cache Bypass

Let's add an option to bypass the cache:

```python
def cache(entries_limit=5):
    def decorator(func):
        cache_file = f"{func.__name__}_cache.json"
        cache_dict = {}
        new_entries = 0

        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache_dict = json.load(f)

        def save_cache():
            with open(cache_file, 'w') as f:
                json.dump(cache_dict, f)

        @wraps(func)
        def wrapper(*args, _bypass_cache=False, **kwargs):
            nonlocal new_entries
            
            # If _bypass_cache is True, run the function without caching
            if _bypass_cache:
                return func(*args, **kwargs)
            
            # Create a cache key from both args and kwargs
            # Exclude _bypass_cache from the key
            key = json.dumps((args, sorted((k, v) for k, v in kwargs.items() if k != '_bypass_cache')), sort_keys=True)
            
            if key in cache_dict:
                return cache_dict[key]
            
            result = func(*args, **kwargs)
            cache_dict[key] = result
            new_entries += 1
            
            if new_entries >= entries_limit:
                save_cache()
                new_entries = 0
            
            return result

        return wrapper
    return decorator
```

Now we can bypass the cache when needed by passing `_bypass_cache=True`. The last major improvement we can make is to support asynchronous functions.

## Version 5: Supporting Async Functions

To support async functions, we need to modify our decorator to work with coroutines:

```python
import asyncio

def cache(entries_limit=5):
    def decorator(func):
        cache_file = f"{func.__name__}_cache.json"
        cache_dict = {}
        new_entries = 0

        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache_dict = json.load(f)

        def save_cache():
            with open(cache_file, 'w') as f:
                json.dump(cache_dict, f)

        @wraps(func)
        async def wrapper(*args, _bypass_cache=False, **kwargs):
            nonlocal new_entries
            
            if _bypass_cache:
                return await func(*args, **kwargs)
            
            key = json.dumps((args, sorted((k, v) for k, v in kwargs.items() if k != '_bypass_cache')), sort_keys=True)
            
            if key in cache_dict:
                return cache_dict[key]
            
            result = await func(*args, **kwargs)
            cache_dict[key] = result
            new_entries += 1
            
            if new_entries >= entries_limit:
                save_cache()
                new_entries = 0
            
            return result

        return wrapper
    return decorator

# Usage example
@cache(entries_limit=5)
async def expensive_async_function(x, y):
    await asyncio.sleep(2)  # Simulate an expensive async operation
    return x + y

# Run the async function
async def main():
    print(await expensive_async_function(2, 3))  # Takes 2 seconds
    print(await expensive_async_function(2, 3))  # Returns immediately
    print(await expensive_async_function(2, 3, _bypass_cache=True))  # Takes 2 seconds again

asyncio.run(main())
```

## Conclusion

We've built a powerful caching decorator that:

1. Supports both positional and keyword arguments
2. Persists the cache to a file
3. Limits the number of cached entries
4. Allows bypassing the cache when needed
5. Supports both synchronous and asynchronous functions

This decorator can significantly improve the performance of your Python applications by reducing redundant computations. Remember to use it judiciously, as caching isn't always beneficial, especially for functions with rapidly changing outputs or side effects.

Feel free to adapt this decorator to your specific needs, and happy coding!

