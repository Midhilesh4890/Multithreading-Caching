import os
import time
import glob
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from cachetools import cached, TTLCache

def read_data1(arr):
    s = []
    for i in arr:
        s.append(len(pd.read_csv(i)))

def read_data2(file):
    s = []
    s.append(len(pd.read_csv(file)))
    return s

@cached(cache = TTLCache(maxsize=10, ttl=3600))
def read_data3(file):
    s = []
    s.append(len(pd.read_csv(file)))
    return s

if __name__ == '__main__':
    files = [r'C:\Users\Dell\Datasets\1.csv',r'C:\Users\Dell\Datasets\2.csv']
    normal = 0
    multithreading = 0
    multithreading_caching = 0

    for i in range(1,21):
        start = time.time()
        read_data1(files)
        end = time.time()
        normal += (end - start)

        start = time.time()
        with ThreadPoolExecutor(5) as executor:
            result = executor.map(read_data2,files)   
        end = time.time()
        multithreading += (end - start)

        start = time.time()
        with ThreadPoolExecutor(5) as executor:
            result = executor.map(read_data3,files)
        end = time.time()
        multithreading_caching += (end - start)

    print(f'Lengths of datasets : {[len(pd.read_csv(i)) for i in files]}')    
    print(f'No.of iterations :{i}')
    print(f'Avg time using Normal Method :{normal/i}')
    print(f'Avg time using Multithreading :{multithreading/i}')
    print(f'Avg time using Multithreading & Caching :{multithreading_caching/i}')