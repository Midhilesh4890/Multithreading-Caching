import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ProcessPoolExecutor
import time
import logging

# Configure the logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

train_cols = ['T' + str(i) for i in range(1, 11)]
train = pd.DataFrame(np.random.rand(100, 10), columns=train_cols)

rest_cols = ['R' + str(i) for i in range(1, 11)]
rest = pd.DataFrame(np.random.rand(100000, 10), columns=rest_cols)

train['row'] = range(1, len(train) + 1)
rest['row'] = range(1, len(rest) + 1)

A = train.values

bin_size = 1000
max_size = rest.shape[0]

def generate_ranges(bin_size, max_size):
    ranges = []
    for i in range(max_size // bin_size):
        start = i * bin_size + 1
        end = min((i + 1) * bin_size, max_size)
        ranges.append((start, end))

    if max_size % bin_size != 0:
        start = (max_size // bin_size) * bin_size + 1
        end = max_size
        ranges.append((start, end))

    return ranges

def compute_similarity(tup):
    start, end = tup
    B = rest[(rest['row'] >= start) & (rest['row'] <= end)].values
    similarities = cosine_similarity(A, B)
    logger.info(f"Computed similarities for rows {start}-{end}")
    return similarities.tolist()

if __name__ == '__main__':
    # Record the start time
    start_time = time.time()

    # Create an empty list to store the similarities
    similarities_list = []
    RANGES = generate_ranges(bin_size, max_size)
    
    # Create a concurrent.futures ProcessPoolExecutor
    num_processes = 4  # You can adjust the number of processes as needed
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        similarities_list = list(executor.map(compute_similarity, RANGES))
        logger.info("Finished computing similarities in parallel")
    
    # Create a DataFrame from the list of similarities
    similarities_df = pd.DataFrame(similarities_list, columns=train.index)

    # Record the end time
    end_time = time.time()

    # Calculate and log the execution time
    execution_time = end_time - start_time
    logger.info(f"Execution time: {execution_time} seconds")
