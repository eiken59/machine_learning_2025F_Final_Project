import numpy as np
from numba import njit
import os

# ==============================================================================
# 1. Numba-Accelerated Core (Bin Packing Algorithm)
#    Performs all mathematical operations at the C-level to avoid Python overhead.
# ==============================================================================
@njit
def bfd(effective_sizes, s, sigma, capacity, rand_noise):
    """
    Executes the Best-Fit Decreasing (BFD) heuristic and calculates overflow 
    within a single Numba-compiled function for maximum performance.
    
    Args:
        effective_sizes (np.array): The 'planned' sizes (nominal + padding) used for bin assignment.
        s (np.array): Nominal sizes of items.
        sigma (np.array): Volatility of items.
        capacity (float): Capacity of each bin.
        rand_noise (np.array): Pre-generated standard normal noise for realization.
        
    Returns:
        bin_count (int): Total number of bins used.
        total_overflow (float): Total overflow amount across all bins.
    """
    n = len(effective_sizes)
    
    # --- A. Planning Phase (BFD) ---
    # Numba does not support reverse argsort efficiently, so we sort ascending
    # and iterate backwards.
    # Note: We must sort s and sigma using the same indices to keep data aligned.
    sort_idx = np.argsort(effective_sizes)
    
    # Use sorted indices to rearrange arrays without creating unnecessary copies
    sorted_items = effective_sizes[sort_idx]
    s_sorted = s[sort_idx]
    sigma_sorted = sigma[sort_idx]
    
    # Pre-allocate memory for bin usage tracking
    # bins_usage: Tracks the 'planned' usage based on effective sizes
    bins_usage = np.zeros(n, dtype=np.float64) 
    
    # real_bin_usage: Tracks the 'actual' usage based on realized sizes
    # We calculate this on-the-fly to avoid storing an item-to-bin mapping array
    real_bin_usage = np.zeros(n, dtype=np.float64)
    
    bin_count = 0
    
    # Iterate through items from largest to smallest (Decreasing order)
    for i in range(n - 1, -1, -1):
        size = sorted_items[i]
        
        # --- B. Realization Phase (Calculate Actual Duration) ---
        # Calculate the realized size using the Log-Normal distribution formula:
        # Real_Size = exp(mu + sigma * Z), where mu = log(nominal_s)
        # We use pre-generated noise (rand_noise) for speed and reproducibility.
        mu = np.log(s_sorted[i])
        real_size = np.exp(mu + sigma_sorted[i] * rand_noise[i])
        
        best_bin_idx = -1
        min_slack = 10000.0
        
        # Search existing bins for the "Best Fit"
        # (The bin that leaves the least remaining space after adding the item)
        for b in range(bin_count):
            if bins_usage[b] + size <= capacity:
                slack = capacity - (bins_usage[b] + size)
                if slack < min_slack:
                    min_slack = slack
                    best_bin_idx = b
        
        # Assign item to the best bin found
        if best_bin_idx != -1:
            bins_usage[best_bin_idx] += size
            real_bin_usage[best_bin_idx] += real_size # Accumulate actual size
        else:
            # If no bin fits, open a new bin
            bins_usage[bin_count] += size
            real_bin_usage[bin_count] += real_size # Accumulate actual size
            bin_count += 1
            
    # --- C. Calculate Overflow ---
    # Sum up the overflow from all bins that exceeded capacity
    total_overflow = 0.0
    for b in range(bin_count):
        if real_bin_usage[b] > capacity:
            total_overflow += (real_bin_usage[b] - capacity)
            
    return bin_count, total_overflow

# ==============================================================================
# 2. Worker Function (Interface for Multiprocessing)
# ==============================================================================
def worker_simulate_episode(args):
    """
    Worker function to simulate a single episode.
    Designed to be mapped across multiple CPU cores.
    
    Args:
        args (tuple): Contains (s, sigma, padding, capacity, lambda_penalty, seed)
    """
    s, sigma, padding, capacity, lambda_penalty, seed = args
    
    # 1. Set Random Seed & Generate Noise
    # We generate random noise in Python (NumPy) which is fast, 
    # and pass it to Numba. This ensures we can control the seed per worker.
    rng = np.random.default_rng(seed)
    rand_noise = rng.standard_normal(len(s))
    
    # 2. Calculate Effective Sizes (Nominal + Padding)
    effective_sizes = s + padding
    
    # 3. Call the Numba-optimized core function
    num_bins, total_overflow = bfd(effective_sizes, s, sigma, capacity, rand_noise)
            
    return num_bins, total_overflow

# ==============================================================================
# 3. Data Generator
# ==============================================================================
class AirportDataGen:
    """
    Generates synthetic flight data following a Log-Normal distribution.
    """
    def __init__(self, capacity=1.0):
        self.capacity = capacity

    def generate_batch_data(self, batch_size, num_items):
        """
        Generates a batch of flight data.
        
        Returns:
            s (np.array): Nominal sizes (scheduled duration).
            sigma (np.array): Volatility (uncertainty).
        """
        # Nominal sizes: Uniformly distributed between 0.1 and 0.25 of bin capacity
        s = np.random.uniform(0.1, 0.25, size=(batch_size, num_items))
        
        # Volatility: Correlated with size (larger flights -> more volatile)
        # plus some random noise.
        sigma = np.random.uniform(0.01, 0.2, size=(batch_size, num_items)) + (s * 0.15)
        
        return s, sigma