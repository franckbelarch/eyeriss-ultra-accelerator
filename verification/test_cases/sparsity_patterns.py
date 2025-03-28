"""
Sparsity Pattern Test Cases for Eyeriss Ultra Verification

This module contains sparsity patterns for validating the sparsity exploitation
capabilities of the Eyeriss Ultra accelerator.
"""

import numpy as np
from typing import Dict, List, Tuple

def generate_random_sparse_matrix(shape, sparsity, seed=None):
    """
    Generate a random matrix with the specified sparsity level.
    
    Args:
        shape: Tuple of (rows, cols)
        sparsity: Fraction of zeros in the matrix (0.0 to 1.0)
        seed: Random seed for reproducibility
    
    Returns:
        np.ndarray: Random sparse matrix
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create a dense matrix with random values
    matrix = np.random.randint(-128, 127, size=shape).astype(np.int8)
    
    # Create a mask for zeros
    mask = np.random.random(shape) < sparsity
    
    # Apply the mask
    matrix[mask] = 0
    
    return matrix

def generate_block_sparse_matrix(shape, sparsity, block_size=4, seed=None):
    """
    Generate a matrix with block-structured sparsity.
    
    Args:
        shape: Tuple of (rows, cols)
        sparsity: Fraction of zeros in the matrix (0.0 to 1.0)
        block_size: Size of the sparse blocks
        seed: Random seed for reproducibility
    
    Returns:
        np.ndarray: Block-sparse matrix
    """
    if seed is not None:
        np.random.seed(seed)
    
    rows, cols = shape
    
    # Create a dense matrix with random values
    matrix = np.random.randint(-128, 127, size=shape).astype(np.int8)
    
    # Calculate number of blocks in each dimension
    block_rows = max(1, rows // block_size)
    block_cols = max(1, cols // block_size)
    
    # Calculate number of blocks to zero out
    total_blocks = block_rows * block_cols
    blocks_to_zero = int(total_blocks * sparsity)
    
    # Randomly select blocks to zero out
    block_indices = np.random.choice(total_blocks, blocks_to_zero, replace=False)
    
    # Apply zeros to selected blocks
    for idx in block_indices:
        block_row = idx // block_cols
        block_col = idx % block_cols
        
        row_start = block_row * block_size
        row_end = min(rows, row_start + block_size)
        col_start = block_col * block_size
        col_end = min(cols, col_start + block_size)
        
        matrix[row_start:row_end, col_start:col_end] = 0
    
    return matrix

def generate_channel_sparse_matrix(shape, sparsity, seed=None):
    """
    Generate a matrix with channel-structured sparsity.
    
    Args:
        shape: Tuple of (rows, cols)
        sparsity: Fraction of zeros in the matrix (0.0 to 1.0)
        seed: Random seed for reproducibility
    
    Returns:
        np.ndarray: Channel-sparse matrix (entire columns are zeros)
    """
    if seed is not None:
        np.random.seed(seed)
    
    rows, cols = shape
    
    # Create a dense matrix with random values
    matrix = np.random.randint(-128, 127, size=shape).astype(np.int8)
    
    # Calculate number of columns to zero out
    cols_to_zero = int(cols * sparsity)
    
    # Randomly select columns to zero out
    zero_cols = np.random.choice(cols, cols_to_zero, replace=False)
    
    # Zero out selected columns
    for col in zero_cols:
        matrix[:, col] = 0
    
    return matrix

# Random sparsity test patterns with varying sparsity levels
RANDOM_SPARSITY_PATTERNS = [
    {
        "name": "low_weight_sparsity",
        "description": "Random sparsity pattern with 30% zeros in weights",
        "weights": generate_random_sparse_matrix((64, 64), 0.3, seed=42),
        "inputs": np.random.randint(-128, 127, size=(64, 64)).astype(np.int8),
        "sparsity_level": 0.3,
        "pattern_type": "random"
    },
    {
        "name": "medium_weight_sparsity",
        "description": "Random sparsity pattern with 50% zeros in weights",
        "weights": generate_random_sparse_matrix((64, 64), 0.5, seed=43),
        "inputs": np.random.randint(-128, 127, size=(64, 64)).astype(np.int8),
        "sparsity_level": 0.5,
        "pattern_type": "random"
    },
    {
        "name": "high_weight_sparsity",
        "description": "Random sparsity pattern with 70% zeros in weights",
        "weights": generate_random_sparse_matrix((64, 64), 0.7, seed=44),
        "inputs": np.random.randint(-128, 127, size=(64, 64)).astype(np.int8),
        "sparsity_level": 0.7,
        "pattern_type": "random"
    },
    {
        "name": "very_high_weight_sparsity",
        "description": "Random sparsity pattern with 90% zeros in weights",
        "weights": generate_random_sparse_matrix((64, 64), 0.9, seed=45),
        "inputs": np.random.randint(-128, 127, size=(64, 64)).astype(np.int8),
        "sparsity_level": 0.9,
        "pattern_type": "random"
    }
]

# Structured sparsity test patterns
STRUCTURED_SPARSITY_PATTERNS = [
    {
        "name": "block_sparsity",
        "description": "Block-structured sparsity pattern with 4x4 blocks",
        "weights": generate_block_sparse_matrix((64, 64), 0.5, block_size=4, seed=46),
        "inputs": np.random.randint(-128, 127, size=(64, 64)).astype(np.int8),
        "sparsity_level": 0.5,
        "pattern_type": "block"
    },
    {
        "name": "channel_sparsity",
        "description": "Channel-structured sparsity pattern (entire columns are zeros)",
        "weights": generate_channel_sparse_matrix((64, 64), 0.5, seed=47),
        "inputs": np.random.randint(-128, 127, size=(64, 64)).astype(np.int8),
        "sparsity_level": 0.5,
        "pattern_type": "channel"
    }
]

# Combined activation and weight sparsity patterns
COMBINED_SPARSITY_PATTERNS = [
    {
        "name": "both_sparse_50",
        "description": "Both weights and activations have 50% sparsity",
        "weights": generate_random_sparse_matrix((64, 64), 0.5, seed=48),
        "inputs": generate_random_sparse_matrix((64, 64), 0.5, seed=49),
        "weight_sparsity": 0.5,
        "activation_sparsity": 0.5,
        "pattern_type": "random"
    },
    {
        "name": "both_sparse_70",
        "description": "Both weights and activations have 70% sparsity",
        "weights": generate_random_sparse_matrix((64, 64), 0.7, seed=50),
        "inputs": generate_random_sparse_matrix((64, 64), 0.7, seed=51),
        "weight_sparsity": 0.7,
        "activation_sparsity": 0.7,
        "pattern_type": "random"
    }
]

# Special test patterns for verification
SPECIAL_TEST_PATTERNS = [
    {
        "name": "alternating_zeros",
        "description": "Alternating zero pattern in both dimensions",
        "weights": np.indices((64, 64)).sum(axis=0) % 2 * np.random.randint(1, 127, size=(64, 64)).astype(np.int8),
        "inputs": np.random.randint(-128, 127, size=(64, 64)).astype(np.int8),
        "sparsity_level": 0.5,
        "pattern_type": "special"
    },
    {
        "name": "checkerboard",
        "description": "Checkerboard pattern of zeros",
        "weights": ((np.indices((64, 64)).sum(axis=0) % 2) ^ 1) * np.random.randint(1, 127, size=(64, 64)).astype(np.int8),
        "inputs": np.random.randint(-128, 127, size=(64, 64)).astype(np.int8),
        "sparsity_level": 0.5,
        "pattern_type": "special"
    }
]

# Collection of all sparsity test patterns
ALL_SPARSITY_PATTERNS = (
    RANDOM_SPARSITY_PATTERNS +
    STRUCTURED_SPARSITY_PATTERNS +
    COMBINED_SPARSITY_PATTERNS +
    SPECIAL_TEST_PATTERNS
)

def analyze_sparsity_pattern(matrix):
    """Analyze a sparsity pattern and return statistics."""
    total_elements = matrix.size
    zero_elements = np.sum(matrix == 0)
    sparsity = zero_elements / total_elements
    
    return {
        "total_elements": total_elements,
        "zero_elements": zero_elements,
        "sparsity": sparsity,
        "shape": matrix.shape
    }

if __name__ == "__main__":
    print(f"Loaded {len(ALL_SPARSITY_PATTERNS)} sparsity test patterns")
    
    for i, pattern in enumerate(ALL_SPARSITY_PATTERNS):
        stats = analyze_sparsity_pattern(pattern["weights"])
        print(f"{i+1}. {pattern['name']}: {pattern['description']}")
        print(f"   Actual sparsity: {stats['sparsity']:.2f} ({stats['zero_elements']} zeros of {stats['total_elements']} elements)")
        print()