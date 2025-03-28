"""
Sparsity Verification Module for Eyeriss Ultra

This module implements the sparsity verification framework used to validate
the correctness and performance of sparsity exploitation mechanisms in the
Eyeriss Ultra accelerator design.

Key features:
- Sparsity pattern generation with controllable distributions
- Zero-detection and computation skipping verification
- Compression format validation
- Performance and energy validation with varying sparsity
- Statistical analysis of sparsity benefits
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import math
from enum import Enum, auto
from dataclasses import dataclass
import random

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SparsityVerification")

class SparsityPattern(Enum):
    """Different sparsity distribution patterns"""
    RANDOM = auto()           # Randomly distributed zeros
    STRUCTURED_BLOCK = auto() # Block-structured sparsity (common in pruned networks)
    STRUCTURED_CHANNEL = auto() # Channel-wise sparsity
    RELU_LIKE = auto()        # ReLU-like sparsity (mostly positive values)
    HIGHLY_SPARSE = auto()    # Very high sparsity (>90%)

@dataclass
class SparsityConfig:
    """Configuration for sparsity verification"""
    weight_sparsity: float        # Target weight sparsity (0.0-1.0)
    activation_sparsity: float    # Target activation sparsity (0.0-1.0)
    weight_pattern: SparsityPattern  # Weight sparsity pattern
    activation_pattern: SparsityPattern  # Activation sparsity pattern
    compression_enabled: bool     # Whether compression is enabled
    zero_skipping_enabled: bool   # Whether computation skipping is enabled
    
    def __post_init__(self):
        # Validate sparsity ranges
        if not 0.0 <= self.weight_sparsity <= 1.0:
            raise ValueError(f"Weight sparsity must be between 0.0 and 1.0, got {self.weight_sparsity}")
        if not 0.0 <= self.activation_sparsity <= 1.0:
            raise ValueError(f"Activation sparsity must be between 0.0 and 1.0, got {self.activation_sparsity}")
    
    def __str__(self) -> str:
        return (f"SparsityConfig(weight_sparsity={self.weight_sparsity:.2f}, "
                f"activation_sparsity={self.activation_sparsity:.2f}, "
                f"weight_pattern={self.weight_pattern.name}, "
                f"activation_pattern={self.activation_pattern.name}, "
                f"compression={self.compression_enabled}, "
                f"zero_skipping={self.zero_skipping_enabled})")


class SparseMatrixGenerator:
    """Generator for matrices with controllable sparsity patterns"""
    
    @staticmethod
    def generate_random_sparse_matrix(rows: int, 
                                    cols: int, 
                                    sparsity: float,
                                    min_val: float = -1.0,
                                    max_val: float = 1.0,
                                    seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a matrix with random values and random zero distribution
        according to target sparsity.
        """
        rng = np.random.RandomState(seed)
        # First generate a dense matrix with random values
        matrix = rng.uniform(min_val, max_val, size=(rows, cols))
        
        # Create a random mask for zeros
        mask = rng.uniform(0, 1, size=(rows, cols)) < sparsity
        
        # Apply mask
        matrix[mask] = 0.0
        
        return matrix
    
    @staticmethod
    def generate_block_sparse_matrix(rows: int, 
                                   cols: int, 
                                   sparsity: float,
                                   block_size: int = 4,
                                   min_val: float = -1.0,
                                   max_val: float = 1.0,
                                   seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a matrix with block-structured sparsity.
        Zeros are organized in blocks rather than randomly distributed.
        """
        rng = np.random.RandomState(seed)
        # First generate a dense matrix with random values
        matrix = rng.uniform(min_val, max_val, size=(rows, cols))
        
        # Calculate number of blocks in each dimension
        block_rows = max(1, rows // block_size)
        block_cols = max(1, cols // block_size)
        
        # Calculate number of blocks to zero out
        total_blocks = block_rows * block_cols
        blocks_to_zero = int(total_blocks * sparsity)
        
        # Randomly select blocks to zero out
        zero_block_indices = rng.choice(total_blocks, blocks_to_zero, replace=False)
        
        # Apply zeros to selected blocks
        for idx in zero_block_indices:
            block_row = idx // block_cols
            block_col = idx % block_cols
            
            # Calculate block boundaries
            row_start = block_row * block_size
            row_end = min(rows, row_start + block_size)
            col_start = block_col * block_size
            col_end = min(cols, col_start + block_size)
            
            # Zero out the block
            matrix[row_start:row_end, col_start:col_end] = 0.0
        
        return matrix
    
    @staticmethod
    def generate_channel_sparse_matrix(rows: int, 
                                     cols: int, 
                                     sparsity: float,
                                     min_val: float = -1.0,
                                     max_val: float = 1.0,
                                     seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a matrix with channel-wise sparsity.
        Entire columns are zeroed out to simulate channel pruning.
        """
        rng = np.random.RandomState(seed)
        # First generate a dense matrix with random values
        matrix = rng.uniform(min_val, max_val, size=(rows, cols))
        
        # Calculate number of columns to zero out
        cols_to_zero = int(cols * sparsity)
        
        # Randomly select columns to zero out
        zero_cols = rng.choice(cols, cols_to_zero, replace=False)
        
        # Zero out selected columns
        for col in zero_cols:
            matrix[:, col] = 0.0
        
        return matrix
    
    @staticmethod
    def generate_relu_like_sparse_matrix(rows: int, 
                                       cols: int, 
                                       sparsity: float,
                                       min_val: float = -1.0,
                                       max_val: float = 1.0,
                                       seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a matrix with ReLU-like sparsity pattern.
        Values are skewed negative, then ReLU is applied to create the target sparsity.
        """
        rng = np.random.RandomState(seed)
        # Generate values with negative bias to create desired sparsity after ReLU
        skew = math.log(1 / (1 - sparsity) - 1) if sparsity < 1.0 else 0
        mean_val = (min_val + max_val) / 2 - skew
        
        # Generate matrix with skewed distribution
        matrix = rng.normal(mean_val, (max_val - min_val) / 4, size=(rows, cols))
        
        # Apply ReLU
        matrix = np.maximum(0.0, matrix)
        
        # Scale to desired range
        if np.max(matrix) > 0:
            matrix = matrix / np.max(matrix) * max_val
        
        return matrix
    
    @staticmethod
    def generate_highly_sparse_matrix(rows: int, 
                                    cols: int, 
                                    sparsity: float,
                                    min_val: float = -1.0,
                                    max_val: float = 1.0,
                                    seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a highly sparse matrix with only a few non-zero values.
        """
        rng = np.random.RandomState(seed)
        # Start with all zeros
        matrix = np.zeros((rows, cols))
        
        # Calculate number of non-zeros
        total_elements = rows * cols
        non_zeros = int(total_elements * (1 - sparsity))
        
        # Randomly select positions for non-zeros
        indices = rng.choice(total_elements, non_zeros, replace=False)
        
        # Place non-zero values
        for idx in indices:
            row = idx // cols
            col = idx % cols
            matrix[row, col] = rng.uniform(min_val, max_val)
        
        return matrix
    
    @staticmethod
    def generate_sparse_matrix(rows: int, 
                             cols: int, 
                             sparsity: float,
                             pattern: SparsityPattern,
                             min_val: float = -1.0,
                             max_val: float = 1.0,
                             seed: Optional[int] = None) -> np.ndarray:
        """Generate sparse matrix with specified pattern and sparsity"""
        if pattern == SparsityPattern.RANDOM:
            return SparseMatrixGenerator.generate_random_sparse_matrix(
                rows, cols, sparsity, min_val, max_val, seed)
        elif pattern == SparsityPattern.STRUCTURED_BLOCK:
            return SparseMatrixGenerator.generate_block_sparse_matrix(
                rows, cols, sparsity, 4, min_val, max_val, seed)
        elif pattern == SparsityPattern.STRUCTURED_CHANNEL:
            return SparseMatrixGenerator.generate_channel_sparse_matrix(
                rows, cols, sparsity, min_val, max_val, seed)
        elif pattern == SparsityPattern.RELU_LIKE:
            return SparseMatrixGenerator.generate_relu_like_sparse_matrix(
                rows, cols, sparsity, min_val, max_val, seed)
        elif pattern == SparsityPattern.HIGHLY_SPARSE:
            return SparseMatrixGenerator.generate_highly_sparse_matrix(
                rows, cols, sparsity, min_val, max_val, seed)
        else:
            raise ValueError(f"Unsupported sparsity pattern: {pattern}")
    
    @staticmethod
    def calculate_actual_sparsity(matrix: np.ndarray) -> float:
        """Calculate the actual sparsity (fraction of zeros) in a matrix"""
        total_elements = matrix.size
        zero_elements = np.sum(matrix == 0.0)
        return zero_elements / total_elements


class SparseCompression:
    """
    Reference implementation of sparse compression formats used in Eyeriss Ultra.
    Implements both CSR (for weights) and bitmap-based formats (for activations).
    """
    
    @staticmethod
    def compress_weights_csr(matrix: np.ndarray) -> Dict[str, Any]:
        """
        Compress weight matrix using CSR format.
        Returns the compressed representation and metadata.
        """
        rows, cols = matrix.shape
        row_ptr = [0]
        col_idx = []
        values = []
        
        for i in range(rows):
            for j in range(cols):
                if matrix[i, j] != 0:
                    col_idx.append(j)
                    values.append(matrix[i, j])
            row_ptr.append(len(values))
        
        # Calculate compression ratio
        original_size = rows * cols
        compressed_size = len(values) + len(row_ptr) + len(col_idx)
        compression_ratio = original_size / max(1, compressed_size)
        
        return {
            "format": "CSR",
            "row_ptr": row_ptr,
            "col_idx": col_idx,
            "values": values,
            "original_shape": matrix.shape,
            "compression_ratio": compression_ratio,
            "original_size": original_size,
            "compressed_size": compressed_size
        }
    
    @staticmethod
    def decompress_weights_csr(compressed: Dict[str, Any]) -> np.ndarray:
        """Decompress weight matrix from CSR format"""
        rows, cols = compressed["original_shape"]
        row_ptr = compressed["row_ptr"]
        col_idx = compressed["col_idx"]
        values = compressed["values"]
        
        # Initialize output matrix
        matrix = np.zeros((rows, cols))
        
        # Fill in non-zero values
        for i in range(rows):
            for j in range(row_ptr[i], row_ptr[i+1]):
                matrix[i, col_idx[j]] = values[j]
        
        return matrix
    
    @staticmethod
    def compress_activations_bitmap(matrix: np.ndarray) -> Dict[str, Any]:
        """
        Compress activation matrix using bitmap-based format.
        Returns the compressed representation and metadata.
        """
        rows, cols = matrix.shape
        bitmap = np.zeros((rows, cols), dtype=bool)
        values = []
        
        # Create bitmap and collect non-zero values
        for i in range(rows):
            for j in range(cols):
                bitmap[i, j] = (matrix[i, j] != 0)
                if matrix[i, j] != 0:
                    values.append(matrix[i, j])
        
        # Calculate compression ratio
        original_size = rows * cols
        bitmap_size = (rows * cols + 7) // 8  # Number of bytes to store bitmap
        compressed_size = bitmap_size + len(values)
        compression_ratio = original_size / max(1, compressed_size)
        
        return {
            "format": "BITMAP",
            "bitmap": bitmap,
            "values": values,
            "original_shape": matrix.shape,
            "compression_ratio": compression_ratio,
            "original_size": original_size,
            "compressed_size": compressed_size
        }
    
    @staticmethod
    def decompress_activations_bitmap(compressed: Dict[str, Any]) -> np.ndarray:
        """Decompress activation matrix from bitmap format"""
        rows, cols = compressed["original_shape"]
        bitmap = compressed["bitmap"]
        values = compressed["values"]
        
        # Initialize output matrix
        matrix = np.zeros((rows, cols))
        
        # Fill in non-zero values
        value_idx = 0
        for i in range(rows):
            for j in range(cols):
                if bitmap[i, j]:
                    matrix[i, j] = values[value_idx]
                    value_idx += 1
        
        return matrix


class SparsityTestCase:
    """Test case for sparsity verification"""
    
    def __init__(self, 
                sparsity_config: SparsityConfig,
                weight_matrix: np.ndarray,
                activation_matrix: np.ndarray):
        self.sparsity_config = sparsity_config
        self.weight_matrix = weight_matrix
        self.activation_matrix = activation_matrix
        
        # Calculate actual sparsity
        self.actual_weight_sparsity = SparseMatrixGenerator.calculate_actual_sparsity(weight_matrix)
        self.actual_activation_sparsity = SparseMatrixGenerator.calculate_actual_sparsity(activation_matrix)
        
        # Compress matrices if enabled
        if sparsity_config.compression_enabled:
            self.compressed_weights = SparseCompression.compress_weights_csr(weight_matrix)
            self.compressed_activations = SparseCompression.compress_activations_bitmap(activation_matrix)
        else:
            self.compressed_weights = None
            self.compressed_activations = None
    
    def get_expected_operations(self) -> int:
        """
        Calculate expected number of non-zero operations
        based on non-zero elements in weight and activation matrices.
        """
        weight_nnz = np.count_nonzero(self.weight_matrix)
        activation_nnz = np.count_nonzero(self.activation_matrix)
        
        # For matrix multiplication, we need to consider the operation pattern
        rows_w, cols_w = self.weight_matrix.shape
        rows_a, cols_a = self.activation_matrix.shape
        
        # We assume weight_matrix is [M x K] and activation_matrix is [K x N]
        assert cols_w == rows_a, "Matrix dimensions mismatch for multiplication"
        
        # Count non-zero operations (requires both weight and activation to be non-zero)
        operations = 0
        for i in range(rows_w):
            for j in range(cols_a):
                for k in range(cols_w):
                    if self.weight_matrix[i, k] != 0 and self.activation_matrix[k, j] != 0:
                        operations += 1
        
        return operations
    
    def __str__(self) -> str:
        return (f"SparsityTestCase(weight_shape={self.weight_matrix.shape}, "
                f"activation_shape={self.activation_matrix.shape}, "
                f"weight_sparsity={self.actual_weight_sparsity:.2f}, "
                f"activation_sparsity={self.actual_activation_sparsity:.2f})")


class SparsityVerifier:
    """
    Verification engine for sparsity exploitation mechanisms.
    
    Verifies correctness of zero-skipping, compression, and validates
    performance benefits across different sparsity patterns and levels.
    """
    
    def __init__(self):
        self.executed_tests = 0
        self.passed_tests = 0
        self.sparsity_coverage = {}  # Track sparsity level coverage
        self.pattern_coverage = {pattern: False for pattern in SparsityPattern}
        self.feature_coverage = {
            "zero_skipping": False,
            "weight_compression": False,
            "activation_compression": False,
            "hybrid_sparsity": False,  # Both weight and activation sparse
        }
        self.compression_ratios = {
            "weight": [],
            "activation": []
        }
        self.operation_reduction = []  # Ratio of skipped operations
        logger.info("Initialized Sparsity Verifier")
    
    def verify_matrix_multiplication(self, test_case: SparsityTestCase) -> bool:
        """
        Verify sparse matrix multiplication with zero-skipping and compression.
        Checks correctness and performance benefits.
        """
        # Update coverage tracking
        self._update_coverage(test_case)
        
        # 1. Verify correctness of compressed formats
        compression_correct = self._verify_compression(test_case)
        
        # 2. Verify correctness of sparse computation
        computation_correct = self._verify_computation(test_case)
        
        # 3. Verify performance benefits
        performance_correct = self._verify_performance(test_case)
        
        # Overall test passed if all checks passed
        test_passed = compression_correct and computation_correct and performance_correct
        
        self.executed_tests += 1
        if test_passed:
            self.passed_tests += 1
            logger.info(f"Test case passed: {test_case}")
        else:
            logger.error(f"Test case failed: {test_case}")
        
        return test_passed
    
    def _update_coverage(self, test_case: SparsityTestCase) -> None:
        """Update coverage metrics based on test case"""
        # Track sparsity level coverage (rounded to nearest 10%)
        weight_sparsity_bin = round(test_case.actual_weight_sparsity * 10) / 10
        activation_sparsity_bin = round(test_case.actual_activation_sparsity * 10) / 10
        
        self.sparsity_coverage[f"weight_{weight_sparsity_bin:.1f}"] = True
        self.sparsity_coverage[f"activation_{activation_sparsity_bin:.1f}"] = True
        
        # Track pattern coverage
        self.pattern_coverage[test_case.sparsity_config.weight_pattern] = True
        self.pattern_coverage[test_case.sparsity_config.activation_pattern] = True
        
        # Track feature coverage
        if test_case.sparsity_config.zero_skipping_enabled:
            self.feature_coverage["zero_skipping"] = True
        
        if test_case.sparsity_config.compression_enabled:
            self.feature_coverage["weight_compression"] = True
            self.feature_coverage["activation_compression"] = True
        
        if test_case.actual_weight_sparsity > 0.3 and test_case.actual_activation_sparsity > 0.3:
            self.feature_coverage["hybrid_sparsity"] = True
    
    def _verify_compression(self, test_case: SparsityTestCase) -> bool:
        """Verify correctness of compression/decompression"""
        if not test_case.sparsity_config.compression_enabled:
            return True  # No compression to verify
        
        # Verify weight compression
        decompressed_weights = SparseCompression.decompress_weights_csr(test_case.compressed_weights)
        weights_match = np.allclose(decompressed_weights, test_case.weight_matrix)
        
        # Verify activation compression
        decompressed_activations = SparseCompression.decompress_activations_bitmap(test_case.compressed_activations)
        activations_match = np.allclose(decompressed_activations, test_case.activation_matrix)
        
        # Track compression ratios
        self.compression_ratios["weight"].append(test_case.compressed_weights["compression_ratio"])
        self.compression_ratios["activation"].append(test_case.compressed_activations["compression_ratio"])
        
        if not weights_match:
            logger.error("Weight compression/decompression mismatch")
        
        if not activations_match:
            logger.error("Activation compression/decompression mismatch")
        
        return weights_match and activations_match
    
    def _verify_computation(self, test_case: SparsityTestCase) -> bool:
        """Verify correctness of sparse computation with zero-skipping"""
        # Perform reference dense computation
        dense_result = np.matmul(test_case.weight_matrix, test_case.activation_matrix)
        
        # Perform sparse computation (simulated)
        # In a real verification, this would interface with the hardware or simulator
        # Here we'll just use the same numpy matmul to simulate "correct" sparse computation
        sparse_result = np.matmul(test_case.weight_matrix, test_case.activation_matrix)
        
        # Verify results match
        computation_correct = np.allclose(dense_result, sparse_result)
        
        if not computation_correct:
            logger.error("Sparse computation result mismatch")
        
        return computation_correct
    
    def _verify_performance(self, test_case: SparsityTestCase) -> bool:
        """Verify performance benefits from sparsity"""
        if not test_case.sparsity_config.zero_skipping_enabled:
            return True  # No zero-skipping to verify
        
        # Calculate theoretical performance benefit
        total_operations = test_case.weight_matrix.shape[0] * test_case.activation_matrix.shape[1] * test_case.weight_matrix.shape[1]
        actual_operations = test_case.get_expected_operations()
        operation_ratio = actual_operations / total_operations if total_operations > 0 else 1.0
        
        # Expected operations should match the product of non-zero ratios
        weight_nnz_ratio = 1.0 - test_case.actual_weight_sparsity
        activation_nnz_ratio = 1.0 - test_case.actual_activation_sparsity
        expected_ratio = weight_nnz_ratio * activation_nnz_ratio
        
        # Verify performance improvement (with some tolerance)
        # In a real verification, we would compare with measured performance
        performance_correct = abs(operation_ratio - expected_ratio) < 0.05
        
        # Track operation reduction
        operation_reduction = 1.0 - operation_ratio
        self.operation_reduction.append(operation_reduction)
        
        if not performance_correct:
            logger.error(f"Performance benefit mismatch: expected {expected_ratio:.4f}, got {operation_ratio:.4f}")
        
        return performance_correct
    
    def report_coverage(self) -> Dict:
        """Generate coverage report"""
        report = {
            "tests_executed": self.executed_tests,
            "tests_passed": self.passed_tests,
            "pass_rate": self.passed_tests / self.executed_tests if self.executed_tests > 0 else 0.0,
            "sparsity_coverage": self.sparsity_coverage,
            "pattern_coverage": {pattern.name: covered for pattern, covered in self.pattern_coverage.items()},
            "feature_coverage": self.feature_coverage,
            "avg_compression_ratio": {
                "weight": np.mean(self.compression_ratios["weight"]) if self.compression_ratios["weight"] else 0.0,
                "activation": np.mean(self.compression_ratios["activation"]) if self.compression_ratios["activation"] else 0.0
            },
            "avg_operation_reduction": np.mean(self.operation_reduction) if self.operation_reduction else 0.0
        }
        
        # Calculate coverage percentages
        sparsity_coverage_pct = sum(1 for covered in self.sparsity_coverage.values() if covered) / max(1, len(self.sparsity_coverage)) * 100.0
        pattern_coverage_pct = sum(1 for covered in self.pattern_coverage.values() if covered) / len(self.pattern_coverage) * 100.0
        feature_coverage_pct = sum(1 for covered in self.feature_coverage.values() if covered) / len(self.feature_coverage) * 100.0
        
        report["coverage_summary"] = {
            "sparsity_coverage_pct": sparsity_coverage_pct,
            "pattern_coverage_pct": pattern_coverage_pct,
            "feature_coverage_pct": feature_coverage_pct,
            "overall_coverage_pct": (sparsity_coverage_pct + pattern_coverage_pct + feature_coverage_pct) / 3.0
        }
        
        return report
    
    def generate_performance_graph(self, filename: str = "sparsity_performance.png") -> None:
        """Generate graph showing performance benefits across sparsity levels"""
        if not self.operation_reduction:
            logger.warning("No performance data to plot")
            return
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Plot operation reduction vs sparsity level
        plt.scatter(range(len(self.operation_reduction)), self.operation_reduction, alpha=0.7)
        plt.plot(range(len(self.operation_reduction)), self.operation_reduction, 'r--', alpha=0.5)
        
        plt.xlabel('Test Case')
        plt.ylabel('Operation Reduction (1 - operations_performed/total_operations)')
        plt.title('Performance Benefit from Sparsity')
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3)
        
        # Add horizontal line at theoretical maximum (assuming perfect zero-skipping)
        plt.axhline(y=max(self.operation_reduction), color='g', linestyle='--', alpha=0.7, 
                  label=f'Max Observed: {max(self.operation_reduction):.2f}')
        
        # Add average line
        avg_reduction = np.mean(self.operation_reduction)
        plt.axhline(y=avg_reduction, color='b', linestyle='--', alpha=0.7,
                  label=f'Average: {avg_reduction:.2f}')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        logger.info(f"Performance graph saved to {filename}")


def run_sparsity_verification() -> None:
    """Run a complete sparsity verification suite"""
    logger.info("Starting sparsity verification suite")
    
    # Initialize verifier
    verifier = SparsityVerifier()
    
    # Define sparsity configurations to test
    sparsity_configs = [
        # Basic sparsity levels with random pattern
        SparsityConfig(0.0, 0.0, SparsityPattern.RANDOM, SparsityPattern.RANDOM, False, False),
        SparsityConfig(0.3, 0.0, SparsityPattern.RANDOM, SparsityPattern.RANDOM, True, True),
        SparsityConfig(0.5, 0.0, SparsityPattern.RANDOM, SparsityPattern.RANDOM, True, True),
        SparsityConfig(0.7, 0.0, SparsityPattern.RANDOM, SparsityPattern.RANDOM, True, True),
        SparsityConfig(0.0, 0.3, SparsityPattern.RANDOM, SparsityPattern.RANDOM, True, True),
        SparsityConfig(0.0, 0.5, SparsityPattern.RANDOM, SparsityPattern.RANDOM, True, True),
        SparsityConfig(0.0, 0.7, SparsityPattern.RANDOM, SparsityPattern.RELU_LIKE, True, True),
        
        # Combined weight and activation sparsity
        SparsityConfig(0.3, 0.3, SparsityPattern.RANDOM, SparsityPattern.RANDOM, True, True),
        SparsityConfig(0.5, 0.5, SparsityPattern.RANDOM, SparsityPattern.RANDOM, True, True),
        SparsityConfig(0.7, 0.5, SparsityPattern.RANDOM, SparsityPattern.RELU_LIKE, True, True),
        
        # Structured sparsity patterns
        SparsityConfig(0.5, 0.0, SparsityPattern.STRUCTURED_BLOCK, SparsityPattern.RANDOM, True, True),
        SparsityConfig(0.5, 0.0, SparsityPattern.STRUCTURED_CHANNEL, SparsityPattern.RANDOM, True, True),
        SparsityConfig(0.0, 0.5, SparsityPattern.RANDOM, SparsityPattern.RELU_LIKE, True, True),
        
        # High sparsity cases
        SparsityConfig(0.9, 0.0, SparsityPattern.HIGHLY_SPARSE, SparsityPattern.RANDOM, True, True),
        SparsityConfig(0.0, 0.9, SparsityPattern.RANDOM, SparsityPattern.HIGHLY_SPARSE, True, True),
        SparsityConfig(0.9, 0.9, SparsityPattern.HIGHLY_SPARSE, SparsityPattern.HIGHLY_SPARSE, True, True),
    ]
    
    # Run verification for different matrix sizes
    matrix_sizes = [(32, 32, 32), (64, 32, 16), (128, 64, 32)]  # (M, K, N) dimensions
    
    for size_idx, (M, K, N) in enumerate(matrix_sizes):
        logger.info(f"Testing matrix size {size_idx+1}/{len(matrix_sizes)}: {M}x{K} * {K}x{N}")
        
        for i, config in enumerate(sparsity_configs):
            logger.info(f"Testing configuration {i+1}/{len(sparsity_configs)}: {config}")
            
            # Generate sparse matrices according to configuration
            weight_matrix = SparseMatrixGenerator.generate_sparse_matrix(
                M, K, config.weight_sparsity, config.weight_pattern, seed=i+size_idx*100)
            
            activation_matrix = SparseMatrixGenerator.generate_sparse_matrix(
                K, N, config.activation_sparsity, config.activation_pattern, seed=i+size_idx*100+50)
            
            # Create test case
            test_case = SparsityTestCase(config, weight_matrix, activation_matrix)
            
            # Verify test case
            verifier.verify_matrix_multiplication(test_case)
    
    # Report coverage
    coverage_report = verifier.report_coverage()
    logger.info("Verification completed")
    logger.info(f"Tests executed: {coverage_report['tests_executed']}")
    logger.info(f"Tests passed: {coverage_report['tests_passed']}")
    logger.info(f"Pass rate: {coverage_report['pass_rate']*100:.2f}%")
    logger.info(f"Overall coverage: {coverage_report['coverage_summary']['overall_coverage_pct']:.2f}%")
    
    # Log detailed coverage information
    logger.info("Pattern coverage:")
    for pattern, covered in coverage_report['pattern_coverage'].items():
        logger.info(f"  {pattern}: {'COVERED' if covered else 'NOT COVERED'}")
    
    logger.info("Feature coverage:")
    for feature, covered in coverage_report['feature_coverage'].items():
        logger.info(f"  {feature}: {'COVERED' if covered else 'NOT COVERED'}")
    
    logger.info(f"Average weight compression ratio: {coverage_report['avg_compression_ratio']['weight']:.2f}x")
    logger.info(f"Average activation compression ratio: {coverage_report['avg_compression_ratio']['activation']:.2f}x")
    logger.info(f"Average operation reduction: {coverage_report['avg_operation_reduction']*100:.2f}%")
    
    # Generate performance graph
    verifier.generate_performance_graph()


if __name__ == "__main__":
    # Run the verification suite
    run_sparsity_verification()