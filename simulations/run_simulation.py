#!/usr/bin/env python3
"""
Eyeriss Ultra Simulation Runner

This script provides a command-line interface to run simulations and experiments
with the Eyeriss Ultra accelerator design.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EyerissUltra")

# Add parent directory to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(project_root))

# Import simulation modules
from simulations.eyeriss_ultra_sim import EyerissUltraSimulation
from simulations.optimization_analysis import OptimizationAnalysis

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Eyeriss Ultra Simulation Runner")
    
    parser.add_argument('--config', type=str, default='standard',
                        choices=['standard', 'mixed_precision', 'sparsity', 'clustering', 'all'],
                        help='Simulation configuration preset')
    
    parser.add_argument('--arch-file', type=str, default='eyeriss_ultra_v3.yaml',
                        help='Architecture YAML file')
    
    parser.add_argument('--problem-file', type=str, default='problem_v3.yaml',
                        help='Problem YAML file')
    
    parser.add_argument('--mapper-file', type=str, default='mapper_v3.yaml',
                        help='Mapper YAML file')
    
    parser.add_argument('--output-dir', type=str, default='sim_results',
                        help='Output directory for simulation results')
    
    parser.add_argument('--analysis', action='store_true',
                        help='Run detailed analysis of optimizations')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    return parser.parse_args()

def setup_simulation_config(config_name):
    """Set up simulation configuration based on preset name"""
    config = {}
    
    if config_name == 'standard':
        # Standard configuration with balanced settings
        config = {
            'precision_mix': {
                'int8': 0.7,
                'int4': 0.2,
                'fp16': 0.1
            },
            'sparsity_config': {
                'weight_sparsity': 0.7,
                'activation_sparsity': 0.5,
                'skipping_efficiency': 0.9
            },
            'pe_clustering': {
                'cluster_size': 4,
                'communication_overhead': 0.15,
                'sharing_benefit': 0.25
            }
        }
    elif config_name == 'mixed_precision':
        # Focus on mixed precision benefits
        config = {
            'precision_mix': {
                'int8': 0.4,
                'int4': 0.5,
                'fp16': 0.1
            },
            'sparsity_config': {
                'weight_sparsity': 0.0,
                'activation_sparsity': 0.0,
                'skipping_efficiency': 0.0
            },
            'pe_clustering': {
                'cluster_size': 4,
                'communication_overhead': 0.15,
                'sharing_benefit': 0.25
            }
        }
    elif config_name == 'sparsity':
        # Focus on sparsity benefits
        config = {
            'precision_mix': {
                'int8': 1.0,
                'int4': 0.0,
                'fp16': 0.0
            },
            'sparsity_config': {
                'weight_sparsity': 0.8,
                'activation_sparsity': 0.7,
                'skipping_efficiency': 0.95
            },
            'pe_clustering': {
                'cluster_size': 4,
                'communication_overhead': 0.15,
                'sharing_benefit': 0.25
            }
        }
    elif config_name == 'clustering':
        # Focus on PE clustering benefits
        config = {
            'precision_mix': {
                'int8': 1.0,
                'int4': 0.0,
                'fp16': 0.0
            },
            'sparsity_config': {
                'weight_sparsity': 0.0,
                'activation_sparsity': 0.0,
                'skipping_efficiency': 0.0
            },
            'pe_clustering': {
                'cluster_size': 8,  # Larger clusters
                'communication_overhead': 0.1,
                'sharing_benefit': 0.4  # More sharing benefits
            }
        }
    elif config_name == 'all':
        # Enable all optimizations at maximum settings
        config = {
            'precision_mix': {
                'int8': 0.3,
                'int4': 0.6,
                'fp16': 0.1
            },
            'sparsity_config': {
                'weight_sparsity': 0.8,
                'activation_sparsity': 0.7,
                'skipping_efficiency': 0.95
            },
            'pe_clustering': {
                'cluster_size': 8,
                'communication_overhead': 0.1,
                'sharing_benefit': 0.4
            }
        }
    
    return config

def run_simulation(args):
    """Run the Eyeriss Ultra simulation with specified configuration"""
    logger.info(f"Running simulation with configuration: {args.config}")
    
    # Initialize simulation
    simulation = EyerissUltraSimulation(
        arch_file=args.arch_file,
        problem_file=args.problem_file,
        mapper_file=args.mapper_file,
        output_dir=args.output_dir
    )
    
    # Apply configuration
    config = setup_simulation_config(args.config)
    simulation.precision_factors = config.get('precision_mix', simulation.precision_factors)
    simulation.sparsity_config = config.get('sparsity_config', simulation.sparsity_config)
    simulation.pe_clustering = config.get('pe_clustering', simulation.pe_clustering)
    
    # Run simulation
    results = simulation.run_enhanced_simulation()
    
    if results:
        logger.info("Simulation completed successfully")
        logger.info(f"Energy improvement: {results['energy_improvement']:.2f}×")
        logger.info(f"Latency improvement: {results['latency_improvement']:.2f}×")
        logger.info(f"Energy-Delay Product improvement: {results['energy_improvement'] * results['latency_improvement']:.2f}×")
        
        # Visualize results
        simulation.visualize_results(results)
    else:
        logger.error("Simulation failed")
    
    # Run optimization analysis if requested
    if args.analysis:
        logger.info("Running optimization analysis")
        analyzer = OptimizationAnalysis()
        analyzer.run_analysis()

if __name__ == "__main__":
    args = parse_args()
    
    # Set verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run simulation
    run_simulation(args)