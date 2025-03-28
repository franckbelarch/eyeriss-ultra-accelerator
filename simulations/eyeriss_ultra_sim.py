import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from pytimeloop.looptree.run import run_looptree
from pytimeloop.fastfusion.fastmodel import compile_mapping
from pytimeloop.fastfusion.sim import SIM, Tiling, Loop, TensorStorage
from pytimeloop.looptree.energy import gather_actions
from pytimeloop.timeloopfe import v4 as tl

class EyerissUltraSimulation:
    def __init__(self, 
                 arch_file='eyeriss_ultra_v3.yaml', 
                 problem_file='problem_v3.yaml',
                 mapper_file='mapper_v3.yaml',
                 output_dir='sim_results'):
        """Initialize the custom simulation environment for Eyeriss Ultra."""
        self.base_dir = Path(__file__).parent.parent
        self.arch_file = self.base_dir / arch_file
        self.problem_file = self.base_dir / problem_file
        self.mapper_file = self.base_dir / mapper_file
        self.output_dir = self.base_dir / output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Memory hierarchy configuration
        self.memory_hierarchy = {
            0: 'DRAM',
            1: 'GlobalBuffer',
            2: 'SPBuffer',
            3: 'RegisterFile',
            4: 'PEArray'
        }
        
        # Mixed precision configuration (relative energy per operation)
        self.precision_factors = {
            'fp32': 1.0,
            'fp16': 0.25,
            'int8': 0.12,
            'int4': 0.06
        }
        
        # Sparsity configuration
        self.sparsity_config = {
            'weight_sparsity': 0.7,  # 70% of weights are zero
            'activation_sparsity': 0.5,  # 50% of activations are zero
            'skipping_efficiency': 0.9  # 90% of zero operations are skipped
        }
        
        # PE clustering configuration
        self.pe_clustering = {
            'cluster_size': 4,  # 4 PEs per cluster
            'communication_overhead': 0.15,  # 15% communication overhead
            'sharing_benefit': 0.25  # 25% benefit from weight sharing
        }
        
        # Load configuration files
        self.load_configs()
    
    def load_configs(self):
        """Load configuration files for architecture, problem, and mapper."""
        try:
            self.spec = tl.Specification.from_yaml_files(
                str(self.arch_file), 
                str(self.problem_file),
                str(self.mapper_file)
            )
            print(f"Successfully loaded configuration files")
        except Exception as e:
            print(f"Error loading configuration files: {e}")
            self.spec = None
    
    def run_baseline_simulation(self):
        """Run a baseline simulation using standard Timeloop."""
        try:
            # Create bindings for memory hierarchy
            bindings = {i: name for i, name in self.memory_hierarchy.items()}
            
            # Run standard looptree simulation
            latency, energy = run_looptree(
                self.base_dir,
                [
                    str(self.arch_file.name),
                    str(self.problem_file.name),
                    str(self.mapper_file.name)
                ],
                self.output_dir,
                bindings,
                True  # Include energy
            )
            
            return {
                'latency': latency,
                'energy': energy,
                'total_energy': sum(energy.values())
            }
        except Exception as e:
            print(f"Error running baseline simulation: {e}")
            return None
    
    def apply_mixed_precision(self, energy_results):
        """Apply mixed precision effects to energy results."""
        # For demonstration, assume 70% int8, 20% int4, 10% fp16 operations
        precision_mix = {
            'int8': 0.7,
            'int4': 0.2,
            'fp16': 0.1
        }
        
        modified_energy = {}
        for component, energy in energy_results.items():
            if component[0] == 'PEArray' and component[1] == 'compute':
                # Apply mixed precision only to compute operations
                mixed_energy = sum(
                    energy * ratio * self.precision_factors[precision] 
                    for precision, ratio in precision_mix.items()
                )
                modified_energy[component] = mixed_energy
            else:
                modified_energy[component] = energy
        
        return modified_energy
    
    def apply_sparsity(self, energy_results):
        """Apply sparsity effects to energy results."""
        modified_energy = {}
        for component, energy in energy_results.items():
            if component[0] == 'PEArray' and component[1] == 'compute':
                # Compute operations benefit from sparsity
                effective_ops = (1 - self.sparsity_config['weight_sparsity'] * 
                               self.sparsity_config['activation_sparsity'] * 
                               self.sparsity_config['skipping_efficiency'])
                modified_energy[component] = energy * effective_ops
            elif component[1] == 'read' or component[1] == 'write':
                # Memory operations benefit partially from sparsity through compression
                if component[0] == 'DRAM' or component[0] == 'GlobalBuffer':
                    weight_benefit = self.sparsity_config['weight_sparsity'] * 0.8
                    act_benefit = self.sparsity_config['activation_sparsity'] * 0.7
                    modified_energy[component] = energy * (1 - (weight_benefit + act_benefit) / 2)
                else:
                    modified_energy[component] = energy
            else:
                modified_energy[component] = energy
        
        return modified_energy
    
    def apply_pe_clustering(self, energy_results):
        """Apply PE clustering effects to energy results."""
        modified_energy = {}
        for component, energy in energy_results.items():
            if component[0] == 'RegisterFile':
                # RegisterFile benefits from weight sharing in clusters
                modified_energy[component] = energy * (1 - self.pe_clustering['sharing_benefit'])
            elif component[0] == 'SPBuffer':
                # Some additional communication overhead
                modified_energy[component] = energy * (1 + self.pe_clustering['communication_overhead'])
            else:
                modified_energy[component] = energy
        
        return modified_energy
    
    def apply_technology_scaling(self, energy_results, old_node=16, new_node=7):
        """Apply technology scaling from 16nm to 7nm."""
        # Rough approximation: energy scales with square of feature size
        scaling_factor = (new_node / old_node) ** 2
        
        return {k: v * scaling_factor for k, v in energy_results.items()}
    
    def run_enhanced_simulation(self):
        """Run enhanced simulation with our improvements."""
        # First run baseline simulation
        baseline_results = self.run_baseline_simulation()
        if not baseline_results:
            return None
        
        # Apply our enhancements
        enhanced_energy = baseline_results['energy']
        
        # Apply mixed precision benefits
        enhanced_energy = self.apply_mixed_precision(enhanced_energy)
        
        # Apply sparsity benefits
        enhanced_energy = self.apply_sparsity(enhanced_energy)
        
        # Apply PE clustering benefits
        enhanced_energy = self.apply_pe_clustering(enhanced_energy)
        
        # Apply technology scaling from 16nm to 7nm
        enhanced_energy = self.apply_technology_scaling(enhanced_energy)
        
        # Calculate total energy
        total_enhanced_energy = sum(enhanced_energy.values())
        
        # Estimate improved latency from parallelism and sparsity
        # - PE clustering provides more parallelism
        # - Sparsity skipping reduces compute operations
        parallelism_improvement = 1 + (self.pe_clustering['cluster_size'] / 16)  # Assuming 16x16 base array
        sparsity_improvement = 1 / (1 - (self.sparsity_config['weight_sparsity'] * 
                                        self.sparsity_config['activation_sparsity'] * 
                                        self.sparsity_config['skipping_efficiency']))
        
        enhanced_latency = baseline_results['latency'] / (parallelism_improvement * sparsity_improvement)
        
        return {
            'baseline_latency': baseline_results['latency'],
            'baseline_energy': baseline_results['energy'],
            'baseline_total_energy': baseline_results['total_energy'],
            'enhanced_latency': enhanced_latency,
            'enhanced_energy': enhanced_energy,
            'enhanced_total_energy': total_enhanced_energy,
            'latency_improvement': baseline_results['latency'] / enhanced_latency,
            'energy_improvement': baseline_results['total_energy'] / total_enhanced_energy
        }
    
    def visualize_results(self, results):
        """Visualize simulation results."""
        if not results:
            print("No results to visualize")
            return
        
        # Prepare data for visualization
        baseline_energy_by_component = defaultdict(float)
        enhanced_energy_by_component = defaultdict(float)
        
        for (component, op), energy in results['baseline_energy'].items():
            baseline_energy_by_component[component] += energy
        
        for (component, op), energy in results['enhanced_energy'].items():
            enhanced_energy_by_component[component] += energy
        
        # Create bar chart for energy comparison
        components = list(baseline_energy_by_component.keys())
        baseline_values = [baseline_energy_by_component[c] for c in components]
        enhanced_values = [enhanced_energy_by_component[c] for c in components]
        
        x = np.arange(len(components))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(x - width/2, baseline_values, width, label='Baseline')
        ax.bar(x + width/2, enhanced_values, width, label='Eyeriss Ultra')
        
        ax.set_yscale('log')
        ax.set_ylabel('Energy (pJ)')
        ax.set_title('Energy Consumption by Component')
        ax.set_xticks(x)
        ax.set_xticklabels(components)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'energy_comparison.png')
        
        # Create summary table
        summary = {
            'Metric': ['Latency (cycles)', 'Energy (pJ)', 'Energy-Delay Product'],
            'Baseline': [
                results['baseline_latency'],
                results['baseline_total_energy'],
                results['baseline_latency'] * results['baseline_total_energy']
            ],
            'Eyeriss Ultra': [
                results['enhanced_latency'],
                results['enhanced_total_energy'],
                results['enhanced_latency'] * results['enhanced_total_energy']
            ],
            'Improvement': [
                f"{results['latency_improvement']:.2f}x",
                f"{results['energy_improvement']:.2f}x",
                f"{(results['latency_improvement'] * results['energy_improvement']):.2f}x"
            ]
        }
        
        df = pd.DataFrame(summary)
        with open(self.output_dir / 'simulation_summary.txt', 'w') as f:
            f.write("Eyeriss Ultra Simulation Results\n")
            f.write("===============================\n\n")
            f.write(df.to_string(index=False))
            f.write("\n\n")
            
            # Add configuration details
            f.write("Simulation Configuration\n")
            f.write("=======================\n\n")
            f.write(f"Architecture file: {self.arch_file.name}\n")
            f.write(f"Problem file: {self.problem_file.name}\n")
            f.write(f"Mapper file: {self.mapper_file.name}\n\n")
            
            f.write("Mixed Precision Configuration\n")
            f.write(f"- FP32 energy factor: {self.precision_factors['fp32']}\n")
            f.write(f"- FP16 energy factor: {self.precision_factors['fp16']}\n") 
            f.write(f"- INT8 energy factor: {self.precision_factors['int8']}\n")
            f.write(f"- INT4 energy factor: {self.precision_factors['int4']}\n\n")
            
            f.write("Sparsity Configuration\n")
            f.write(f"- Weight sparsity: {self.sparsity_config['weight_sparsity']*100}%\n")
            f.write(f"- Activation sparsity: {self.sparsity_config['activation_sparsity']*100}%\n")
            f.write(f"- Skipping efficiency: {self.sparsity_config['skipping_efficiency']*100}%\n\n")
            
            f.write("PE Clustering Configuration\n")
            f.write(f"- Cluster size: {self.pe_clustering['cluster_size']} PEs\n")
            f.write(f"- Communication overhead: {self.pe_clustering['communication_overhead']*100}%\n")
            f.write(f"- Weight sharing benefit: {self.pe_clustering['sharing_benefit']*100}%\n\n")
            
            f.write("Technology Scaling\n")
            f.write(f"- Original node: 16nm\n")
            f.write(f"- New node: 7nm\n")
            f.write(f"- Energy scaling factor: {(7/16)**2:.3f}\n")
        
        print(f"Results saved to {self.output_dir}")

if __name__ == "__main__":
    simulation = EyerissUltraSimulation()
    results = simulation.run_enhanced_simulation()
    if results:
        simulation.visualize_results(results)
    else:
        print("Simulation failed to produce results")