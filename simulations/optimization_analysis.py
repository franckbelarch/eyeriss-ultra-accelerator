import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from improved_design.simulation.eyeriss_ultra_sim import EyerissUltraSimulation

class OptimizationAnalysis:
    def __init__(self):
        """Initialize the optimization analysis tool."""
        self.simulator = EyerissUltraSimulation()
        self.output_dir = self.simulator.output_dir
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define the optimizations to analyze
        self.optimizations = [
            "baseline",
            "mixed_precision",
            "sparsity",
            "pe_clustering",
            "tech_scaling",
            "all_optimizations"
        ]
        
        # Optimization descriptions for reporting
        self.optimization_desc = {
            "baseline": "Baseline Eyeriss Architecture",
            "mixed_precision": "Mixed Precision (INT8/INT4/FP16)",
            "sparsity": "Sparsity Exploitation",
            "pe_clustering": "PE Clustering",
            "tech_scaling": "Technology Scaling (16nm → 7nm)",
            "all_optimizations": "All Optimizations Combined"
        }
    
    def run_optimization_analysis(self):
        """Run analysis for each optimization separately."""
        # First run baseline simulation
        baseline_results = self.simulator.run_baseline_simulation()
        if not baseline_results:
            print("Baseline simulation failed")
            return
        
        # Save baseline results
        results = {
            "baseline": {
                "latency": baseline_results["latency"],
                "energy": baseline_results["energy"],
                "total_energy": baseline_results["total_energy"]
            }
        }
        
        # Run with mixed precision only
        mixed_precision_energy = self.simulator.apply_mixed_precision(baseline_results["energy"])
        results["mixed_precision"] = {
            "latency": baseline_results["latency"],  # No latency improvement from mixed precision alone
            "energy": mixed_precision_energy,
            "total_energy": sum(mixed_precision_energy.values())
        }
        
        # Run with sparsity only
        sparsity_energy = self.simulator.apply_sparsity(baseline_results["energy"])
        # Calculate latency improvement from sparsity
        sparsity_improvement = 1 / (1 - (self.simulator.sparsity_config['weight_sparsity'] * 
                                       self.simulator.sparsity_config['activation_sparsity'] * 
                                       self.simulator.sparsity_config['skipping_efficiency']))
        results["sparsity"] = {
            "latency": baseline_results["latency"] / sparsity_improvement,
            "energy": sparsity_energy,
            "total_energy": sum(sparsity_energy.values())
        }
        
        # Run with PE clustering only
        pe_clustering_energy = self.simulator.apply_pe_clustering(baseline_results["energy"])
        # Calculate latency improvement from PE clustering
        parallelism_improvement = 1 + (self.simulator.pe_clustering['cluster_size'] / 16)
        results["pe_clustering"] = {
            "latency": baseline_results["latency"] / parallelism_improvement,
            "energy": pe_clustering_energy,
            "total_energy": sum(pe_clustering_energy.values())
        }
        
        # Run with technology scaling only
        tech_scaling_energy = self.simulator.apply_technology_scaling(baseline_results["energy"])
        results["tech_scaling"] = {
            "latency": baseline_results["latency"] * 0.7,  # Assume 30% freq improvement from tech scaling
            "energy": tech_scaling_energy,
            "total_energy": sum(tech_scaling_energy.values())
        }
        
        # Run with all optimizations
        # Start with baseline
        all_opt_energy = baseline_results["energy"]
        # Apply optimizations
        all_opt_energy = self.simulator.apply_mixed_precision(all_opt_energy)
        all_opt_energy = self.simulator.apply_sparsity(all_opt_energy)
        all_opt_energy = self.simulator.apply_pe_clustering(all_opt_energy)
        all_opt_energy = self.simulator.apply_technology_scaling(all_opt_energy)
        
        # Calculate combined latency improvement
        combined_latency_improvement = sparsity_improvement * parallelism_improvement * 1.3  # 1.3 for tech scaling
        
        results["all_optimizations"] = {
            "latency": baseline_results["latency"] / combined_latency_improvement,
            "energy": all_opt_energy,
            "total_energy": sum(all_opt_energy.values())
        }
        
        return results
    
    def calculate_improvements(self, results):
        """Calculate improvement factors for each optimization."""
        improvements = {}
        baseline = results["baseline"]
        
        for opt in self.optimizations:
            if opt == "baseline":
                continue
                
            opt_result = results[opt]
            improvements[opt] = {
                "latency_improvement": baseline["latency"] / opt_result["latency"],
                "energy_improvement": baseline["total_energy"] / opt_result["total_energy"],
                "edp_improvement": (baseline["latency"] * baseline["total_energy"]) / 
                                  (opt_result["latency"] * opt_result["total_energy"])
            }
        
        return improvements
    
    def visualize_improvements(self, improvements):
        """Create visualizations for improvement analysis."""
        # Prepare data for plotting
        opts = [opt for opt in self.optimizations if opt != "baseline"]
        latency_imp = [improvements[opt]["latency_improvement"] for opt in opts]
        energy_imp = [improvements[opt]["energy_improvement"] for opt in opts]
        edp_imp = [improvements[opt]["edp_improvement"] for opt in opts]
        
        # Convert optimization keys to readable descriptions
        labels = [self.optimization_desc[opt] for opt in opts]
        
        # Create bar chart
        x = np.arange(len(labels))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.bar(x - width, latency_imp, width, label='Latency Improvement')
        ax.bar(x, energy_imp, width, label='Energy Improvement')
        ax.bar(x + width, edp_imp, width, label='Energy-Delay Product Improvement')
        
        ax.set_ylabel('Improvement Factor (×)')
        ax.set_title('Performance Improvements by Optimization')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        
        # Add value labels
        for i, v in enumerate(latency_imp):
            ax.text(i - width, v + 0.1, f'{v:.1f}×', ha='center')
        for i, v in enumerate(energy_imp):
            ax.text(i, v + 0.1, f'{v:.1f}×', ha='center')
        for i, v in enumerate(edp_imp):
            ax.text(i + width, v + 0.1, f'{v:.1f}×', ha='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'optimization_improvements.png')
        
        # Create stacked bar chart for energy breakdown
        return fig
    
    def generate_report(self, results, improvements):
        """Generate comprehensive analysis report."""
        # Create summary table
        summary = []
        
        for opt in self.optimizations:
            if opt == "baseline":
                row = {
                    "Optimization": self.optimization_desc[opt],
                    "Latency (cycles)": results[opt]["latency"],
                    "Energy (pJ)": results[opt]["total_energy"],
                    "Energy-Delay Product": results[opt]["latency"] * results[opt]["total_energy"],
                    "Latency Improvement": "1.0×",
                    "Energy Improvement": "1.0×",
                    "EDP Improvement": "1.0×"
                }
            else:
                row = {
                    "Optimization": self.optimization_desc[opt],
                    "Latency (cycles)": results[opt]["latency"],
                    "Energy (pJ)": results[opt]["total_energy"],
                    "Energy-Delay Product": results[opt]["latency"] * results[opt]["total_energy"],
                    "Latency Improvement": f"{improvements[opt]['latency_improvement']:.2f}×",
                    "Energy Improvement": f"{improvements[opt]['energy_improvement']:.2f}×",
                    "EDP Improvement": f"{improvements[opt]['edp_improvement']:.2f}×"
                }
            summary.append(row)
        
        df = pd.DataFrame(summary)
        
        # Generate breakdown of energy savings by component
        energy_breakdown = defaultdict(dict)
        
        baseline_energy = results["baseline"]["energy"]
        all_opt_energy = results["all_optimizations"]["energy"]
        
        # Group by component
        baseline_by_component = defaultdict(float)
        optimized_by_component = defaultdict(float)
        
        for (component, op), energy in baseline_energy.items():
            baseline_by_component[component] += energy
        
        for (component, op), energy in all_opt_energy.items():
            optimized_by_component[component] += energy
        
        # Calculate savings
        savings = {}
        for component in baseline_by_component:
            baseline = baseline_by_component[component]
            optimized = optimized_by_component.get(component, 0)
            savings[component] = {
                "baseline": baseline,
                "optimized": optimized,
                "savings": baseline - optimized,
                "savings_percent": (baseline - optimized) / baseline * 100 if baseline > 0 else 0
            }
        
        # Generate report
        with open(self.output_dir / 'optimization_analysis.txt', 'w') as f:
            f.write("Eyeriss Ultra Optimization Analysis\n")
            f.write("==================================\n\n")
            
            f.write("Performance Summary\n")
            f.write("-----------------\n")
            f.write(df.to_string(index=False))
            f.write("\n\n")
            
            f.write("Energy Savings by Component\n")
            f.write("-------------------------\n")
            for component, data in savings.items():
                f.write(f"{component}:\n")
                f.write(f"  Baseline: {data['baseline']:.2f} pJ\n")
                f.write(f"  Optimized: {data['optimized']:.2f} pJ\n")
                f.write(f"  Savings: {data['savings']:.2f} pJ ({data['savings_percent']:.1f}%)\n\n")
            
            f.write("Optimization Details\n")
            f.write("------------------\n")
            
            # Mixed precision
            f.write("Mixed Precision:\n")
            f.write(f"  - Energy improvement: {improvements['mixed_precision']['energy_improvement']:.2f}×\n")
            f.write("  - Implementation: INT8 (70%), INT4 (20%), FP16 (10%)\n")
            f.write("  - Benefits: Reduced compute energy, smaller storage requirements\n\n")
            
            # Sparsity
            f.write("Sparsity Exploitation:\n")
            f.write(f"  - Energy improvement: {improvements['sparsity']['energy_improvement']:.2f}×\n")
            f.write(f"  - Latency improvement: {improvements['sparsity']['latency_improvement']:.2f}×\n")
            f.write(f"  - Weight sparsity: {self.simulator.sparsity_config['weight_sparsity']*100}%\n")
            f.write(f"  - Activation sparsity: {self.simulator.sparsity_config['activation_sparsity']*100}%\n")
            f.write(f"  - Zero-operation skipping efficiency: {self.simulator.sparsity_config['skipping_efficiency']*100}%\n\n")
            
            # PE Clustering
            f.write("PE Clustering:\n")
            f.write(f"  - Energy improvement: {improvements['pe_clustering']['energy_improvement']:.2f}×\n") 
            f.write(f"  - Latency improvement: {improvements['pe_clustering']['latency_improvement']:.2f}×\n")
            f.write(f"  - Cluster size: {self.simulator.pe_clustering['cluster_size']} PEs\n")
            f.write(f"  - Weight sharing benefit: {self.simulator.pe_clustering['sharing_benefit']*100}%\n")
            f.write(f"  - Communication overhead: {self.simulator.pe_clustering['communication_overhead']*100}%\n\n")
            
            # Technology Scaling
            f.write("Technology Scaling:\n")
            f.write(f"  - Energy improvement: {improvements['tech_scaling']['energy_improvement']:.2f}×\n")
            f.write(f"  - Latency improvement: {improvements['tech_scaling']['latency_improvement']:.2f}×\n")
            f.write("  - Scaling from 16nm to 7nm process\n\n")
            
            # Combined optimizations
            f.write("Combined Optimizations:\n") 
            f.write(f"  - Energy improvement: {improvements['all_optimizations']['energy_improvement']:.2f}×\n")
            f.write(f"  - Latency improvement: {improvements['all_optimizations']['latency_improvement']:.2f}×\n")
            f.write(f"  - Energy-Delay Product improvement: {improvements['all_optimizations']['edp_improvement']:.2f}×\n")
        
        print(f"Analysis report saved to {self.output_dir / 'optimization_analysis.txt'}")
    
    def run_analysis(self):
        """Run the complete optimization analysis."""
        results = self.run_optimization_analysis()
        if not results:
            return
        
        improvements = self.calculate_improvements(results)
        self.visualize_improvements(improvements)
        self.generate_report(results, improvements)

if __name__ == "__main__":
    analyzer = OptimizationAnalysis()
    analyzer.run_analysis()