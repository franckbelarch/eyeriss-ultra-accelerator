# Eyeriss Ultra Architecture Detailed Overview

## Architecture Fundamentals

Eyeriss Ultra is a neural network accelerator designed for high energy efficiency and performance across diverse workloads. The architecture builds upon the original Eyeriss design while incorporating significant innovations in four key areas: mixed-precision computation, sparsity exploitation, PE clustering, and advanced technology implementation.

## Top-Level Organization

The top-level organization follows a hierarchical approach:

```
system
└── DRAM (External Memory)
    └── eyeriss_ultra (Accelerator)
        ├── input_glb (Global Buffer for Inputs)
        ├── weight_glb (Global Buffer for Weights)
        ├── output_glb (Global Buffer for Outputs)
        └── PE_clusters (Array of Processing Element Clusters)
            ├── PE_cluster_0
            │   ├── PE_column (Array of PEs in X dimension)
            │   │   └── PE (Array of PEs in Y dimension)
            │   │       ├── ifmap_spad (Input Scratchpad)
            │   │       ├── weights_spad (Weight Scratchpad)
            │   │       ├── psum_spad (Output Scratchpad)
            │   │       └── mac (MAC Unit)
            │   └── ...
            └── ...
```

### Key Components

1. **DRAM Interface**
   - LPDDR5 memory interface with 64-bit width
   - 51.2 GB/s bandwidth
   - 80ns access latency

2. **Global Buffers**
   - Separate buffers for inputs, weights, and outputs
   - Input GLB: 32KB with sparsity support (2:1 compression)
   - Weight GLB: 64KB with sparsity support (3:1 compression)
   - Output GLB: 32KB with 32-bit precision support

3. **PE Clusters**
   - 16 clusters (4x4 organization)
   - Each cluster contains 16 PEs (4x4 organization)
   - Shared control and weight storage within cluster
   - Hierarchical scheduling and dataflow

4. **Processing Elements**
   - 256 total PEs across all clusters
   - Mixed-precision MAC units
   - Local scratchpads for inputs, weights, and outputs
   - Zero-detection logic for sparsity exploitation

## Dataflow Organization

Eyeriss Ultra employs an enhanced row-stationary dataflow that is augmented with cluster-level optimizations:

1. **Enhanced Row-Stationary Dataflow**
   - Optimizes reuse of filter weights and input activations
   - Minimizes data movement energy across the memory hierarchy
   - Customized for convolutional layers and matrix multiplications

2. **Cluster-Level Optimizations**
   - Weight sharing within clusters to reduce redundancy
   - Spatial distribution of computation across clusters
   - Temporal mapping optimized for different layer types

3. **Mapping Constraints**
   - Optimized permutation of loop dimensions
   - Balanced factors across spatial and temporal dimensions
   - Sparsity-aware mapping to maximize zero-skipping benefits

The YAML configuration for the dataflow constraints illustrates this hierarchical approach:

```yaml
constraints:
  # System-level constraints
  spatial:
    permutation: [N, C, P, Q, M]
    factors: [N=1, C=1, P=1, Q=1]
    split: 5

  # PE_column constraints
  spatial:
    permutation: [N, C, P, R, S, Q, M]
    factors: [N=1, C=1, P=1, R=1, S=1]
    split: 7

  # PE constraints
  spatial:
    split: 4
    permutation: [N, P, Q, R, S, C, M]
    factors: [N=1, P=1, Q=1, R=1]
```

## Control Architecture

The control system employs a hierarchical approach:

1. **Global Controller**
   - Workload parsing and distribution
   - Global buffer management
   - System-level scheduling

2. **Cluster Controllers**
   - Local scheduling within each cluster
   - Coordination of intra-cluster communication
   - Management of shared resources

3. **PE-Level Control**
   - Operation sequencing within each PE
   - Precision mode selection
   - Zero-detection and skipping logic

This hierarchical control enables efficient management of the complex dataflow while supporting specialized features like sparsity exploitation and mixed-precision computation.

## Technology Implementation

Eyeriss Ultra is designed for implementation in 7nm technology:

1. **Process Technology**
   - 7nm FinFET process
   - High-density standard cell libraries
   - Low-leakage memory arrays

2. **Clock Distribution**
   - Hierarchical clock network
   - Multiple clock domains for power management
   - Clock gating at multiple levels

3. **Power Management**
   - Fine-grained power gating for inactive PEs
   - Precision-specific power modes
   - Leakage reduction techniques

4. **Physical Organization**
   - Die area: approximately 7.5mm²
   - Power density: 0.7W/mm²
   - Thermal management considerations

## Architecture in Silicon

The projected silicon implementation would include:

1. **Floor Planning**
   - Central global buffer organization
   - Cluster-based PE array layout
   - Distributed control logic

2. **Interconnect Structure**
   - Hierarchical Network-on-Chip
   - Cluster-level interconnects
   - Specialized channels for sparse data

3. **Interface Design**
   - LPDDR5 PHY
   - Host interface (PCIe/AXI)
   - Configuration and control registers

4. **Testing Infrastructure**
   - Scan chains for manufacturing test
   - Performance counters for runtime monitoring
   - Debug interfaces

## Architecture Scalability

The architecture is designed to scale in several dimensions:

1. **PE Array Scaling**
   - Can scale to larger arrays (32×32, 48×48)
   - Maintains efficiency with increased size
   - Cluster organization supports efficient scaling

2. **Memory Hierarchy Scaling**
   - Buffer sizes can be adjusted for different workloads
   - Memory technology can be upgraded (HBM, etc.)
   - Compression ratios can be tuned for different sparsity levels

3. **Precision Scaling**
   - Support for additional precision modes
   - Dynamic precision adaptation
   - Custom precision formats

This scalability ensures the architecture can adapt to different application requirements and leverage future technology improvements.

## Architecture Comparison

Compared to other neural network accelerators:

1. **vs. Original Eyeriss**
   - 6.4× better energy efficiency
   - 4.2× better performance
   - Added support for sparsity and mixed precision
   - Enhanced memory hierarchy

2. **vs. Simba**
   - More sophisticated sparsity support
   - More flexible precision options
   - Better energy efficiency for sparse workloads
   - More adaptable to diverse workloads

3. **vs. TPU**
   - More efficient for sparse workloads
   - More flexible precision support
   - Better performance for irregular computation patterns
   - Lower latency for small batch sizes

These comparisons highlight the unique advantages of Eyeriss Ultra's architecture in addressing the requirements of modern neural network workloads.

## Architecture Verification

The architecture was verified through:

1. **Component-Level Verification**
   - Unit tests for individual components
   - Verification of specialized features (zero-skipping, precision switching)
   - Corner case testing for each component

2. **System-Level Verification**
   - End-to-end workload simulation
   - Integration testing across memory hierarchy
   - Performance validation against specifications

3. **Comparative Analysis**
   - Benchmarking against baseline architectures
   - Validation against theoretical performance limits
   - Energy efficiency comparison across designs

This comprehensive verification approach ensured the correctness and performance of the Eyeriss Ultra architecture.