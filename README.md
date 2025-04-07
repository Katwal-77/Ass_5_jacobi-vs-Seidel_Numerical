# Comparison of Jacobi and Gauss-Seidel Methods

**Author: Prem Katuwal (202424080129)**
Numerical Analysis Assignment

This project implements and compares the Jacobi and Gauss-Seidel iterative methods for solving systems of linear equations. The comparison focuses on:

1. CPU time (execution time)
2. Convergence rate
3. Accuracy of the solution
4. Number of iterations required for convergence

The project includes comprehensive benchmarks, advanced test cases, and a real-world application example.

## Background

### Jacobi Method

The Jacobi method is an iterative algorithm for solving a system of linear equations Ax = b. For each diagonal element of A, the method solves for the corresponding element of x, using the previous iteration's values for all other elements of x.

The iteration formula is:
```
x_i^(k+1) = (b_i - sum_{j≠i} A_ij * x_j^(k)) / A_ii
```

### Gauss-Seidel Method

The Gauss-Seidel method is similar to the Jacobi method but uses updated values as soon as they are available. This often leads to faster convergence.

The iteration formula is:
```
x_i^(k+1) = (b_i - sum_{j<i} A_ij * x_j^(k+1) - sum_{j>i} A_ij * x_j^(k)) / A_ii
```

## Implementation

The project consists of the following files:

### Core Implementation
- `iterative_solvers.py`: Implements both the Jacobi and Gauss-Seidel methods with detailed documentation

### Testing and Benchmarking
- `comparison.py`: Contains functions to generate basic test cases and compare the methods
- `main.py`: Main script to run the basic comparison
- `advanced_test.py`: Implements more challenging test cases (2D Poisson equation, nearly singular matrices)
- `comprehensive_benchmark.py`: Combines all test cases into a single comprehensive benchmark

### Analysis and Documentation
- `analysis.md`: Detailed analysis of the methods and results
- `summary.md`: Concise summary of key findings

### Real-World Application
- `real_world_example.py`: Demonstrates how to use the methods to solve a 2D steady-state heat equation

## Test Cases

### Basic Test Cases
1. Small well-conditioned system (10×10)
2. Medium-sized system with moderate condition number (50×50)
3. Larger system with higher condition number (100×100)
4. Tridiagonal system (common in PDEs) (100×100)

### Advanced Test Cases
5. 2D Poisson equation (225×225)
6. Nearly singular matrix (50×50)

### Real-World Application
- 2D steady-state heat equation with mixed boundary conditions

## Results

The project generates various plots and analysis files:

### Basic Comparison
- `convergence_comparison.png`: Shows the convergence history for both methods on basic test cases
- `performance_comparison.png`: Compares iterations and CPU time for basic test cases

### Advanced Tests
- `poisson_convergence.png`: Convergence history for the 2D Poisson equation
- `singular_convergence.png`: Convergence history for the nearly singular matrix

### Comprehensive Benchmark
- `all_convergence_comparison.png`: Convergence history for all test cases
- `all_performance_comparison.png`: Performance comparison for all test cases
- `speedup_comparison.png`: Speedup factors for all test cases
- `benchmark_results.csv`: Detailed benchmark results in CSV format

### Real-World Application
- `heat_distribution_jacobi.png`: Heat distribution computed using Jacobi method
- `heat_distribution_gauss_seidel.png`: Heat distribution computed using Gauss-Seidel method
- `heat_distribution_difference.png`: Difference between the two solutions
- `heat_equation_convergence.png`: Convergence history for the heat equation

## Running the Code

### Basic Comparison
```
python main.py
```

### Advanced Tests
```
python advanced_test.py
```

### Comprehensive Benchmark
```
python comprehensive_benchmark.py
```

### Real-World Example
```
python real_world_example.py
```

## Key Findings

1. **Gauss-Seidel requires fewer iterations** in all test cases (1.4× to 4.3× improvement)

2. **CPU time advantage is problem-dependent**:
   - Gauss-Seidel is faster for small to medium-sized dense systems
   - Jacobi can be more efficient for large sparse systems and special structures

3. **Both methods achieve similar accuracy** when they converge

4. **For real-world applications** like the heat equation, the choice depends on problem size and structure

## Requirements

- NumPy
- Matplotlib
- Pandas (for comprehensive benchmark)
