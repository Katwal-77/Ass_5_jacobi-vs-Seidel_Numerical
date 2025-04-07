"""
Comprehensive Benchmark for Jacobi and Gauss-Seidel Methods

This script provides a comprehensive benchmark of the Jacobi and Gauss-Seidel methods
for solving systems of linear equations. It combines all test cases from the basic
and advanced tests into a single benchmark.

The benchmark measures:
1. CPU time
2. Number of iterations
3. Accuracy
4. Convergence rate

Results are presented in tables and visualized through plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from iterative_solvers import jacobi_method, gauss_seidel_method, check_convergence_conditions
from comparison import generate_diagonally_dominant_matrix
from advanced_test import create_poisson_matrix, create_nearly_singular_matrix


def generate_all_test_cases():
    """
    Generate all test cases for the benchmark.
    
    Returns:
    --------
    list
        List of test case dictionaries
    """
    test_cases = []
    
    # Basic Test Cases
    
    # Test case 1: Small well-conditioned system
    n = 10
    A = generate_diagonally_dominant_matrix(n, condition_number=10)
    x_exact = np.ones(n)
    b = A @ x_exact
    test_cases.append({
        'A': A,
        'b': b,
        'x_exact': x_exact,
        'name': f"Well-conditioned {n}×{n} system",
        'size': n,
        'condition_number': np.linalg.cond(A)
    })
    
    # Test case 2: Medium-sized system with moderate condition number
    n = 50
    A = generate_diagonally_dominant_matrix(n, condition_number=100)
    x_exact = np.ones(n)
    b = A @ x_exact
    test_cases.append({
        'A': A,
        'b': b,
        'x_exact': x_exact,
        'name': f"Moderate-conditioned {n}×{n} system",
        'size': n,
        'condition_number': np.linalg.cond(A)
    })
    
    # Test case 3: Larger system with higher condition number
    n = 100
    A = generate_diagonally_dominant_matrix(n, condition_number=1000)
    x_exact = np.ones(n)
    b = A @ x_exact
    test_cases.append({
        'A': A,
        'b': b,
        'x_exact': x_exact,
        'name': f"Ill-conditioned {n}×{n} system",
        'size': n,
        'condition_number': np.linalg.cond(A)
    })
    
    # Test case 4: Tridiagonal system (common in PDEs)
    n = 100
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 4.0
        if i > 0:
            A[i, i-1] = -1.0
        if i < n-1:
            A[i, i+1] = -1.0
    x_exact = np.ones(n)
    b = A @ x_exact
    test_cases.append({
        'A': A,
        'b': b,
        'x_exact': x_exact,
        'name': f"Tridiagonal {n}×{n} system",
        'size': n,
        'condition_number': np.linalg.cond(A)
    })
    
    # Advanced Test Cases
    
    # Test case 5: 2D Poisson equation
    n = 15  # 15×15 grid (225×225 matrix)
    A = create_poisson_matrix(n)
    N = A.shape[0]
    x_exact = np.ones(N)
    b = A @ x_exact
    test_cases.append({
        'A': A,
        'b': b,
        'x_exact': x_exact,
        'name': f"2D Poisson {N}×{N} system",
        'size': N,
        'condition_number': np.linalg.cond(A)
    })
    
    # Test case 6: Nearly singular matrix
    n = 50
    A = create_nearly_singular_matrix(n)
    x_exact = np.ones(n)
    b = A @ x_exact
    test_cases.append({
        'A': A,
        'b': b,
        'x_exact': x_exact,
        'name': f"Nearly singular {n}×{n} system",
        'size': n,
        'condition_number': np.linalg.cond(A)
    })
    
    return test_cases


def run_benchmark(test_cases, max_iterations=2000, tolerance=1e-8):
    """
    Run the benchmark on all test cases.
    
    Parameters:
    -----------
    test_cases : list
        List of test case dictionaries
    max_iterations : int, optional
        Maximum number of iterations
    tolerance : float, optional
        Convergence tolerance
    
    Returns:
    --------
    dict
        Dictionary containing benchmark results
    """
    results = {}
    summary_data = []
    
    print("\n" + "="*80)
    print("COMPREHENSIVE BENCHMARK: JACOBI VS GAUSS-SEIDEL METHODS")
    print("="*80)
    
    for i, test_case in enumerate(test_cases):
        A = test_case['A']
        b = test_case['b']
        x_exact = test_case.get('x_exact')
        name = test_case['name']
        
        print(f"\nTest Case {i+1}: {name}")
        print(f"Matrix size: {A.shape[0]}×{A.shape[1]}")
        print(f"Condition number: {test_case.get('condition_number', 'Unknown'):.2e}")
        
        # Check convergence conditions
        conditions = check_convergence_conditions(A)
        print("Convergence conditions:")
        for condition, satisfied in conditions.items():
            print(f"  - {condition}: {'Yes' if satisfied else 'No'}")
        
        # Run Jacobi method
        jacobi_result = jacobi_method(A, b, max_iterations=max_iterations, 
                                     tolerance=tolerance, return_history=True)
        
        # Run Gauss-Seidel method
        gauss_seidel_result = gauss_seidel_method(A, b, max_iterations=max_iterations, 
                                                tolerance=tolerance, return_history=True)
        
        # Calculate error with respect to exact solution if available
        if x_exact is not None:
            jacobi_solution_error = np.linalg.norm(jacobi_result['solution'] - x_exact) / np.linalg.norm(x_exact)
            gauss_seidel_solution_error = np.linalg.norm(gauss_seidel_result['solution'] - x_exact) / np.linalg.norm(x_exact)
            
            print(f"Jacobi solution error: {jacobi_solution_error:.6e}")
            print(f"Gauss-Seidel solution error: {gauss_seidel_solution_error:.6e}")
            
            jacobi_result['solution_error'] = jacobi_solution_error
            gauss_seidel_result['solution_error'] = gauss_seidel_solution_error
        
        # Print performance metrics
        print("\nPerformance Comparison:")
        print(f"Jacobi: {jacobi_result['iterations']} iterations, {jacobi_result['cpu_time']:.6f} seconds")
        print(f"Gauss-Seidel: {gauss_seidel_result['iterations']} iterations, {gauss_seidel_result['cpu_time']:.6f} seconds")
        
        iteration_speedup = jacobi_result['iterations'] / gauss_seidel_result['iterations']
        time_speedup = jacobi_result['cpu_time'] / gauss_seidel_result['cpu_time']
        
        print(f"Speedup (iterations): {iteration_speedup:.2f}x")
        print(f"Speedup (time): {time_speedup:.2f}x")
        
        # Store results
        results[name] = {
            'jacobi': jacobi_result,
            'gauss_seidel': gauss_seidel_result,
            'conditions': conditions,
            'iteration_speedup': iteration_speedup,
            'time_speedup': time_speedup
        }
        
        # Add to summary data
        summary_data.append({
            'Test Case': name,
            'Matrix Size': f"{A.shape[0]}×{A.shape[1]}",
            'Condition Number': f"{test_case.get('condition_number', 'Unknown'):.2e}",
            'Jacobi Iterations': jacobi_result['iterations'],
            'Gauss-Seidel Iterations': gauss_seidel_result['iterations'],
            'Iteration Speedup': f"{iteration_speedup:.2f}×",
            'Jacobi Time (s)': f"{jacobi_result['cpu_time']:.6f}",
            'Gauss-Seidel Time (s)': f"{gauss_seidel_result['cpu_time']:.6f}",
            'Time Speedup': f"{time_speedup:.2f}× {'(GS)' if time_speedup > 1 else '(J)'}",
            'Jacobi Error': f"{jacobi_result.get('solution_error', 'N/A'):.2e}",
            'Gauss-Seidel Error': f"{gauss_seidel_result.get('solution_error', 'N/A'):.2e}"
        })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    return results, summary_df


def plot_all_convergence(results):
    """
    Plot convergence history for all test cases.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing benchmark results
    """
    n_cases = len(results)
    fig, axes = plt.subplots(n_cases, 1, figsize=(12, 5 * n_cases))
    
    if n_cases == 1:
        axes = [axes]
    
    for i, (name, result) in enumerate(results.items()):
        ax = axes[i]
        
        # Get residual histories
        jacobi_residuals = result['jacobi']['residuals']
        gauss_seidel_residuals = result['gauss_seidel']['residuals']
        
        # Plot on log scale
        ax.semilogy(range(1, len(jacobi_residuals) + 1), jacobi_residuals, 'b-', label='Jacobi')
        ax.semilogy(range(1, len(gauss_seidel_residuals) + 1), gauss_seidel_residuals, 'r-', label='Gauss-Seidel')
        
        ax.set_title(f"Convergence History: {name}")
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Relative Residual (log scale)')
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('all_convergence_comparison.png')
    print("\nConvergence plot saved as 'all_convergence_comparison.png'")


def plot_performance_summary(summary_df):
    """
    Create summary plots comparing performance metrics.
    
    Parameters:
    -----------
    summary_df : pandas.DataFrame
        DataFrame containing summary data
    """
    # Extract data for plotting
    test_cases = summary_df['Test Case'].tolist()
    jacobi_iterations = summary_df['Jacobi Iterations'].tolist()
    gauss_seidel_iterations = summary_df['Gauss-Seidel Iterations'].tolist()
    
    jacobi_times = [float(t) for t in summary_df['Jacobi Time (s)'].tolist()]
    gauss_seidel_times = [float(t) for t in summary_df['Gauss-Seidel Time (s)'].tolist()]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot iterations comparison
    x = np.arange(len(test_cases))
    width = 0.35
    
    ax1.bar(x - width/2, jacobi_iterations, width, label='Jacobi')
    ax1.bar(x + width/2, gauss_seidel_iterations, width, label='Gauss-Seidel')
    
    ax1.set_title('Number of Iterations')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Case {i+1}" for i in range(len(test_cases))], rotation=45)
    ax1.set_ylabel('Iterations')
    ax1.legend()
    
    # Plot CPU time comparison
    ax2.bar(x - width/2, jacobi_times, width, label='Jacobi')
    ax2.bar(x + width/2, gauss_seidel_times, width, label='Gauss-Seidel')
    
    ax2.set_title('CPU Time')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Case {i+1}" for i in range(len(test_cases))], rotation=45)
    ax2.set_ylabel('Time (seconds)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('all_performance_comparison.png')
    print("Performance summary plot saved as 'all_performance_comparison.png'")
    
    # Create a second figure for speedup comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract speedup data
    iteration_speedups = [float(s.split('×')[0]) for s in summary_df['Iteration Speedup'].tolist()]
    time_speedups = [float(s.split('×')[0]) for s in summary_df['Time Speedup'].tolist()]
    
    # Adjust time speedups to show which method is faster
    adjusted_time_speedups = []
    for s, full_s in zip(time_speedups, summary_df['Time Speedup'].tolist()):
        if '(J)' in full_s:
            adjusted_time_speedups.append(-1/s)  # Negative for Jacobi being faster
        else:
            adjusted_time_speedups.append(s)  # Positive for Gauss-Seidel being faster
    
    # Plot speedup comparison
    ax.bar(x - width/2, iteration_speedups, width, label='Iteration Speedup (GS/J)')
    ax.bar(x + width/2, adjusted_time_speedups, width, label='Time Speedup')
    
    # Add a horizontal line at y=1 and y=-1
    ax.axhline(y=1, color='k', linestyle='--', alpha=0.3)
    ax.axhline(y=-1, color='k', linestyle='--', alpha=0.3)
    
    # Add text annotations
    ax.text(len(test_cases)-1, 1.1, 'Gauss-Seidel faster', ha='right')
    ax.text(len(test_cases)-1, -1.1, 'Jacobi faster', ha='right')
    
    ax.set_title('Performance Speedup Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Case {i+1}" for i in range(len(test_cases))], rotation=45)
    ax.set_ylabel('Speedup Factor')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('speedup_comparison.png')
    print("Speedup comparison plot saved as 'speedup_comparison.png'")


def save_summary_table(summary_df):
    """
    Save the summary table to a CSV file and print it.
    
    Parameters:
    -----------
    summary_df : pandas.DataFrame
        DataFrame containing summary data
    """
    # Save to CSV
    summary_df.to_csv('benchmark_results.csv', index=False)
    print("\nBenchmark results saved to 'benchmark_results.csv'")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    print(summary_df.to_string(index=False))


def main():
    """
    Main function to run the comprehensive benchmark.
    """
    print("Starting comprehensive benchmark of Jacobi and Gauss-Seidel methods...")
    
    # Generate all test cases
    test_cases = generate_all_test_cases()
    
    # Run benchmark
    results, summary_df = run_benchmark(test_cases)
    
    # Plot results
    plot_all_convergence(results)
    plot_performance_summary(summary_df)
    
    # Save and print summary table
    save_summary_table(summary_df)
    
    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()
