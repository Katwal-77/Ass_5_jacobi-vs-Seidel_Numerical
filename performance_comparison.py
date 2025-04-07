"""
Performance Comparison of Jacobi and Gauss-Seidel Methods

This script focuses specifically on comparing the performance of Jacobi and Gauss-Seidel methods
in terms of:
1. CPU time (execution time)
2. Accuracy (error relative to exact solution)

The comparison is performed on various test cases with different characteristics.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from iterative_solvers import jacobi_method, gauss_seidel_method


def generate_diagonally_dominant_matrix(n, condition_number=None):
    """
    Generate a random diagonally dominant matrix of size n×n.
    
    Parameters:
    -----------
    n : int
        Size of the matrix
    condition_number : float, optional
        Approximate condition number of the matrix
    
    Returns:
    --------
    np.ndarray
        Diagonally dominant matrix of shape (n, n)
    """
    # Generate a random matrix
    A = np.random.rand(n, n)
    
    # Make it diagonally dominant
    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i, :])) + np.random.rand()
    
    # Adjust condition number if specified
    if condition_number is not None:
        # Get SVD
        U, s, Vh = np.linalg.svd(A)
        
        # Calculate new singular values to achieve desired condition number
        s_new = np.linspace(condition_number, 1, n)
        
        # Reconstruct matrix with new singular values
        A = U @ np.diag(s_new) @ Vh
        
        # Ensure diagonal dominance again
        for i in range(n):
            row_sum = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
            A[i, i] = row_sum + np.random.rand() + 1.0
    
    return A


def create_test_cases():
    """
    Create a variety of test cases for comparing the methods.
    
    Returns:
    --------
    list
        List of test case dictionaries
    """
    test_cases = []
    
    # Test case 1: Small system (10×10)
    n = 10
    A = generate_diagonally_dominant_matrix(n)
    x_exact = np.ones(n)
    b = A @ x_exact
    test_cases.append({
        'A': A,
        'b': b,
        'x_exact': x_exact,
        'name': f"Small system ({n}×{n})",
        'size': n
    })
    
    # Test case 2: Medium system (50×50)
    n = 50
    A = generate_diagonally_dominant_matrix(n)
    x_exact = np.ones(n)
    b = A @ x_exact
    test_cases.append({
        'A': A,
        'b': b,
        'x_exact': x_exact,
        'name': f"Medium system ({n}×{n})",
        'size': n
    })
    
    # Test case 3: Large system (200×200)
    n = 200
    A = generate_diagonally_dominant_matrix(n)
    x_exact = np.ones(n)
    b = A @ x_exact
    test_cases.append({
        'A': A,
        'b': b,
        'x_exact': x_exact,
        'name': f"Large system ({n}×{n})",
        'size': n
    })
    
    # Test case 4: Tridiagonal system (100×100)
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
        'name': f"Tridiagonal system ({n}×{n})",
        'size': n
    })
    
    # Test case 5: Ill-conditioned system (50×50)
    n = 50
    A = generate_diagonally_dominant_matrix(n, condition_number=1000)
    x_exact = np.ones(n)
    b = A @ x_exact
    test_cases.append({
        'A': A,
        'b': b,
        'x_exact': x_exact,
        'name': f"Ill-conditioned system ({n}×{n})",
        'size': n
    })
    
    return test_cases


def compare_performance(test_cases, max_iterations=5000, tolerance=1e-8, num_runs=5):
    """
    Compare the performance of Jacobi and Gauss-Seidel methods on the given test cases.
    
    Parameters:
    -----------
    test_cases : list
        List of test case dictionaries
    max_iterations : int, optional
        Maximum number of iterations for both methods
    tolerance : float, optional
        Convergence tolerance for both methods
    num_runs : int, optional
        Number of runs to average for CPU time measurement
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing performance comparison results
    """
    results = []
    
    print("=" * 80)
    print("PERFORMANCE COMPARISON: JACOBI VS GAUSS-SEIDEL")
    print("=" * 80)
    print(f"Tolerance: {tolerance}")
    print(f"Maximum iterations: {max_iterations}")
    print(f"Number of runs for timing: {num_runs}")
    print("-" * 80)
    
    for test_case in test_cases:
        A = test_case['A']
        b = test_case['b']
        x_exact = test_case['x_exact']
        name = test_case['name']
        
        print(f"\nTest Case: {name}")
        print(f"Matrix size: {A.shape[0]}×{A.shape[1]}")
        
        # Run Jacobi method multiple times to get average CPU time
        jacobi_times = []
        for _ in range(num_runs):
            start_time = time.time()
            jacobi_result = jacobi_method(A, b, max_iterations=max_iterations, 
                                         tolerance=tolerance)
            jacobi_times.append(time.time() - start_time)
        
        jacobi_avg_time = np.mean(jacobi_times)
        jacobi_std_time = np.std(jacobi_times)
        
        # Run Gauss-Seidel method multiple times to get average CPU time
        gauss_seidel_times = []
        for _ in range(num_runs):
            start_time = time.time()
            gauss_seidel_result = gauss_seidel_method(A, b, max_iterations=max_iterations, 
                                                    tolerance=tolerance)
            gauss_seidel_times.append(time.time() - start_time)
        
        gauss_seidel_avg_time = np.mean(gauss_seidel_times)
        gauss_seidel_std_time = np.std(gauss_seidel_times)
        
        # Calculate errors
        jacobi_error = np.linalg.norm(jacobi_result['solution'] - x_exact) / np.linalg.norm(x_exact)
        gauss_seidel_error = np.linalg.norm(gauss_seidel_result['solution'] - x_exact) / np.linalg.norm(x_exact)
        
        # Calculate speedup
        time_speedup = jacobi_avg_time / gauss_seidel_avg_time
        faster_method = "Gauss-Seidel" if time_speedup > 1 else "Jacobi"
        speedup_factor = max(time_speedup, 1/time_speedup)
        
        # Print results
        print("\nPerformance Results:")
        print(f"Jacobi: {jacobi_result['iterations']} iterations, {jacobi_avg_time:.6f} ± {jacobi_std_time:.6f} seconds, error: {jacobi_error:.6e}")
        print(f"Gauss-Seidel: {gauss_seidel_result['iterations']} iterations, {gauss_seidel_avg_time:.6f} ± {gauss_seidel_std_time:.6f} seconds, error: {gauss_seidel_error:.6e}")
        print(f"Faster method: {faster_method} ({speedup_factor:.2f}× faster)")
        print(f"Accuracy comparison: {'Jacobi' if jacobi_error < gauss_seidel_error else 'Gauss-Seidel'} is more accurate by a factor of {max(jacobi_error, gauss_seidel_error) / min(jacobi_error, gauss_seidel_error):.2f}×")
        
        # Store results
        results.append({
            'Test Case': name,
            'Matrix Size': f"{A.shape[0]}×{A.shape[1]}",
            'Jacobi Iterations': jacobi_result['iterations'],
            'Jacobi Time (s)': f"{jacobi_avg_time:.6f} ± {jacobi_std_time:.6f}",
            'Jacobi Error': f"{jacobi_error:.6e}",
            'Gauss-Seidel Iterations': gauss_seidel_result['iterations'],
            'Gauss-Seidel Time (s)': f"{gauss_seidel_avg_time:.6f} ± {gauss_seidel_std_time:.6f}",
            'Gauss-Seidel Error': f"{gauss_seidel_error:.6e}",
            'Faster Method': faster_method,
            'Time Speedup': f"{speedup_factor:.2f}×",
            'More Accurate Method': 'Jacobi' if jacobi_error < gauss_seidel_error else 'Gauss-Seidel',
            'Accuracy Ratio': f"{max(jacobi_error, gauss_seidel_error) / min(jacobi_error, gauss_seidel_error):.2f}×",
            '_jacobi_time': jacobi_avg_time,
            '_gauss_seidel_time': gauss_seidel_avg_time,
            '_jacobi_error': jacobi_error,
            '_gauss_seidel_error': gauss_seidel_error,
            '_jacobi_iterations': jacobi_result['iterations'],
            '_gauss_seidel_iterations': gauss_seidel_result['iterations']
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    return df


def plot_performance_comparison(df):
    """
    Create plots comparing CPU time and accuracy.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing performance comparison results
    """
    # Extract data for plotting
    test_cases = df['Test Case'].tolist()
    jacobi_times = [row['_jacobi_time'] for _, row in df.iterrows()]
    gauss_seidel_times = [row['_gauss_seidel_time'] for _, row in df.iterrows()]
    jacobi_errors = [row['_jacobi_error'] for _, row in df.iterrows()]
    gauss_seidel_errors = [row['_gauss_seidel_error'] for _, row in df.iterrows()]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot CPU time comparison
    x = np.arange(len(test_cases))
    width = 0.35
    
    ax1.bar(x - width/2, jacobi_times, width, label='Jacobi')
    ax1.bar(x + width/2, gauss_seidel_times, width, label='Gauss-Seidel')
    
    ax1.set_title('CPU Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_cases, rotation=45, ha='right')
    ax1.set_ylabel('Time (seconds)')
    ax1.legend()
    
    # Add values on top of bars
    for i, v in enumerate(jacobi_times):
        ax1.text(i - width/2, v + 0.01*max(jacobi_times + gauss_seidel_times), 
                f"{v:.4f}", ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(gauss_seidel_times):
        ax1.text(i + width/2, v + 0.01*max(jacobi_times + gauss_seidel_times), 
                f"{v:.4f}", ha='center', va='bottom', fontsize=8)
    
    # Plot accuracy comparison
    ax2.bar(x - width/2, jacobi_errors, width, label='Jacobi')
    ax2.bar(x + width/2, gauss_seidel_errors, width, label='Gauss-Seidel')
    
    ax2.set_title('Accuracy Comparison (Lower is Better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(test_cases, rotation=45, ha='right')
    ax2.set_ylabel('Relative Error')
    ax2.set_yscale('log')  # Use log scale for errors
    ax2.legend()
    
    # Add values on top of bars
    for i, v in enumerate(jacobi_errors):
        ax2.text(i - width/2, v * 1.1, f"{v:.1e}", ha='center', va='bottom', fontsize=8, rotation=90)
    for i, v in enumerate(gauss_seidel_errors):
        ax2.text(i + width/2, v * 1.1, f"{v:.1e}", ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.tight_layout()
    plt.savefig('cpu_accuracy_comparison.png', dpi=300)
    print("\nPerformance comparison plot saved as 'cpu_accuracy_comparison.png'")
    
    # Create a second figure for speedup and accuracy ratio
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Calculate speedup factors (>1 means Gauss-Seidel is faster, <1 means Jacobi is faster)
    speedups = [j/g if j > g else -g/j for j, g in zip(jacobi_times, gauss_seidel_times)]
    
    # Plot speedup comparison
    colors = ['green' if s > 0 else 'red' for s in speedups]
    ax1.bar(x, [abs(s) for s in speedups], color=colors)
    
    ax1.set_title('Time Speedup Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_cases, rotation=45, ha='right')
    ax1.set_ylabel('Speedup Factor')
    ax1.axhline(y=1, color='k', linestyle='--', alpha=0.3)
    
    # Add annotations
    for i, s in enumerate(speedups):
        method = "GS faster" if s > 0 else "J faster"
        ax1.text(i, abs(s) + 0.1, f"{abs(s):.2f}× ({method})", ha='center', va='bottom', fontsize=8)
    
    # Calculate accuracy ratios (>1 means one method is more accurate)
    accuracy_ratios = [j/g if j < g else -g/j for j, g in zip(jacobi_errors, gauss_seidel_errors)]
    
    # Plot accuracy ratio comparison
    colors = ['green' if s < 0 else 'red' for s in accuracy_ratios]
    ax2.bar(x, [abs(s) for s in accuracy_ratios], color=colors)
    
    ax2.set_title('Accuracy Ratio Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(test_cases, rotation=45, ha='right')
    ax2.set_ylabel('Accuracy Ratio')
    ax2.axhline(y=1, color='k', linestyle='--', alpha=0.3)
    
    # Add annotations
    for i, s in enumerate(accuracy_ratios):
        method = "J more accurate" if s < 0 else "GS more accurate"
        ax2.text(i, abs(s) + 0.1, f"{abs(s):.2f}× ({method})", ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('speedup_accuracy_ratio.png', dpi=300)
    print("Speedup and accuracy ratio plot saved as 'speedup_accuracy_ratio.png'")


def main():
    """
    Main function to run the performance comparison.
    """
    print("Starting performance comparison of Jacobi and Gauss-Seidel methods...")
    
    # Create test cases
    test_cases = create_test_cases()
    
    # Compare performance
    results_df = compare_performance(test_cases)
    
    # Plot results
    plot_performance_comparison(results_df)
    
    # Save results to CSV
    results_df.drop(columns=[col for col in results_df.columns if col.startswith('_')]).to_csv('performance_results.csv', index=False)
    print("Performance results saved to 'performance_results.csv'")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY OF PERFORMANCE COMPARISON")
    print("=" * 80)
    summary_df = results_df.drop(columns=[col for col in results_df.columns if col.startswith('_')])
    print(summary_df.to_string(index=False))
    
    print("\nPerformance comparison completed!")


if __name__ == "__main__":
    main()
