"""
Comparison of Jacobi and Gauss-Seidel Methods

This script compares the performance of the Jacobi and Gauss-Seidel methods
for solving systems of linear equations in terms of:
1. CPU time
2. Convergence rate
3. Accuracy

Various test cases are generated to evaluate the methods under different conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from iterative_solvers import jacobi_method, gauss_seidel_method, check_convergence_conditions
from typing import Dict, List, Tuple


def generate_diagonally_dominant_matrix(n: int, condition_number: float = None) -> np.ndarray:
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


def generate_test_cases() -> List[Dict]:
    """
    Generate a variety of test cases for comparing the methods.
    
    Returns:
    --------
    List[Dict]
        List of test case dictionaries, each containing:
        - 'A': coefficient matrix
        - 'b': right-hand side vector
        - 'x_exact': exact solution (if known)
        - 'name': descriptive name of the test case
        - 'size': size of the system
        - 'condition_number': condition number of A (if known)
    """
    test_cases = []
    
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
    
    return test_cases


def compare_methods(test_cases: List[Dict], max_iterations: int = 1000, 
                   tolerance: float = 1e-8) -> Dict:
    """
    Compare Jacobi and Gauss-Seidel methods on the given test cases.
    
    Parameters:
    -----------
    test_cases : List[Dict]
        List of test case dictionaries
    max_iterations : int, optional
        Maximum number of iterations for both methods
    tolerance : float, optional
        Convergence tolerance for both methods
    
    Returns:
    --------
    Dict
        Dictionary containing comparison results
    """
    results = {}
    
    for i, test_case in enumerate(test_cases):
        A = test_case['A']
        b = test_case['b']
        x_exact = test_case.get('x_exact')
        name = test_case['name']
        
        print(f"\nTest Case {i+1}: {name}")
        print(f"Matrix size: {A.shape[0]}×{A.shape[1]}")
        print(f"Condition number: {test_case.get('condition_number', 'Unknown')}")
        
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
        
        # Print performance metrics
        print("\nPerformance Comparison:")
        print(f"Jacobi: {jacobi_result['iterations']} iterations, {jacobi_result['cpu_time']:.6f} seconds")
        print(f"Gauss-Seidel: {gauss_seidel_result['iterations']} iterations, {gauss_seidel_result['cpu_time']:.6f} seconds")
        print(f"Speedup (iterations): {jacobi_result['iterations'] / gauss_seidel_result['iterations']:.2f}x")
        print(f"Speedup (time): {jacobi_result['cpu_time'] / gauss_seidel_result['cpu_time']:.2f}x")
        
        # Store results
        results[name] = {
            'jacobi': jacobi_result,
            'gauss_seidel': gauss_seidel_result,
            'conditions': conditions
        }
        
        if x_exact is not None:
            results[name]['jacobi']['solution_error'] = jacobi_solution_error
            results[name]['gauss_seidel']['solution_error'] = gauss_seidel_solution_error
    
    return results


def plot_convergence(results: Dict):
    """
    Plot convergence history for both methods on each test case.
    
    Parameters:
    -----------
    results : Dict
        Dictionary containing comparison results
    """
    n_cases = len(results)
    fig, axes = plt.subplots(n_cases, 1, figsize=(10, 4 * n_cases))
    
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
    plt.savefig('convergence_comparison.png')
    print("\nConvergence plot saved as 'convergence_comparison.png'")


def plot_performance_summary(results: Dict):
    """
    Create summary plots comparing performance metrics.
    
    Parameters:
    -----------
    results : Dict
        Dictionary containing comparison results
    """
    # Extract data for plotting
    names = list(results.keys())
    jacobi_iterations = [results[name]['jacobi']['iterations'] for name in names]
    gauss_seidel_iterations = [results[name]['gauss_seidel']['iterations'] for name in names]
    
    jacobi_times = [results[name]['jacobi']['cpu_time'] for name in names]
    gauss_seidel_times = [results[name]['gauss_seidel']['cpu_time'] for name in names]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot iterations comparison
    x = np.arange(len(names))
    width = 0.35
    
    ax1.bar(x - width/2, jacobi_iterations, width, label='Jacobi')
    ax1.bar(x + width/2, gauss_seidel_iterations, width, label='Gauss-Seidel')
    
    ax1.set_title('Number of Iterations')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Case {i+1}" for i in range(len(names))], rotation=45)
    ax1.set_ylabel('Iterations')
    ax1.legend()
    
    # Plot CPU time comparison
    ax2.bar(x - width/2, jacobi_times, width, label='Jacobi')
    ax2.bar(x + width/2, gauss_seidel_times, width, label='Gauss-Seidel')
    
    ax2.set_title('CPU Time')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Case {i+1}" for i in range(len(names))], rotation=45)
    ax2.set_ylabel('Time (seconds)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    print("Performance summary plot saved as 'performance_comparison.png'")


def main():
    """
    Main function to run the comparison.
    """
    print("Comparing Jacobi and Gauss-Seidel Methods for Solving Linear Systems")
    print("=" * 70)
    
    # Generate test cases
    print("\nGenerating test cases...")
    test_cases = generate_test_cases()
    
    # Compare methods
    print("\nRunning comparison...")
    results = compare_methods(test_cases)
    
    # Plot results
    print("\nGenerating plots...")
    plot_convergence(results)
    plot_performance_summary(results)
    
    print("\nComparison completed!")


if __name__ == "__main__":
    main()
