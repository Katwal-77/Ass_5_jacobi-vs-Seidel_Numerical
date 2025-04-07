"""
Advanced Test Cases for Jacobi and Gauss-Seidel Methods

This script provides more challenging test cases to further explore the
differences between Jacobi and Gauss-Seidel methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from iterative_solvers import jacobi_method, gauss_seidel_method, check_convergence_conditions


def create_poisson_matrix(n):
    """
    Create a matrix for the 2D Poisson equation on an n×n grid.
    
    This creates a matrix for the finite difference discretization of
    the Poisson equation with Dirichlet boundary conditions.
    
    Parameters:
    -----------
    n : int
        Number of interior points in each dimension
    
    Returns:
    --------
    np.ndarray
        Matrix of shape (n²×n²)
    """
    N = n**2
    A = np.zeros((N, N))
    
    for i in range(N):
        A[i, i] = 4
        
        # Connect to neighbors
        if i % n != 0:  # Not on left boundary
            A[i, i-1] = -1
        if (i+1) % n != 0:  # Not on right boundary
            A[i, i+1] = -1
        if i >= n:  # Not on top boundary
            A[i, i-n] = -1
        if i < N-n:  # Not on bottom boundary
            A[i, i+n] = -1
    
    return A


def create_nearly_singular_matrix(n, epsilon=1e-3):
    """
    Create a nearly singular matrix with a small eigenvalue.
    
    Parameters:
    -----------
    n : int
        Size of the matrix
    epsilon : float
        Small value to control how close to singular the matrix is
    
    Returns:
    --------
    np.ndarray
        Nearly singular matrix of shape (n×n)
    """
    # Create a random orthogonal matrix
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    
    # Create diagonal matrix with one small value
    D = np.eye(n)
    D[n-1, n-1] = epsilon
    
    # Create the nearly singular matrix
    A = Q @ D @ Q.T
    
    # Make it diagonally dominant to ensure convergence
    for i in range(n):
        row_sum = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
        A[i, i] = row_sum + 1.0
    
    return A


def run_advanced_tests():
    """
    Run advanced test cases and compare the methods.
    """
    print("\n" + "="*70)
    print("ADVANCED TEST CASES")
    print("="*70)
    
    # Test Case 1: 2D Poisson equation
    print("\nTest Case: 2D Poisson Equation (15×15 grid)")
    n = 15  # 15×15 grid (225×225 matrix)
    A = create_poisson_matrix(n)
    N = A.shape[0]
    
    # Create exact solution and right-hand side
    x_exact = np.ones(N)
    b = A @ x_exact
    
    print(f"Matrix size: {N}×{N}")
    print(f"Condition number: {np.linalg.cond(A):.2e}")
    
    # Check convergence conditions
    conditions = check_convergence_conditions(A)
    print("Convergence conditions:")
    for condition, satisfied in conditions.items():
        print(f"  - {condition}: {'Yes' if satisfied else 'No'}")
    
    # Run Jacobi method
    jacobi_result = jacobi_method(A, b, max_iterations=2000, 
                                 tolerance=1e-8, return_history=True)
    
    # Run Gauss-Seidel method
    gauss_seidel_result = gauss_seidel_method(A, b, max_iterations=2000, 
                                            tolerance=1e-8, return_history=True)
    
    # Calculate error with respect to exact solution
    jacobi_error = np.linalg.norm(jacobi_result['solution'] - x_exact) / np.linalg.norm(x_exact)
    gauss_seidel_error = np.linalg.norm(gauss_seidel_result['solution'] - x_exact) / np.linalg.norm(x_exact)
    
    print(f"Jacobi solution error: {jacobi_error:.6e}")
    print(f"Gauss-Seidel solution error: {gauss_seidel_error:.6e}")
    
    # Print performance metrics
    print("\nPerformance Comparison:")
    print(f"Jacobi: {jacobi_result['iterations']} iterations, {jacobi_result['cpu_time']:.6f} seconds")
    print(f"Gauss-Seidel: {gauss_seidel_result['iterations']} iterations, {gauss_seidel_result['cpu_time']:.6f} seconds")
    print(f"Speedup (iterations): {jacobi_result['iterations'] / gauss_seidel_result['iterations']:.2f}x")
    print(f"Speedup (time): {jacobi_result['cpu_time'] / gauss_seidel_result['cpu_time']:.2f}x")
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, len(jacobi_result['residuals']) + 1), jacobi_result['residuals'], 'b-', label='Jacobi')
    plt.semilogy(range(1, len(gauss_seidel_result['residuals']) + 1), gauss_seidel_result['residuals'], 'r-', label='Gauss-Seidel')
    plt.title('Convergence History: 2D Poisson Equation')
    plt.xlabel('Iteration')
    plt.ylabel('Relative Residual (log scale)')
    plt.grid(True)
    plt.legend()
    plt.savefig('poisson_convergence.png')
    print("\nConvergence plot saved as 'poisson_convergence.png'")
    
    # Test Case 2: Nearly singular matrix
    print("\n" + "-"*70)
    print("\nTest Case: Nearly Singular Matrix")
    n = 50
    A = create_nearly_singular_matrix(n)
    
    # Create exact solution and right-hand side
    x_exact = np.ones(n)
    b = A @ x_exact
    
    print(f"Matrix size: {n}×{n}")
    print(f"Condition number: {np.linalg.cond(A):.2e}")
    
    # Check convergence conditions
    conditions = check_convergence_conditions(A)
    print("Convergence conditions:")
    for condition, satisfied in conditions.items():
        print(f"  - {condition}: {'Yes' if satisfied else 'No'}")
    
    # Run Jacobi method
    jacobi_result = jacobi_method(A, b, max_iterations=2000, 
                                 tolerance=1e-8, return_history=True)
    
    # Run Gauss-Seidel method
    gauss_seidel_result = gauss_seidel_method(A, b, max_iterations=2000, 
                                            tolerance=1e-8, return_history=True)
    
    # Calculate error with respect to exact solution
    jacobi_error = np.linalg.norm(jacobi_result['solution'] - x_exact) / np.linalg.norm(x_exact)
    gauss_seidel_error = np.linalg.norm(gauss_seidel_result['solution'] - x_exact) / np.linalg.norm(x_exact)
    
    print(f"Jacobi solution error: {jacobi_error:.6e}")
    print(f"Gauss-Seidel solution error: {gauss_seidel_error:.6e}")
    
    # Print performance metrics
    print("\nPerformance Comparison:")
    print(f"Jacobi: {jacobi_result['iterations']} iterations, {jacobi_result['cpu_time']:.6f} seconds")
    print(f"Gauss-Seidel: {gauss_seidel_result['iterations']} iterations, {gauss_seidel_result['cpu_time']:.6f} seconds")
    print(f"Speedup (iterations): {jacobi_result['iterations'] / gauss_seidel_result['iterations']:.2f}x")
    print(f"Speedup (time): {jacobi_result['cpu_time'] / gauss_seidel_result['cpu_time']:.2f}x")
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, len(jacobi_result['residuals']) + 1), jacobi_result['residuals'], 'b-', label='Jacobi')
    plt.semilogy(range(1, len(gauss_seidel_result['residuals']) + 1), gauss_seidel_result['residuals'], 'r-', label='Gauss-Seidel')
    plt.title('Convergence History: Nearly Singular Matrix')
    plt.xlabel('Iteration')
    plt.ylabel('Relative Residual (log scale)')
    plt.grid(True)
    plt.legend()
    plt.savefig('singular_convergence.png')
    print("\nConvergence plot saved as 'singular_convergence.png'")


if __name__ == "__main__":
    run_advanced_tests()
