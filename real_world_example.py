"""
Real-World Application of Jacobi and Gauss-Seidel Methods

This script demonstrates how to use the Jacobi and Gauss-Seidel methods
to solve a real-world problem: the steady-state heat distribution in a 2D plate.

The heat equation in steady state can be discretized to form a system of linear
equations that can be solved using iterative methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from iterative_solvers import jacobi_method, gauss_seidel_method
import time


def create_heat_equation_system(n, boundary_conditions):
    """
    Create the linear system for the 2D steady-state heat equation.
    
    The heat equation is discretized using finite differences on an n×n grid.
    The boundary conditions are fixed temperatures at the edges.
    
    Parameters:
    -----------
    n : int
        Number of interior points in each dimension
    boundary_conditions : dict
        Dictionary with keys 'top', 'bottom', 'left', 'right' containing
        the temperature values at each boundary
    
    Returns:
    --------
    tuple
        (A, b) where A is the coefficient matrix and b is the right-hand side vector
    """
    # Total number of interior points
    N = n * n
    
    # Create coefficient matrix A
    A = np.zeros((N, N))
    
    # Create right-hand side vector b
    b = np.zeros(N)
    
    # Fill A and b based on the discretized heat equation
    for i in range(n):
        for j in range(n):
            # Convert 2D index to 1D index
            idx = i * n + j
            
            # Set diagonal element (central point)
            A[idx, idx] = 4
            
            # Connect to neighbors
            # Left neighbor
            if j > 0:
                A[idx, idx - 1] = -1
            else:
                # Left boundary condition
                b[idx] += boundary_conditions['left']
            
            # Right neighbor
            if j < n - 1:
                A[idx, idx + 1] = -1
            else:
                # Right boundary condition
                b[idx] += boundary_conditions['right']
            
            # Top neighbor
            if i > 0:
                A[idx, idx - n] = -1
            else:
                # Top boundary condition
                b[idx] += boundary_conditions['top']
            
            # Bottom neighbor
            if i < n - 1:
                A[idx, idx + n] = -1
            else:
                # Bottom boundary condition
                b[idx] += boundary_conditions['bottom']
    
    return A, b


def reshape_solution(solution, n):
    """
    Reshape the 1D solution vector into a 2D grid.
    
    Parameters:
    -----------
    solution : numpy.ndarray
        1D solution vector
    n : int
        Number of interior points in each dimension
    
    Returns:
    --------
    numpy.ndarray
        2D grid of solution values
    """
    return solution.reshape(n, n)


def add_boundaries(interior_solution, boundary_conditions):
    """
    Add boundary values to the interior solution.
    
    Parameters:
    -----------
    interior_solution : numpy.ndarray
        2D grid of interior solution values
    boundary_conditions : dict
        Dictionary with keys 'top', 'bottom', 'left', 'right' containing
        the temperature values at each boundary
    
    Returns:
    --------
    numpy.ndarray
        2D grid including boundary values
    """
    n = interior_solution.shape[0]
    
    # Create a larger grid that includes boundaries
    full_solution = np.zeros((n + 2, n + 2))
    
    # Set interior points
    full_solution[1:-1, 1:-1] = interior_solution
    
    # Set boundary values
    full_solution[0, 1:-1] = boundary_conditions['top']
    full_solution[-1, 1:-1] = boundary_conditions['bottom']
    full_solution[1:-1, 0] = boundary_conditions['left']
    full_solution[1:-1, -1] = boundary_conditions['right']
    
    # Set corners (average of adjacent boundaries)
    full_solution[0, 0] = (boundary_conditions['top'] + boundary_conditions['left']) / 2
    full_solution[0, -1] = (boundary_conditions['top'] + boundary_conditions['right']) / 2
    full_solution[-1, 0] = (boundary_conditions['bottom'] + boundary_conditions['left']) / 2
    full_solution[-1, -1] = (boundary_conditions['bottom'] + boundary_conditions['right']) / 2
    
    return full_solution


def plot_heat_distribution(solution, title):
    """
    Plot the heat distribution as a 3D surface.
    
    Parameters:
    -----------
    solution : numpy.ndarray
        2D grid of temperature values
    title : str
        Title for the plot
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create grid coordinates
    n = solution.shape[0]
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    
    # Plot surface
    surf = ax.plot_surface(X, Y, solution, cmap='viridis', edgecolor='none')
    
    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Temperature')
    ax.set_title(title)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    return fig


def main():
    """
    Main function to solve the heat equation and visualize the results.
    """
    print("Solving 2D Steady-State Heat Equation")
    print("=" * 50)
    
    # Problem parameters
    n = 50  # Number of interior points in each dimension
    
    # Boundary conditions (temperatures at the edges)
    boundary_conditions = {
        'top': 100.0,    # Hot top edge
        'bottom': 0.0,   # Cold bottom edge
        'left': 25.0,    # Warm left edge
        'right': 50.0    # Warmer right edge
    }
    
    print(f"Grid size: {n}×{n} interior points")
    print("Boundary conditions:")
    for edge, temp in boundary_conditions.items():
        print(f"  {edge}: {temp}°C")
    
    # Create the linear system
    print("\nCreating linear system...")
    A, b = create_heat_equation_system(n, boundary_conditions)
    print(f"System size: {A.shape[0]}×{A.shape[1]}")
    
    # Solve using Jacobi method
    print("\nSolving using Jacobi method...")
    start_time = time.time()
    jacobi_result = jacobi_method(A, b, max_iterations=5000, tolerance=1e-6, return_history=True)
    jacobi_time = time.time() - start_time
    
    # Solve using Gauss-Seidel method
    print("Solving using Gauss-Seidel method...")
    start_time = time.time()
    gauss_seidel_result = gauss_seidel_method(A, b, max_iterations=5000, tolerance=1e-6, return_history=True)
    gauss_seidel_time = time.time() - start_time
    
    # Print performance comparison
    print("\nPerformance Comparison:")
    print(f"Jacobi: {jacobi_result['iterations']} iterations, {jacobi_time:.4f} seconds")
    print(f"Gauss-Seidel: {gauss_seidel_result['iterations']} iterations, {gauss_seidel_time:.4f} seconds")
    print(f"Iteration speedup: {jacobi_result['iterations'] / gauss_seidel_result['iterations']:.2f}×")
    print(f"Time speedup: {jacobi_time / gauss_seidel_time:.2f}×")
    
    # Reshape solutions to 2D grids
    jacobi_interior = reshape_solution(jacobi_result['solution'], n)
    gauss_seidel_interior = reshape_solution(gauss_seidel_result['solution'], n)
    
    # Add boundaries
    jacobi_full = add_boundaries(jacobi_interior, boundary_conditions)
    gauss_seidel_full = add_boundaries(gauss_seidel_interior, boundary_conditions)
    
    # Calculate difference between solutions
    max_diff = np.max(np.abs(jacobi_full - gauss_seidel_full))
    print(f"\nMaximum difference between solutions: {max_diff:.6e}")
    
    # Plot convergence history
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, len(jacobi_result['residuals']) + 1), jacobi_result['residuals'], 'b-', label='Jacobi')
    plt.semilogy(range(1, len(gauss_seidel_result['residuals']) + 1), gauss_seidel_result['residuals'], 'r-', label='Gauss-Seidel')
    plt.title('Convergence History: 2D Heat Equation')
    plt.xlabel('Iteration')
    plt.ylabel('Relative Residual (log scale)')
    plt.grid(True)
    plt.legend()
    plt.savefig('heat_equation_convergence.png')
    print("\nConvergence plot saved as 'heat_equation_convergence.png'")
    
    # Plot heat distributions
    jacobi_fig = plot_heat_distribution(jacobi_full, 'Heat Distribution (Jacobi Method)')
    jacobi_fig.savefig('heat_distribution_jacobi.png')
    print("Heat distribution plot (Jacobi) saved as 'heat_distribution_jacobi.png'")
    
    gauss_seidel_fig = plot_heat_distribution(gauss_seidel_full, 'Heat Distribution (Gauss-Seidel Method)')
    gauss_seidel_fig.savefig('heat_distribution_gauss_seidel.png')
    print("Heat distribution plot (Gauss-Seidel) saved as 'heat_distribution_gauss_seidel.png'")
    
    # Plot difference between solutions
    diff_fig = plt.figure(figsize=(10, 8))
    ax = diff_fig.add_subplot(111, projection='3d')
    
    # Create grid coordinates
    n_full = jacobi_full.shape[0]
    x = np.linspace(0, 1, n_full)
    y = np.linspace(0, 1, n_full)
    X, Y = np.meshgrid(x, y)
    
    # Plot difference surface
    diff = np.abs(jacobi_full - gauss_seidel_full)
    surf = ax.plot_surface(X, Y, diff, cmap='hot', edgecolor='none')
    
    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Absolute Difference')
    ax.set_title('Absolute Difference Between Jacobi and Gauss-Seidel Solutions')
    
    # Add colorbar
    diff_fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    diff_fig.savefig('heat_distribution_difference.png')
    print("Difference plot saved as 'heat_distribution_difference.png'")
    
    print("\nAnalysis completed!")


if __name__ == "__main__":
    main()
