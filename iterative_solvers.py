"""
Iterative Methods for Solving Linear Systems of Equations

This module implements the Jacobi and Gauss-Seidel iterative methods for solving
systems of linear equations of the form Ax = b.

Both methods are implemented with careful attention to numerical stability and
performance monitoring.
"""

import numpy as np
import time
from typing import Tuple, Optional, Dict, Any


def jacobi_method(A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None,
                 max_iterations: int = 1000, tolerance: float = 1e-10,
                 return_history: bool = False) -> Dict[str, Any]:
    """
    Solve a system of linear equations Ax = b using the Jacobi iterative method.
    
    Parameters:
    -----------
    A : np.ndarray
        Coefficient matrix of shape (n, n)
    b : np.ndarray
        Right-hand side vector of shape (n,)
    x0 : np.ndarray, optional
        Initial guess for the solution, defaults to zeros
    max_iterations : int, optional
        Maximum number of iterations to perform
    tolerance : float, optional
        Convergence criterion: stop when the relative residual is below this value
    return_history : bool, optional
        Whether to return the history of residuals and solutions
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - 'solution': The solution vector
        - 'iterations': Number of iterations performed
        - 'residuals': List of residual norms (if return_history=True)
        - 'cpu_time': Time taken for computation
        - 'converged': Boolean indicating whether the method converged
        - 'error': Final relative error
    """
    # Start timing
    start_time = time.time()
    
    # Get the size of the system
    n = A.shape[0]
    
    # Initialize the solution vector if not provided
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()
    
    # Extract diagonal and create inverse diagonal matrix D^-1
    D = np.diag(A)
    
    # Check if the method can be applied (no zeros on the diagonal)
    if np.any(np.abs(D) < 1e-10):
        return {
            'solution': None,
            'iterations': 0,
            'converged': False,
            'cpu_time': time.time() - start_time,
            'error': np.inf,
            'message': "Method cannot be applied: zero(s) on the diagonal"
        }
    
    # Initialize variables for iteration
    x_new = np.zeros_like(x)
    residuals = []
    initial_residual = np.linalg.norm(b - A @ x)
    
    # Precompute the rest of the matrix (A - D)
    R = A - np.diag(D)
    
    # Iteration loop
    for iteration in range(max_iterations):
        # Compute new approximation: x_new = D^-1 * (b - R*x)
        for i in range(n):
            x_new[i] = (b[i] - np.sum(R[i] * x)) / D[i]
        
        # Calculate residual
        residual = np.linalg.norm(b - A @ x_new) / np.linalg.norm(b)
        if return_history:
            residuals.append(residual)
        
        # Check for convergence
        if residual < tolerance:
            x = x_new.copy()
            break
        
        # Update solution for next iteration
        x = x_new.copy()
    
    # Calculate final error and CPU time
    final_residual = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
    cpu_time = time.time() - start_time
    
    # Prepare return dictionary
    result = {
        'solution': x,
        'iterations': iteration + 1,
        'converged': residual < tolerance,
        'cpu_time': cpu_time,
        'error': final_residual
    }
    
    if return_history:
        result['residuals'] = residuals
    
    return result


def gauss_seidel_method(A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None,
                       max_iterations: int = 1000, tolerance: float = 1e-10,
                       return_history: bool = False) -> Dict[str, Any]:
    """
    Solve a system of linear equations Ax = b using the Gauss-Seidel iterative method.
    
    Parameters:
    -----------
    A : np.ndarray
        Coefficient matrix of shape (n, n)
    b : np.ndarray
        Right-hand side vector of shape (n,)
    x0 : np.ndarray, optional
        Initial guess for the solution, defaults to zeros
    max_iterations : int, optional
        Maximum number of iterations to perform
    tolerance : float, optional
        Convergence criterion: stop when the relative residual is below this value
    return_history : bool, optional
        Whether to return the history of residuals and solutions
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - 'solution': The solution vector
        - 'iterations': Number of iterations performed
        - 'residuals': List of residual norms (if return_history=True)
        - 'cpu_time': Time taken for computation
        - 'converged': Boolean indicating whether the method converged
        - 'error': Final relative error
    """
    # Start timing
    start_time = time.time()
    
    # Get the size of the system
    n = A.shape[0]
    
    # Initialize the solution vector if not provided
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()
    
    # Extract diagonal elements
    D = np.diag(A)
    
    # Check if the method can be applied (no zeros on the diagonal)
    if np.any(np.abs(D) < 1e-10):
        return {
            'solution': None,
            'iterations': 0,
            'converged': False,
            'cpu_time': time.time() - start_time,
            'error': np.inf,
            'message': "Method cannot be applied: zero(s) on the diagonal"
        }
    
    # Initialize variables for iteration
    residuals = []
    
    # Iteration loop
    for iteration in range(max_iterations):
        x_old = x.copy()
        
        # Update each component of x
        for i in range(n):
            # Compute sum of already updated components
            sum1 = np.sum(A[i, :i] * x[:i])
            # Compute sum of components not yet updated
            sum2 = np.sum(A[i, i+1:] * x_old[i+1:])
            # Update x[i]
            x[i] = (b[i] - sum1 - sum2) / A[i, i]
        
        # Calculate residual
        residual = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
        if return_history:
            residuals.append(residual)
        
        # Check for convergence
        if residual < tolerance:
            break
    
    # Calculate final error and CPU time
    final_residual = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
    cpu_time = time.time() - start_time
    
    # Prepare return dictionary
    result = {
        'solution': x,
        'iterations': iteration + 1,
        'converged': residual < tolerance,
        'cpu_time': cpu_time,
        'error': final_residual
    }
    
    if return_history:
        result['residuals'] = residuals
    
    return result


def is_diagonally_dominant(A: np.ndarray) -> bool:
    """
    Check if a matrix is diagonally dominant.
    
    A matrix is diagonally dominant if for each row, the absolute value of the
    diagonal element is greater than or equal to the sum of the absolute values
    of the other elements in that row.
    
    Parameters:
    -----------
    A : np.ndarray
        Square matrix to check
    
    Returns:
    --------
    bool
        True if the matrix is diagonally dominant, False otherwise
    """
    n = A.shape[0]
    for i in range(n):
        if abs(A[i, i]) < sum(abs(A[i, j]) for j in range(n) if j != i):
            return False
    return True


def check_convergence_conditions(A: np.ndarray) -> Dict[str, bool]:
    """
    Check various conditions that guarantee convergence of iterative methods.
    
    Parameters:
    -----------
    A : np.ndarray
        Coefficient matrix
    
    Returns:
    --------
    Dict[str, bool]
        Dictionary with convergence conditions and whether they are satisfied
    """
    conditions = {}
    
    # Check if matrix is diagonally dominant
    conditions['diagonally_dominant'] = is_diagonally_dominant(A)
    
    # Check if matrix is symmetric
    conditions['symmetric'] = np.allclose(A, A.T)
    
    # Check if matrix is positive definite (if symmetric)
    if conditions['symmetric']:
        try:
            # A positive definite matrix has all positive eigenvalues
            eigenvalues = np.linalg.eigvals(A)
            conditions['positive_definite'] = np.all(eigenvalues > 0)
        except np.linalg.LinAlgError:
            conditions['positive_definite'] = False
    else:
        conditions['positive_definite'] = False
    
    return conditions
