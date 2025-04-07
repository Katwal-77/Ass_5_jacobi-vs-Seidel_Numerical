# Theoretical Analysis of Convergence Properties

## Introduction

This document provides a theoretical analysis of the convergence properties of the Jacobi and Gauss-Seidel iterative methods for solving systems of linear equations of the form Ax = b.

## Matrix Splitting Approach

Both Jacobi and Gauss-Seidel methods can be understood through the matrix splitting approach. We decompose the coefficient matrix A into:

A = D - L - U

where:
- D is the diagonal of A
- L is the strictly lower triangular part of A (with negative signs)
- U is the strictly upper triangular part of A (with negative signs)

### Jacobi Method

The Jacobi iteration can be written as:

x^(k+1) = D^(-1)(L + U)x^(k) + D^(-1)b

or equivalently:

x^(k+1) = Bx^(k) + c

where:
- B = D^(-1)(L + U) is the Jacobi iteration matrix
- c = D^(-1)b

### Gauss-Seidel Method

The Gauss-Seidel iteration can be written as:

x^(k+1) = (D - L)^(-1)Ux^(k) + (D - L)^(-1)b

or equivalently:

x^(k+1) = Gx^(k) + f

where:
- G = (D - L)^(-1)U is the Gauss-Seidel iteration matrix
- f = (D - L)^(-1)b

## Convergence Criteria

For both methods, convergence is guaranteed if the spectral radius (maximum absolute eigenvalue) of the iteration matrix is less than 1:

- For Jacobi: ρ(B) < 1
- For Gauss-Seidel: ρ(G) < 1

### Sufficient Conditions for Convergence

Several matrix properties guarantee convergence:

1. **Strict Diagonal Dominance**:
   A matrix A is strictly diagonally dominant if:
   |a_ii| > ∑(j≠i) |a_ij| for all i
   
   If A is strictly diagonally dominant, both Jacobi and Gauss-Seidel methods converge for any initial guess.

2. **Symmetric Positive Definiteness**:
   If A is symmetric (A = A^T) and positive definite (x^T A x > 0 for all x ≠ 0), then:
   - The Gauss-Seidel method always converges for any initial guess
   - The Jacobi method converges if 0 < ω < 2/ρ(B) in its relaxed form

3. **Irreducible Diagonal Dominance**:
   If A is irreducibly diagonally dominant (diagonally dominant and its directed graph is strongly connected), both methods converge.

## Convergence Rate Comparison

### Theoretical Relationship

For symmetric positive definite matrices, it can be proven that:

ρ(G) ≤ [ρ(B)]^2 < ρ(B) < 1

This inequality explains why Gauss-Seidel typically converges faster than Jacobi for such matrices. The spectral radius directly relates to the asymptotic convergence rate.

### Asymptotic Error Reduction

If ρ is the spectral radius of the iteration matrix, then after k iterations, the error is reduced approximately by a factor of ρ^k. This means:

- For Jacobi: error_k ≈ [ρ(B)]^k × error_0
- For Gauss-Seidel: error_k ≈ [ρ(G)]^k × error_0

Since ρ(G) ≤ [ρ(B)]^2 for symmetric positive definite matrices, Gauss-Seidel reduces the error much faster.

## Computational Complexity

### Per-Iteration Cost

- **Jacobi**: O(n²) operations per iteration
- **Gauss-Seidel**: O(n²) operations per iteration

While both methods have the same asymptotic complexity per iteration, Gauss-Seidel has a higher sequential dependency that can impact performance on parallel architectures.

### Total Computational Cost

The total cost depends on both the per-iteration cost and the number of iterations required:

Total Cost = (Per-Iteration Cost) × (Number of Iterations)

Since Gauss-Seidel typically requires fewer iterations but has more sequential dependencies, the overall performance comparison depends on:
1. The specific matrix structure
2. The implementation details
3. The computing architecture (sequential vs. parallel)

## Special Cases

### Tridiagonal Systems

For tridiagonal systems (common in PDEs), both methods have reduced per-iteration complexity of O(n) instead of O(n²). In these cases, the implementation details and memory access patterns can significantly impact performance.

### Large Sparse Systems

For large sparse systems where most elements are zero, the per-iteration cost reduces to O(nnz) where nnz is the number of non-zero elements. In these cases, Jacobi's simpler structure may lead to better cache performance and vectorization opportunities.

## Conclusion

While Gauss-Seidel typically converges in fewer iterations due to its more favorable spectral properties, the actual computational efficiency depends on various factors including matrix structure, problem size, and computing architecture.

The theoretical analysis explains our empirical observations:
1. Gauss-Seidel consistently requires fewer iterations
2. The performance advantage of Gauss-Seidel diminishes for large sparse systems
3. Jacobi can be more efficient for certain matrix structures despite requiring more iterations
