# Summary: Jacobi vs. Gauss-Seidel Methods

## Overview

This project implements and compares the Jacobi and Gauss-Seidel iterative methods for solving systems of linear equations. Both methods are widely used in numerical analysis, particularly for large sparse systems.

## Key Findings

### Iteration Efficiency

- **Gauss-Seidel consistently requires fewer iterations** than Jacobi across all test cases
- Improvement ranges from 1.4× to 4.3× depending on the problem characteristics
- The iteration advantage diminishes as system size increases

### CPU Time Performance

- **CPU time advantage is problem-dependent**
- Gauss-Seidel is generally faster for small to medium-sized dense systems
- Jacobi can be more efficient for:
  - Large sparse systems (e.g., 2D Poisson equation: Jacobi was 1.05× faster)
  - Systems with special structures (e.g., tridiagonal matrices: Jacobi was 1.28× faster)
  - Nearly singular systems (Jacobi was 1.43× faster)

### Accuracy

- Both methods achieve similar levels of accuracy when they converge
- Relative errors typically on the order of 10^-8 to 10^-9
- No significant accuracy advantage for either method

## Performance Comparison Table

| Test Case | Matrix Size | Jacobi Iterations | Gauss-Seidel Iterations | Iteration Speedup | Jacobi Time (s) | Gauss-Seidel Time (s) | Time Speedup |
|-----------|-------------|-------------------|-------------------------|-------------------|-----------------|------------------------|--------------|
| Well-conditioned | 10×10 | 43 | 10 | 4.30× | 0.001596 | 0.000571 | 2.80× (GS) |
| Moderate-conditioned | 50×50 | 31 | 9 | 3.44× | 0.004410 | 0.002366 | 1.86× (GS) |
| Ill-conditioned | 100×100 | 19 | 7 | 2.71× | 0.005698 | 0.004998 | 1.14× (GS) |
| Tridiagonal | 100×100 | 27 | 17 | 1.59× | 0.008234 | 0.010535 | 1.28× (J) |
| 2D Poisson | 225×225 | 841 | 422 | 1.99× | 0.484123 | 0.510000 | 1.05× (J) |
| Nearly Singular | 50×50 | 17 | 12 | 1.42× | 0.002421 | 0.003443 | 1.43× (J) |

*Note: In the Time Speedup column, (GS) indicates Gauss-Seidel is faster, while (J) indicates Jacobi is faster.*

## Recommendations

1. **For small to medium-sized dense systems**: Use Gauss-Seidel for faster convergence

2. **For large sparse systems** (e.g., from PDEs): Consider Jacobi, especially if parallel computing is available

3. **For tridiagonal or banded systems**: Jacobi may offer better performance despite requiring more iterations

4. **For parallel computing environments**: Prefer Jacobi due to its naturally parallelizable structure

## Conclusion

The choice between Jacobi and Gauss-Seidel methods should be based on the specific characteristics of the problem and the computing environment. While Gauss-Seidel generally converges in fewer iterations, Jacobi can be more efficient in terms of total computation time for certain problem classes, particularly large sparse systems and those amenable to parallel computing.
