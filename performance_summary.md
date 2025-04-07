# Performance Summary: Jacobi vs. Gauss-Seidel Methods

## Overview

This document provides a concise summary of the performance comparison between the Jacobi and Gauss-Seidel iterative methods for solving systems of linear equations, focusing specifically on CPU time and accuracy.

## CPU Time Performance

### Key Findings

- **Gauss-Seidel is generally faster** in 4 out of 5 test cases
- **Performance gap widens with system size**:
  - Small system (10×10): Gauss-Seidel is 3.24× faster
  - Medium system (50×50): Gauss-Seidel is 19.98× faster
  - Large system (200×200): Gauss-Seidel is 66.81× faster
- **Exception**: For tridiagonal systems, Jacobi is 1.13× faster

### Iteration Counts

| Test Case | Jacobi Iterations | Gauss-Seidel Iterations | Reduction |
|-----------|-------------------|-------------------------|-----------|
| Small system (10×10) | 90 | 12 | 7.5× |
| Medium system (50×50) | 488 | 13 | 37.5× |
| Large system (200×200) | 1819 | 13 | 139.9× |
| Tridiagonal system (100×100) | 27 | 17 | 1.6× |
| Ill-conditioned system (50×50) | 24 | 9 | 2.7× |

The dramatic reduction in iteration count is the primary driver of Gauss-Seidel's CPU time advantage.

## Accuracy Performance

### Key Findings

- **Both methods achieve high accuracy** with errors on the order of 10^-9
- **Gauss-Seidel is more accurate** in 3 out of 5 test cases
- **Accuracy advantage for larger systems**:
  - Medium system: Gauss-Seidel is 2.43× more accurate
  - Large system: Gauss-Seidel is 2.97× more accurate
- **Jacobi is slightly more accurate** for small and tridiagonal systems

### Error Comparison

| Test Case | Jacobi Error | Gauss-Seidel Error | More Accurate | Factor |
|-----------|--------------|--------------------|--------------------|--------|
| Small system (10×10) | 9.00e-09 | 1.40e-08 | Jacobi | 1.56× |
| Medium system (50×50) | 9.76e-09 | 4.02e-09 | Gauss-Seidel | 2.43× |
| Large system (200×200) | 9.96e-09 | 3.35e-09 | Gauss-Seidel | 2.97× |
| Tridiagonal system (100×100) | 7.04e-09 | 7.19e-09 | Jacobi | 1.02× |
| Ill-conditioned system (50×50) | 8.91e-09 | 3.12e-09 | Gauss-Seidel | 2.85× |

## Combined Performance Assessment

| Test Case | Faster Method | More Accurate Method | Overall Better Method |
|-----------|---------------|----------------------|----------------------|
| Small system (10×10) | Gauss-Seidel | Jacobi | Gauss-Seidel |
| Medium system (50×50) | Gauss-Seidel | Gauss-Seidel | Gauss-Seidel |
| Large system (200×200) | Gauss-Seidel | Gauss-Seidel | Gauss-Seidel |
| Tridiagonal system (100×100) | Jacobi | Jacobi | Jacobi |
| Ill-conditioned system (50×50) | Gauss-Seidel | Gauss-Seidel | Gauss-Seidel |

## Recommendations

1. **For general-purpose applications**: Use Gauss-Seidel as the default choice
   - Significantly faster for medium to large systems
   - Generally more accurate, especially for larger systems

2. **For tridiagonal or banded systems**: Consider Jacobi
   - Slightly faster and more accurate for these specific structures

3. **For large systems**: Strongly prefer Gauss-Seidel
   - Up to 66.81× faster
   - Up to 2.97× more accurate

4. **For ill-conditioned systems**: Use Gauss-Seidel
   - Better numerical stability
   - 2.85× more accurate while being 1.30× faster

5. **For parallel computing environments**: Consider Jacobi despite slower sequential performance
   - Naturally parallelizable algorithm
   - May outperform Gauss-Seidel in highly parallel environments

## Conclusion

Gauss-Seidel is the superior method for most applications, offering both faster computation and higher accuracy in most test cases. The performance advantage becomes more pronounced as system size increases. However, for specific matrix structures like tridiagonal systems, Jacobi may be the better choice.
