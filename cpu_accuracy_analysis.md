# CPU Time and Accuracy Analysis: Jacobi vs. Gauss-Seidel Methods

## Introduction

This document presents a focused analysis of the Jacobi and Gauss-Seidel iterative methods for solving systems of linear equations, with specific emphasis on:

1. **CPU Time Performance**: How efficiently each method executes in terms of computational time
2. **Solution Accuracy**: How precise the solutions are relative to the exact solution

## Test Cases

We evaluated both methods on a diverse set of test cases:

1. **Small system (10×10)**: A small dense system with good numerical properties
2. **Medium system (50×50)**: A medium-sized dense system
3. **Large system (200×200)**: A large dense system to test scaling behavior
4. **Tridiagonal system (100×100)**: A sparse system with a specific structure common in PDEs
5. **Ill-conditioned system (50×50)**: A system with challenging numerical properties

All matrices were constructed to be diagonally dominant to ensure convergence.

## CPU Time Performance

### Summary of Results

| Test Case | Jacobi Time (s) | Gauss-Seidel Time (s) | Faster Method | Speedup |
|-----------|-----------------|------------------------|---------------|---------|
| Small system (10×10) | 0.002607 ± 0.000098 | 0.000803 ± 0.000256 | Gauss-Seidel | 3.24× |
| Medium system (50×50) | 0.061881 ± 0.006290 | 0.003097 ± 0.000066 | Gauss-Seidel | 19.98× |
| Large system (200×200) | 0.929627 ± 0.022764 | 0.013915 ± 0.000359 | Gauss-Seidel | 66.81× |
| Tridiagonal system (100×100) | 0.007780 ± 0.001059 | 0.008789 ± 0.000156 | Jacobi | 1.13× |
| Ill-conditioned system (50×50) | 0.003274 ± 0.000307 | 0.002525 ± 0.000102 | Gauss-Seidel | 1.30× |

### Key Observations on CPU Time

1. **Gauss-Seidel is generally faster**: In 4 out of 5 test cases, Gauss-Seidel outperformed Jacobi in terms of CPU time.

2. **Scaling advantage for Gauss-Seidel**: The performance gap widens dramatically as the system size increases:
   - For 10×10: Gauss-Seidel is 3.24× faster
   - For 50×50: Gauss-Seidel is 19.98× faster
   - For 200×200: Gauss-Seidel is 66.81× faster

3. **Exception for special structures**: For the tridiagonal system, Jacobi was actually 1.13× faster than Gauss-Seidel, suggesting that matrix structure plays a significant role in relative performance.

4. **Iteration count impact**: The primary reason for Gauss-Seidel's time advantage is the significantly lower number of iterations required:
   - Small system: 90 (Jacobi) vs. 12 (Gauss-Seidel)
   - Medium system: 488 (Jacobi) vs. 13 (Gauss-Seidel)
   - Large system: 1819 (Jacobi) vs. 13 (Gauss-Seidel)

5. **Per-iteration cost**: While Gauss-Seidel has a higher per-iteration cost due to its sequential nature, this is overwhelmingly offset by the reduced iteration count for dense systems.

## Accuracy Analysis

### Summary of Results

| Test Case | Jacobi Error | Gauss-Seidel Error | More Accurate Method | Accuracy Ratio |
|-----------|--------------|--------------------|-----------------------|----------------|
| Small system (10×10) | 9.000058e-09 | 1.402426e-08 | Jacobi | 1.56× |
| Medium system (50×50) | 9.760354e-09 | 4.021888e-09 | Gauss-Seidel | 2.43× |
| Large system (200×200) | 9.964854e-09 | 3.353225e-09 | Gauss-Seidel | 2.97× |
| Tridiagonal system (100×100) | 7.037531e-09 | 7.188398e-09 | Jacobi | 1.02× |
| Ill-conditioned system (50×50) | 8.906184e-09 | 3.122802e-09 | Gauss-Seidel | 2.85× |

### Key Observations on Accuracy

1. **Both methods achieve high accuracy**: Both Jacobi and Gauss-Seidel methods achieved errors on the order of 10^-9, which is excellent for most practical applications.

2. **Gauss-Seidel often more accurate**: In 3 out of 5 test cases, Gauss-Seidel produced more accurate solutions than Jacobi.

3. **Accuracy advantage for larger systems**: For medium and large systems, Gauss-Seidel's accuracy advantage was more pronounced (2.43× to 2.97× more accurate).

4. **Jacobi's accuracy advantage in specific cases**: For the small system and tridiagonal system, Jacobi produced slightly more accurate results.

5. **Ill-conditioned systems**: For the ill-conditioned system, Gauss-Seidel was significantly more accurate (2.85× better), suggesting better numerical stability.

## Relationship Between CPU Time and Accuracy

Analyzing the relationship between CPU time and accuracy reveals interesting patterns:

1. **Gauss-Seidel's dual advantage**: For medium and large systems, Gauss-Seidel offers both faster computation and higher accuracy—an ideal combination.

2. **Trade-offs in special cases**: For the tridiagonal system, Jacobi offers both slightly better accuracy and faster computation, making it the preferred choice for this specific structure.

3. **Efficiency-accuracy balance**: When considering both metrics together, Gauss-Seidel generally provides the better balance, especially as system size increases.

## Factors Affecting Performance

Several factors influence the relative CPU time and accuracy performance:

1. **System size**: As the system size increases, Gauss-Seidel's advantages in both CPU time and accuracy become more pronounced.

2. **Matrix structure**: Special structures like tridiagonal matrices can favor Jacobi in terms of both CPU time and accuracy.

3. **Condition number**: For ill-conditioned systems, Gauss-Seidel demonstrates better accuracy while maintaining a CPU time advantage.

4. **Convergence rate**: Gauss-Seidel's faster convergence rate (requiring fewer iterations) is the primary driver of its CPU time advantage.

## Practical Recommendations

Based on our analysis of CPU time and accuracy:

1. **For general-purpose applications**: Use Gauss-Seidel as the default choice, especially for medium to large systems, as it offers significant advantages in both CPU time and accuracy.

2. **For tridiagonal or banded systems**: Consider Jacobi, as it may offer both CPU time and accuracy advantages for these specific structures.

3. **For ill-conditioned systems**: Prefer Gauss-Seidel, which demonstrates better numerical stability and accuracy while maintaining a CPU time advantage.

4. **For real-time applications with small systems**: Either method can be appropriate, but Gauss-Seidel will generally be faster.

5. **For parallel computing environments**: Despite its generally slower sequential performance, Jacobi might be preferable due to its natural parallelizability.

## Conclusion

Our comprehensive analysis of CPU time and accuracy reveals that Gauss-Seidel is generally the superior method for most applications, offering both faster computation and higher accuracy in most test cases. However, the specific characteristics of the linear system, particularly its structure and size, can significantly influence the relative performance of these methods.

For practical applications, the choice between Jacobi and Gauss-Seidel should be guided by:
1. The size and structure of the system
2. The relative importance of CPU time vs. accuracy
3. The computing environment (sequential vs. parallel)

This analysis provides a solid foundation for making informed decisions when selecting an iterative method for solving systems of linear equations in various application domains.
