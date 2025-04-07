"""
Visualization of CPU Time and Accuracy for Jacobi and Gauss-Seidel Methods

This script creates detailed visualizations focusing specifically on the CPU time
and accuracy performance of the Jacobi and Gauss-Seidel methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_performance_data():
    """
    Load performance data from CSV file or create sample data if file doesn't exist.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing performance data
    """
    try:
        # Try to load existing data
        df = pd.read_csv('performance_results.csv')
        
        # Extract numeric values from string columns
        df['Jacobi Time'] = df['Jacobi Time (s)'].str.split(' ±').str[0].astype(float)
        df['Gauss-Seidel Time'] = df['Gauss-Seidel Time (s)'].str.split(' ±').str[0].astype(float)
        df['Jacobi Error'] = df['Jacobi Error'].str.split('e').str[0].astype(float) * 10**-9
        df['Gauss-Seidel Error'] = df['Gauss-Seidel Error'].str.split('e').str[0].astype(float) * 10**-9
        
        return df
    except:
        # Create sample data if file doesn't exist
        print("Performance results file not found. Using sample data.")
        
        data = {
            'Test Case': [
                'Small system (10×10)',
                'Medium system (50×50)',
                'Large system (200×200)',
                'Tridiagonal system (100×100)',
                'Ill-conditioned system (50×50)'
            ],
            'Matrix Size': ['10×10', '50×50', '200×200', '100×100', '50×50'],
            'Jacobi Iterations': [90, 488, 1819, 27, 24],
            'Gauss-Seidel Iterations': [12, 13, 13, 17, 9],
            'Jacobi Time': [0.002607, 0.061881, 0.929627, 0.007780, 0.003274],
            'Gauss-Seidel Time': [0.000803, 0.003097, 0.013915, 0.008789, 0.002525],
            'Jacobi Error': [9.000058e-09, 9.760354e-09, 9.964854e-09, 7.037531e-09, 8.906184e-09],
            'Gauss-Seidel Error': [1.402426e-08, 4.021888e-09, 3.353225e-09, 7.188398e-09, 3.122802e-09]
        }
        
        return pd.DataFrame(data)


def create_cpu_time_plot(df):
    """
    Create a detailed visualization of CPU time performance.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing performance data
    """
    plt.figure(figsize=(12, 8))
    
    # Create bar plot
    x = np.arange(len(df['Test Case']))
    width = 0.35
    
    # Use log scale for better visualization
    plt.bar(x - width/2, df['Jacobi Time'], width, label='Jacobi', color='#3498db', alpha=0.8)
    plt.bar(x + width/2, df['Gauss-Seidel Time'], width, label='Gauss-Seidel', color='#e74c3c', alpha=0.8)
    
    # Add labels and title
    plt.xlabel('Test Case', fontsize=12)
    plt.ylabel('CPU Time (seconds)', fontsize=12)
    plt.title('CPU Time Comparison: Jacobi vs. Gauss-Seidel', fontsize=16)
    plt.xticks(x, df['Test Case'], rotation=45, ha='right', fontsize=10)
    plt.yscale('log')  # Use log scale for better visualization
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, which='both')
    
    # Add value labels on bars
    for i, v in enumerate(df['Jacobi Time']):
        plt.text(i - width/2, v * 1.1, f"{v:.4f}s", ha='center', va='bottom', fontsize=9, rotation=90)
    for i, v in enumerate(df['Gauss-Seidel Time']):
        plt.text(i + width/2, v * 1.1, f"{v:.4f}s", ha='center', va='bottom', fontsize=9, rotation=90)
    
    # Add speedup annotations
    for i in range(len(df)):
        speedup = df['Jacobi Time'][i] / df['Gauss-Seidel Time'][i]
        if speedup > 1:
            method = "GS"
            factor = speedup
        else:
            method = "J"
            factor = 1/speedup
        
        y_pos = max(df['Jacobi Time'][i], df['Gauss-Seidel Time'][i]) * 1.5
        plt.text(i, y_pos, f"{factor:.1f}× ({method} faster)", 
                ha='center', va='bottom', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    plt.tight_layout()
    plt.savefig('cpu_time_detailed.png', dpi=300)
    print("CPU time visualization saved as 'cpu_time_detailed.png'")


def create_accuracy_plot(df):
    """
    Create a detailed visualization of accuracy performance.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing performance data
    """
    plt.figure(figsize=(12, 8))
    
    # Create bar plot
    x = np.arange(len(df['Test Case']))
    width = 0.35
    
    plt.bar(x - width/2, df['Jacobi Error'], width, label='Jacobi', color='#3498db', alpha=0.8)
    plt.bar(x + width/2, df['Gauss-Seidel Error'], width, label='Gauss-Seidel', color='#e74c3c', alpha=0.8)
    
    # Add labels and title
    plt.xlabel('Test Case', fontsize=12)
    plt.ylabel('Relative Error', fontsize=12)
    plt.title('Accuracy Comparison: Jacobi vs. Gauss-Seidel', fontsize=16)
    plt.xticks(x, df['Test Case'], rotation=45, ha='right', fontsize=10)
    plt.yscale('log')  # Use log scale for better visualization
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, which='both')
    
    # Add value labels on bars
    for i, v in enumerate(df['Jacobi Error']):
        plt.text(i - width/2, v * 1.1, f"{v:.2e}", ha='center', va='bottom', fontsize=9, rotation=90)
    for i, v in enumerate(df['Gauss-Seidel Error']):
        plt.text(i + width/2, v * 1.1, f"{v:.2e}", ha='center', va='bottom', fontsize=9, rotation=90)
    
    # Add accuracy comparison annotations
    for i in range(len(df)):
        j_error = df['Jacobi Error'][i]
        gs_error = df['Gauss-Seidel Error'][i]
        
        if j_error < gs_error:
            method = "J"
            factor = gs_error / j_error
        else:
            method = "GS"
            factor = j_error / gs_error
        
        y_pos = max(j_error, gs_error) * 1.5
        plt.text(i, y_pos, f"{factor:.1f}× ({method} more accurate)", 
                ha='center', va='bottom', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    plt.tight_layout()
    plt.savefig('accuracy_detailed.png', dpi=300)
    print("Accuracy visualization saved as 'accuracy_detailed.png'")


def create_combined_performance_plot(df):
    """
    Create a visualization that combines CPU time and accuracy.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing performance data
    """
    plt.figure(figsize=(14, 10))
    
    # Create scatter plot
    plt.scatter(df['Jacobi Time'], df['Jacobi Error'], s=200, marker='o', color='#3498db', 
               label='Jacobi', alpha=0.7, edgecolor='black', linewidth=1)
    plt.scatter(df['Gauss-Seidel Time'], df['Gauss-Seidel Error'], s=200, marker='s', color='#e74c3c', 
               label='Gauss-Seidel', alpha=0.7, edgecolor='black', linewidth=1)
    
    # Connect pairs with lines
    for i in range(len(df)):
        plt.plot([df['Jacobi Time'][i], df['Gauss-Seidel Time'][i]], 
                [df['Jacobi Error'][i], df['Gauss-Seidel Error'][i]], 
                'k--', alpha=0.3)
    
    # Add labels for each point
    for i, case in enumerate(df['Test Case']):
        plt.annotate(case, 
                    (df['Jacobi Time'][i], df['Jacobi Error'][i]),
                    xytext=(10, 5), textcoords='offset points',
                    fontsize=9, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        plt.annotate(case, 
                    (df['Gauss-Seidel Time'][i], df['Gauss-Seidel Error'][i]),
                    xytext=(10, 5), textcoords='offset points',
                    fontsize=9, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    # Add labels and title
    plt.xlabel('CPU Time (seconds)', fontsize=14)
    plt.ylabel('Relative Error', fontsize=14)
    plt.title('CPU Time vs. Accuracy: Jacobi vs. Gauss-Seidel', fontsize=18)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=14)
    
    # Add "better" region annotation
    plt.annotate('Better Performance\n(Faster & More Accurate)', 
                xy=(0.01, 0.01), xycoords='axes fraction',
                xytext=(0.05, 0.05), textcoords='axes fraction',
                fontsize=12, ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='#d4efdf', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))
    
    plt.tight_layout()
    plt.savefig('combined_performance.png', dpi=300)
    print("Combined performance visualization saved as 'combined_performance.png'")


def create_iteration_efficiency_plot(df):
    """
    Create a visualization of iteration efficiency (time per iteration).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing performance data
    """
    # Calculate time per iteration
    df['Jacobi Time per Iteration'] = df['Jacobi Time'] / df['Jacobi Iterations']
    df['Gauss-Seidel Time per Iteration'] = df['Gauss-Seidel Time'] / df['Gauss-Seidel Iterations']
    
    plt.figure(figsize=(12, 8))
    
    # Create bar plot
    x = np.arange(len(df['Test Case']))
    width = 0.35
    
    plt.bar(x - width/2, df['Jacobi Time per Iteration'] * 1000, width, label='Jacobi', color='#3498db', alpha=0.8)
    plt.bar(x + width/2, df['Gauss-Seidel Time per Iteration'] * 1000, width, label='Gauss-Seidel', color='#e74c3c', alpha=0.8)
    
    # Add labels and title
    plt.xlabel('Test Case', fontsize=12)
    plt.ylabel('Time per Iteration (milliseconds)', fontsize=12)
    plt.title('Iteration Efficiency: Jacobi vs. Gauss-Seidel', fontsize=16)
    plt.xticks(x, df['Test Case'], rotation=45, ha='right', fontsize=10)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(df['Jacobi Time per Iteration']):
        plt.text(i - width/2, v * 1000 * 1.05, f"{v*1000:.3f}ms", ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(df['Gauss-Seidel Time per Iteration']):
        plt.text(i + width/2, v * 1000 * 1.05, f"{v*1000:.3f}ms", ha='center', va='bottom', fontsize=9)
    
    # Add efficiency comparison annotations
    for i in range(len(df)):
        j_eff = df['Jacobi Time per Iteration'][i]
        gs_eff = df['Gauss-Seidel Time per Iteration'][i]
        
        if j_eff < gs_eff:
            method = "J"
            factor = gs_eff / j_eff
        else:
            method = "GS"
            factor = j_eff / gs_eff
        
        y_pos = max(j_eff, gs_eff) * 1000 * 1.2
        plt.text(i, y_pos, f"{factor:.1f}× ({method} more efficient)", 
                ha='center', va='bottom', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    plt.tight_layout()
    plt.savefig('iteration_efficiency.png', dpi=300)
    print("Iteration efficiency visualization saved as 'iteration_efficiency.png'")


def main():
    """
    Main function to create all visualizations.
    """
    print("Creating detailed visualizations of CPU time and accuracy...")
    
    # Load performance data
    df = load_performance_data()
    
    # Create visualizations
    create_cpu_time_plot(df)
    create_accuracy_plot(df)
    create_combined_performance_plot(df)
    create_iteration_efficiency_plot(df)
    
    print("\nAll visualizations completed!")


if __name__ == "__main__":
    main()
