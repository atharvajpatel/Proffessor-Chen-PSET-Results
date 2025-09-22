"""
Random Matrix Theory Analysis Module
Eigenspectrum analysis and conditioning diagnostics
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh


def analyze_covariance_spectrum(X, title="Covariance Eigenspectrum"):
    """
    Analyze eigenspectrum of covariance matrix
    
    Args:
        X: Feature matrix (n_samples, n_features)
        title: Title for plots
    
    Returns:
        dict: Dictionary with spectrum analysis results
    """
    # Compute covariance matrix
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered.T)
    
    # Compute eigenvalues
    eigenvals, eigenvecs = eigh(cov_matrix)
    eigenvals = np.sort(eigenvals)[::-1]  # Sort descending
    
    # Effective rank
    eff_rank = (eigenvals.sum()**2) / (eigenvals**2).sum()
    condition_number = eigenvals.max() / eigenvals.min() if eigenvals.min() > 0 else np.inf
    
    # Marchenko-Pastur prediction
    n, p = X.shape
    gamma = p / n
    sigma2 = 1.0  # Assuming standardized features
    
    # MP bounds
    lambda_minus = sigma2 * (1 - np.sqrt(gamma))**2
    lambda_plus = sigma2 * (1 + np.sqrt(gamma))**2
    
    # Plot eigenspectrum - larger and cleaner
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Eigenvalue distribution
    ax1.hist(eigenvals, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(lambda_minus, color='red', linestyle='--', linewidth=2, 
                label=f'MP lower: {lambda_minus:.2f}')
    ax1.axvline(lambda_plus, color='red', linestyle='--', linewidth=2, 
                label=f'MP upper: {lambda_plus:.2f}')
    ax1.set_xlabel('Eigenvalue', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Eigenvalue Distribution', fontsize=14, pad=15)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.tick_params(labelsize=10)
    
    # Plot 2: Eigenvalue decay
    ax2.semilogy(eigenvals, 'o-', markersize=2, linewidth=1.5, color='darkblue')
    ax2.set_xlabel('Eigenvalue Index', fontsize=12)
    ax2.set_ylabel('Eigenvalue (log scale)', fontsize=12)
    ax2.set_title('Eigenvalue Decay', fontsize=14, pad=15)
    ax2.grid(alpha=0.3)
    ax2.tick_params(labelsize=10)
    
    plt.tight_layout()
    
    # Get figure before showing
    current_fig = plt.gcf()
    
    plt.show()
    
    print(f"\n{title} Statistics:")
    print(f"Matrix dimensions: {X.shape}")
    print(f"Effective rank: {eff_rank:.2f} (out of {p})")
    print(f"Condition number: {condition_number:.2e}")
    print(f"Top 5 eigenvalues: {eigenvals[:5].round(3)}")
    print(f"Bottom 5 eigenvalues: {eigenvals[-5:].round(3)}")
    print(f"Marchenko-Pastur bounds: [{lambda_minus:.3f}, {lambda_plus:.3f}]")
    print(f"Eigenvalues outside MP: {np.sum((eigenvals < lambda_minus) | (eigenvals > lambda_plus))}")
    
    return {
        'eigenvals': eigenvals,
        'eigenvecs': eigenvecs,
        'eff_rank': eff_rank,
        'condition_number': condition_number,
        'mp_bounds': (lambda_minus, lambda_plus),
        'n_samples': n,
        'n_features': p,
        'figure': current_fig
    }


def analyze_subgroup_spectra(X_processed, y, demographic_col, demo_data):
    """
    Analyze covariance spectra by demographic subgroups
    
    Args:
        X_processed: Preprocessed feature matrix
        y: Target variable
        demographic_col: Demographic column name
        demo_data: DataFrame with demographic data
    
    Returns:
        dict: Subgroup spectrum results
    """
    print(f"Analyzing spectra by {demographic_col}:")
    print("=" * 50)
    
    subgroup_spectra = {}
    
    # Get unique groups
    groups = demo_data[demographic_col].unique()
    
    # Create larger, cleaner plots
    n_groups = len(groups)
    fig, axes = plt.subplots(2, n_groups, figsize=(6 * n_groups, 12))
    if n_groups == 1:
        axes = axes.reshape(-1, 1)
    
    # Adjust spacing to prevent overlap
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    for i, group in enumerate(groups):
        mask = demo_data[demographic_col] == group
        
        if mask.sum() < 50:  # Skip small groups
            continue
            
        X_group = X_processed[mask]
        
        # Compute eigenspectrum (without plotting)
        X_centered = X_group - np.mean(X_group, axis=0)
        cov_matrix = np.cov(X_centered.T)
        eigenvals, _ = eigh(cov_matrix)
        eigenvals = np.sort(eigenvals)[::-1]
        
        eff_rank = (eigenvals.sum()**2) / (eigenvals**2).sum()
        condition_number = eigenvals.max() / eigenvals.min() if eigenvals.min() > 0 else np.inf
        
        subgroup_spectra[group] = {
            'eigenvals': eigenvals,
            'eff_rank': eff_rank,
            'condition_number': condition_number,
            'n_samples': X_group.shape[0]
        }
        
        # Plot eigenvalue distribution
        axes[0, i].hist(eigenvals, bins=30, density=True, alpha=0.7, color='steelblue')
        # Clean up group name for title
        clean_group = str(group).replace('?', 'Unknown')[:15]  # Truncate long names
        axes[0, i].set_title(f'{clean_group}\n(n={X_group.shape[0]})', fontsize=12, pad=10)
        axes[0, i].set_xlabel('Eigenvalue', fontsize=10)
        axes[0, i].set_ylabel('Density', fontsize=10)
        axes[0, i].grid(alpha=0.3)
        axes[0, i].tick_params(labelsize=9)
        
        # Plot eigenvalue decay
        axes[1, i].semilogy(eigenvals[:50], 'o-', markersize=2, linewidth=1)  # Top 50 eigenvalues
        axes[1, i].set_xlabel('Eigenvalue Index', fontsize=10)
        axes[1, i].set_ylabel('Eigenvalue (log scale)', fontsize=10)
        axes[1, i].set_title(f'Decay: {clean_group}', fontsize=12, pad=10)
        axes[1, i].grid(alpha=0.3)
        axes[1, i].tick_params(labelsize=9)
    
    plt.tight_layout()
    
    # Get figure before showing
    current_fig = plt.gcf()
    
    plt.show()
    
    # Compare effective ranks
    print(f"\nEffective Rank Comparison ({demographic_col}):")
    print("-" * 30)
    for group, spectrum in subgroup_spectra.items():
        print(f"{group:15s}: {spectrum['eff_rank']:6.2f} (condition: {spectrum['condition_number']:.2e})")
    
    # Add figure to results
    subgroup_spectra['figure'] = current_fig
    
    return subgroup_spectra


def generate_rmt_insights(overall_spectrum, subgroup_spectra=None):
    """
    Generate insights from RMT analysis
    
    Args:
        overall_spectrum: Overall spectrum analysis results
        subgroup_spectra: Optional subgroup spectrum results
    
    Returns:
        str: RMT insights summary
    """
    insights = []
    insights.append("RANDOM MATRIX THEORY INSIGHTS:")
    insights.append("=" * 40)
    
    # Overall conditioning
    insights.append(f"• Feature matrix dimensions: {overall_spectrum['n_samples']} x {overall_spectrum['n_features']}")
    insights.append(f"• Effective rank: {overall_spectrum['eff_rank']:.1f} out of {overall_spectrum['n_features']}")
    insights.append(f"• Condition number: {overall_spectrum['condition_number']:.2e}")
    
    # Interpret conditioning
    if overall_spectrum['condition_number'] > 1e12:
        insights.append("• SEVERE ill-conditioning detected - ridge regularization strongly recommended")
    elif overall_spectrum['condition_number'] > 1e6:
        insights.append("• Moderate ill-conditioning - regularization recommended")
    else:
        insights.append("• Well-conditioned feature matrix")
    
    # MP bounds analysis
    lambda_minus, lambda_plus = overall_spectrum['mp_bounds']
    eigenvals = overall_spectrum['eigenvals']
    n_outside_mp = np.sum((eigenvals < lambda_minus) | (eigenvals > lambda_plus))
    
    insights.append(f"• Eigenvalues outside Marchenko-Pastur bounds: {n_outside_mp}")
    if n_outside_mp > 0:
        insights.append("• Spikes detected - likely due to correlated feature groups or outliers")
    
    # Subgroup differences
    if subgroup_spectra:
        insights.append("\nSubgroup Spectrum Differences:")
        all_eff_ranks = []
        for demo_col, demo_spectra in subgroup_spectra.items():
            # Skip the 'figure' key
            if demo_col == 'figure':
                continue
            for group, spec in demo_spectra.items():
                # Skip the 'figure' key within each demographic
                if group == 'figure':
                    continue
                all_eff_ranks.append(spec['eff_rank'])
        
        if len(all_eff_ranks) > 1:
            if max(all_eff_ranks) / min(all_eff_ranks) > 1.5:
                insights.append("• Significant effective rank differences between subgroups")
                insights.append("• This may explain differential model performance across demographics")
            else:
                insights.append("• Similar effective ranks across subgroups")
        else:
            insights.append("• Insufficient subgroup data for comparison")
    
    return "\n".join(insights)
