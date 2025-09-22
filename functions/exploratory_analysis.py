"""
Exploratory Data Analysis Module for Diabetes Readmission Analysis
Part 1.1 - Demographic readmission rate analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def wilson_ci(successes, n, confidence=0.95):
    """
    Wilson confidence interval for proportions
    
    Args:
        successes: Number of successes
        n: Total number of trials
        confidence: Confidence level (default 0.95)
    
    Returns:
        tuple: (proportion, lower_bound, upper_bound)
    """
    if n == 0:
        return 0, 0, 0
    
    p = successes / n
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z**2 / (2 * n)
    adjusted_standard_deviation = np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    
    lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
    
    # Ensure bounds are within [0, 1]
    return p, max(0, lower_bound), min(1, upper_bound)


def plot_readmission_rates(df, outcome, demographic_col, title):
    """
    Plot readmission rates by demographic with confidence intervals
    
    Args:
        df: Dataframe with demographic data
        outcome: Binary outcome array
        demographic_col: Column name for demographic variable
        title: Plot title
    
    Returns:
        rates_df: DataFrame with rates and confidence intervals
    """
    # Calculate rates and CIs for each group
    groups = df[demographic_col].unique()
    rates_data = []
    
    for group in groups:
        mask = df[demographic_col] == group
        n_total = mask.sum()
        n_readmit = outcome[mask].sum()
        
        if n_total > 0:
            rate, ci_lower, ci_upper = wilson_ci(n_readmit, n_total)
            rates_data.append({
                'Group': group,
                'Rate': rate,
                'CI_Lower': ci_lower,
                'CI_Upper': ci_upper,
                'N_Total': n_total,
                'N_Readmit': n_readmit
            })
    
    rates_df = pd.DataFrame(rates_data).sort_values('Rate')
    
    # Create larger, cleaner plot
    fig, ax = plt.subplots(figsize=(14, 8))
    x_pos = np.arange(len(rates_df))
    
    bars = ax.bar(x_pos, rates_df['Rate'], alpha=0.7, color='steelblue', edgecolor='navy', linewidth=1)
    
    # Add error bars (ensure no negative values)
    error_lower = np.maximum(0, rates_df['Rate'] - rates_df['CI_Lower'])
    error_upper = np.maximum(0, rates_df['CI_Upper'] - rates_df['Rate'])
    
    ax.errorbar(x_pos, rates_df['Rate'], 
                yerr=[error_lower, error_upper], 
                fmt='none', color='black', capsize=6, capthick=2, linewidth=2)
    
    # Customize plot with better formatting
    ax.set_xlabel(demographic_col.title(), fontsize=14, fontweight='bold')
    ax.set_ylabel('30-day Readmission Rate', fontsize=14, fontweight='bold')
    ax.set_title(f'{title}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    
    # Clean up group names and rotate labels
    clean_labels = [str(group).replace('?', 'Unknown')[:20] for group in rates_df['Group']]
    ax.set_xticklabels(clean_labels, rotation=45, ha='right', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(labelsize=11)
    
    # Set y-axis limits with better spacing
    max_y = max(rates_df['CI_Upper']) * 1.25
    ax.set_ylim(0, max_y)
    
    # Add sample sizes as text with better positioning
    for i, (idx, row) in enumerate(rates_df.iterrows()):
        y_pos = row['CI_Upper'] + max_y * 0.03
        ax.text(i, y_pos, f'n={row["N_Total"]:,}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Return figure object so it can be saved before showing
    current_fig = plt.gcf()
    
    plt.show()
    
    # Print summary table
    print(f"\n{title} Summary:")
    print(rates_df[['Group', 'Rate', 'CI_Lower', 'CI_Upper', 'N_Total', 'N_Readmit']].round(3))
    
    return rates_df, current_fig


def plot_readmission_rates_fixed(df, y):
    """
    Create comprehensive demographic analysis plots for all demographics
    
    Args:
        df: DataFrame with demographic data  
        y: Binary outcome array
        
    Returns:
        matplotlib.figure.Figure: Figure with subplots
    """
    # Create figure with subplots for all demographics
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    demographics = [
        ('age', 'Age Groups'),
        ('gender', 'Gender'), 
        ('race', 'Race')
    ]
    
    for i, (demo_col, title) in enumerate(demographics):
        if demo_col in df.columns:
            # Calculate rates and CIs for each group
            groups = df[demo_col].unique()
            rates_data = []
            
            for group in groups:
                mask = df[demo_col] == group
                n_total = mask.sum()
                n_readmit = y[mask].sum()
                
                if n_total > 0:
                    rate, ci_lower, ci_upper = wilson_ci(n_readmit, n_total)
                    rates_data.append({
                        'Group': group,
                        'Rate': rate,
                        'CI_Lower': ci_lower,
                        'CI_Upper': ci_upper,
                        'N_Total': n_total,
                        'N_Readmit': n_readmit
                    })
            
            rates_df = pd.DataFrame(rates_data).sort_values('Rate')
            
            # Create subplot
            ax = axes[i]
            x_pos = np.arange(len(rates_df))
            
            bars = ax.bar(x_pos, rates_df['Rate'], alpha=0.7, color='steelblue')
            
            # Add error bars (ensure no negative values)
            error_lower = np.maximum(0, rates_df['Rate'] - rates_df['CI_Lower'])
            error_upper = np.maximum(0, rates_df['CI_Upper'] - rates_df['Rate'])
            
            ax.errorbar(x_pos, rates_df['Rate'], 
                        yerr=[error_lower, error_upper], 
                        fmt='none', color='black', capsize=5)
            
            # Customize subplot
            ax.set_xlabel(demo_col.title())
            ax.set_ylabel('30-Day Readmission Rate')
            ax.set_title(f'{title} Readmission Rates')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(rates_df['Group'], rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, max(rates_df['CI_Upper']) * 1.1)
            
            # Add sample size annotations
            for j, (idx, row) in enumerate(rates_df.iterrows()):
                ax.text(j, row['Rate'] + 0.01, f"n={row['N_Total']}", 
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


def analyze_demographic_patterns(df, y):
    """
    Analyze readmission patterns by age, gender, and race
    
    Args:
        df: Original dataframe with demographic columns
        y: Binary outcome array
    
    Returns:
        dict: Dictionary with analysis results for each demographic
    """
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS - DEMOGRAPHIC PATTERNS")
    print("=" * 60)
    
    demographic_cols = ['age', 'gender', 'race']
    results = {}
    
    for demo_col in demographic_cols:
        if demo_col in df.columns:
            title = f'Readmission Rate by {demo_col.title()} Group'
            print(f"\nAnalyzing {demo_col}...")
            rates_df = plot_readmission_rates(df, y, demo_col, title)
            results[demo_col] = rates_df
        else:
            print(f"Warning: {demo_col} column not found in dataframe")
    
    return results


def generate_clinical_insights(demographic_results):
    """
    Generate clinical insights from demographic analysis
    
    Args:
        demographic_results: Dict with demographic analysis results
    
    Returns:
        str: Clinical insights summary
    """
    insights = []
    insights.append("CLINICAL INSIGHTS FROM DEMOGRAPHIC ANALYSIS:")
    insights.append("=" * 50)
    
    for demo_col, rates_df in demographic_results.items():
        insights.append(f"\n{demo_col.upper()} Patterns:")
        
        # Sort by rate for analysis
        sorted_rates = rates_df.sort_values('Rate')
        lowest_group = sorted_rates.iloc[0]
        highest_group = sorted_rates.iloc[-1]
        
        insights.append(f"  • Lowest risk: {lowest_group['Group']} ({lowest_group['Rate']:.1%} readmission rate, n={lowest_group['N_Total']})")
        insights.append(f"  • Highest risk: {highest_group['Group']} ({highest_group['Rate']:.1%} readmission rate, n={highest_group['N_Total']})")
        
        # Check for significant differences (non-overlapping CIs)
        significant_diffs = []
        for i, row1 in rates_df.iterrows():
            for j, row2 in rates_df.iterrows():
                if i < j and row1['CI_Upper'] < row2['CI_Lower']:
                    significant_diffs.append(f"{row2['Group']} > {row1['Group']}")
        
        if significant_diffs:
            insights.append(f"  • Significant differences: {', '.join(significant_diffs)}")
        else:
            insights.append("  • No statistically significant differences detected (overlapping CIs)")
    
    return "\n".join(insights)
