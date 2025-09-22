"""
Scaling Analysis Module
Power law fitting and data efficiency analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from scipy.optimize import curve_fit
from tqdm import tqdm


def power_law(n, A, alpha, B):
    """
    Power law function: 1 - AUC(n) = A * n^(-alpha) + B
    
    Args:
        n: Sample size
        A: Amplitude parameter
        alpha: Scaling exponent
        B: Asymptotic offset
    
    Returns:
        float: 1 - AUC value
    """
    return A * np.power(n, -alpha) + B


def generate_scaling_curves(preprocessor, X_train, y_train, X_test, y_test, 
                           fractions=None, n_repeats=3, model_class=GradientBoostingClassifier):
    """
    Generate scaling curves for a model class
    
    Args:
        preprocessor: Sklearn preprocessor
        X_train, y_train: Training data
        X_test, y_test: Test data
        fractions: List of data fractions to test
        n_repeats: Number of repetitions per fraction
        model_class: Model class to use for scaling analysis
    
    Returns:
        dict: Scaling results
    """
    if fractions is None:
        fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    scaling_results = {}
    
    print("Generating scaling curves...")
    print("=" * 30)
    
    for frac in tqdm(fractions, desc="Scaling fractions"):
        aucs = []
        
        print(f"Testing fraction {frac:.1f}...")
        
        for repeat in tqdm(range(n_repeats), desc=f"Repeats for {frac:.1f}", leave=False):
            # Create subsample
            n_samples = int(frac * len(X_train))
            
            if n_samples < 100:  # Skip very small samples
                continue
            
            # Handle edge case where we want the full dataset
            if n_samples >= len(X_train):
                X_sub, y_sub = X_train, y_train
            else:
                # Stratified subsample
                X_sub, _, y_sub, _ = train_test_split(
                    X_train, y_train, 
                    train_size=n_samples, 
                    stratify=y_train,
                    random_state=42 + repeat
                )
            
            # Train model
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model_class(random_state=42))
            ])
            
            try:
                model.fit(X_sub, y_sub)
                y_pred = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_pred)
                aucs.append(auc)
            except Exception as e:
                print(f"    Warning: Failed for repeat {repeat}: {e}")
                continue
        
        if aucs:
            scaling_results[frac] = {
                'mean_auc': np.mean(aucs),
                'std_auc': np.std(aucs),
                'n_samples': int(frac * len(X_train)),
                'aucs': aucs
            }
            print(f"    AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        else:
            print(f"    No successful runs for fraction {frac}")
    
    return scaling_results


def fit_scaling_law(scaling_data):
    """
    Fit power law to scaling data
    
    Args:
        scaling_data: Dictionary with scaling results
    
    Returns:
        dict: Fitted parameters and diagnostics
    """
    # Extract data for fitting
    fractions = list(scaling_data.keys())
    sample_sizes = [scaling_data[f]['n_samples'] for f in fractions]
    aucs = [scaling_data[f]['mean_auc'] for f in fractions]
    auc_stds = [scaling_data[f]['std_auc'] for f in fractions]
    
    # Convert to arrays
    sample_sizes = np.array(sample_sizes)
    aucs = np.array(aucs)
    auc_stds = np.array(auc_stds)
    
    # Fit power law to 1 - AUC
    one_minus_auc = 1 - aucs
    
    try:
        # Fit power law: 1 - AUC(n) = A * n^(-alpha) + B
        popt, pcov = curve_fit(power_law, sample_sizes, one_minus_auc, 
                              p0=[0.1, 0.5, 0.05], maxfev=5000)
        A_fit, alpha_fit, B_fit = popt
        
        # Calculate fit quality
        y_pred = power_law(sample_sizes, A_fit, alpha_fit, B_fit)
        r_squared = 1 - np.sum((one_minus_auc - y_pred)**2) / np.sum((one_minus_auc - np.mean(one_minus_auc))**2)
        
        print(f"\nPower Law Fit Results:")
        print(f"1 - AUC(n) = {A_fit:.4f} * n^(-{alpha_fit:.3f}) + {B_fit:.4f}")
        print(f"R-squared: {r_squared:.3f}")
        print(f"Alpha (scaling exponent): {alpha_fit:.3f}")
        
        # Interpret scaling regime
        if alpha_fit > 0.3:
            regime = "data-limited (more data helps significantly)"
        elif alpha_fit > 0.1:
            regime = "moderately data-limited"
        else:
            regime = "noise-limited (plateauing performance)"
        
        print(f"Scaling regime: {regime}")
        
        return {
            'A': A_fit,
            'alpha': alpha_fit,
            'B': B_fit,
            'r_squared': r_squared,
            'regime': regime,
            'sample_sizes': sample_sizes,
            'aucs': aucs,
            'auc_stds': auc_stds,
            'fit_successful': True
        }
        
    except Exception as e:
        print(f"Could not fit power law: {e}")
        return {
            'sample_sizes': sample_sizes,
            'aucs': aucs,
            'auc_stds': auc_stds,
            'fit_successful': False
        }


def plot_scaling_curves(fit_results):
    """
    Plot scaling curves with power law fit
    
    Args:
        fit_results: Dictionary with fitting results
    """
    sample_sizes = fit_results['sample_sizes']
    aucs = fit_results['aucs']
    auc_stds = fit_results['auc_stds']
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: AUC vs sample size
    ax1.errorbar(sample_sizes, aucs, yerr=auc_stds, fmt='o-', capsize=5, 
                 label='Observed AUC', color='blue', markersize=8)
    
    if fit_results['fit_successful']:
        # Generate fitted curve
        n_fit = np.linspace(sample_sizes.min(), sample_sizes.max(), 100)
        auc_fit = 1 - power_law(n_fit, fit_results['A'], fit_results['alpha'], fit_results['B'])
        ax1.plot(n_fit, auc_fit, 'r--', linewidth=2, 
                 label=f'Power law fit (α={fit_results["alpha"]:.3f})')
    
    ax1.set_xlabel('Training Sample Size')
    ax1.set_ylabel('Test AUC')
    ax1.set_title('Scaling Curve: AUC vs Training Sample Size')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Log-log plot of 1-AUC vs sample size
    one_minus_auc = 1 - aucs
    ax2.loglog(sample_sizes, one_minus_auc, 'o-', markersize=8, label='Observed 1-AUC')
    
    if fit_results['fit_successful']:
        one_minus_auc_fit = power_law(n_fit, fit_results['A'], fit_results['alpha'], fit_results['B'])
        ax2.loglog(n_fit, one_minus_auc_fit, 'r--', linewidth=2, 
                   label=f'Power law fit (slope=-{fit_results["alpha"]:.3f})')
    
    ax2.set_xlabel('Training Sample Size (log)')
    ax2.set_ylabel('1 - AUC (log)')
    ax2.set_title('Log-Log Scaling: Power Law Behavior')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Get figure before showing
    current_fig = plt.gcf()
    
    plt.show()
    
    return current_fig


def analyze_data_efficiency(fit_results, current_n):
    """
    Analyze data efficiency and make recommendations
    
    Args:
        fit_results: Dictionary with fitting results
        current_n: Current training set size
    
    Returns:
        str: Data efficiency analysis
    """
    insights = []
    insights.append("DATA EFFICIENCY ANALYSIS:")
    insights.append("=" * 30)
    
    if not fit_results['fit_successful']:
        insights.append("• Power law fit unsuccessful - analysis limited")
        return "\n".join(insights)
    
    alpha = fit_results['alpha']
    A = fit_results['A']
    B = fit_results['B']
    current_auc = fit_results['aucs'][-1]  # Assuming last point is full dataset
    
    insights.append(f"• Current dataset size: {current_n:,} samples")
    insights.append(f"• Current AUC: {current_auc:.4f}")
    insights.append(f"• Scaling exponent α: {alpha:.3f}")
    insights.append(f"• Asymptotic AUC: {1 - B:.4f}")
    
    # Regime interpretation
    insights.append(f"• Scaling regime: {fit_results['regime']}")
    
    # Marginal benefit analysis
    if alpha > 0.2:
        # Estimate AUC gain from doubling data
        double_auc = 1 - power_law(current_n * 2, A, alpha, B)
        gain = double_auc - current_auc
        insights.append(f"• Doubling data would improve AUC by ~{gain:.4f}")
        insights.append("• Recommendation: COLLECT MORE DATA - significant gains expected")
    elif alpha > 0.05:
        insights.append("• Moderate data efficiency - some benefit from more data")
        insights.append("• Recommendation: Consider data collection vs other improvements")
    else:
        insights.append("• Low data efficiency - approaching performance plateau")
        insights.append("• Recommendation: Focus on features/models rather than more data")
    
    # Sample size recommendations
    if alpha > 0.3:
        # Estimate sample size for 1% AUC improvement
        target_auc = current_auc + 0.01
        target_error = 1 - target_auc
        required_n = ((target_error - B) / A) ** (1 / (-alpha))
        if required_n > current_n and required_n < current_n * 10:
            insights.append(f"• For +0.01 AUC improvement, need ~{required_n:,.0f} samples")
    
    return "\n".join(insights)


def generate_scaling_summary(scaling_data, fit_results):
    """
    Generate comprehensive scaling analysis summary
    
    Args:
        scaling_data: Raw scaling data
        fit_results: Power law fitting results
    
    Returns:
        str: Complete scaling summary
    """
    insights = []
    insights.append("SCALING ANALYSIS SUMMARY:")
    insights.append("=" * 40)
    
    # Data points summary
    insights.append("\nScaling Curve Data Points:")
    for frac in sorted(scaling_data.keys()):
        data = scaling_data[frac]
        insights.append(f"  {frac:.1f}: n={data['n_samples']:5d}, AUC={data['mean_auc']:.4f} ± {data['std_auc']:.4f}")
    
    # Power law results
    if fit_results['fit_successful']:
        insights.append(f"\nPower Law: 1-AUC(n) = {fit_results['A']:.4f} * n^(-{fit_results['alpha']:.3f}) + {fit_results['B']:.4f}")
        insights.append(f"Fit quality (R²): {fit_results['r_squared']:.3f}")
        insights.append(f"Scaling regime: {fit_results['regime']}")
    
    # Practical implications
    insights.append("\nPractical Implications:")
    if fit_results['fit_successful'] and fit_results['alpha'] > 0.3:
        insights.append("• High data efficiency - prioritize data collection")
        insights.append("• Model performance strongly limited by sample size")
    elif fit_results['fit_successful'] and fit_results['alpha'] > 0.1:
        insights.append("• Moderate data efficiency - balanced approach recommended")
        insights.append("• Consider both data collection and model improvements")
    else:
        insights.append("• Low data efficiency - focus on model/feature improvements")
        insights.append("• Additional data unlikely to provide major gains")
    
    return "\n".join(insights)
