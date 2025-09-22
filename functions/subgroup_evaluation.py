"""
Subgroup Evaluation Module
Fixed threshold analysis for fairness assessment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score
from .exploratory_analysis import wilson_ci
from tqdm import tqdm


def find_threshold_for_target_accuracy(y_true, y_scores, target_accuracy=0.95):
    """
    Find threshold that achieves target accuracy
    
    Args:
        y_true: True binary labels
        y_scores: Predicted scores
        target_accuracy: Target accuracy level
    
    Returns:
        float: Optimal threshold
    """
    thresholds = np.linspace(0, 1, 1000)
    best_threshold = 0.5
    best_diff = float('inf')
    
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        acc = accuracy_score(y_true, y_pred)
        diff = abs(acc - target_accuracy)
        
        if diff < best_diff:
            best_diff = diff
            best_threshold = thresh
    
    return best_threshold


def evaluate_subgroups(model, X_test, y_test, demographic_cols, threshold):
    """
    Evaluate model performance by demographic subgroups at fixed threshold
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        demographic_cols: List of demographic column names
        threshold: Fixed decision threshold
    
    Returns:
        dict: Subgroup performance results
    """
    y_scores = model.predict_proba(X_test)[:, 1]
    y_pred = (y_scores >= threshold).astype(int)
    
    # Overall performance
    overall_acc = accuracy_score(y_test, y_pred)
    overall_sens = recall_score(y_test, y_pred)  # TPR/Sensitivity
    overall_spec = recall_score(1 - y_test, 1 - y_pred)  # TNR/Specificity
    
    print(f"Overall Performance at threshold {threshold:.4f}:")
    print(f"  Accuracy: {overall_acc:.4f}")
    print(f"  Sensitivity (TPR): {overall_sens:.4f}")
    print(f"  Specificity (TNR): {overall_spec:.4f}")
    print()
    
    subgroup_results = {}
    
    for demo_col in tqdm(demographic_cols, desc="Evaluating Demographics"):
        print(f"Performance by {demo_col.upper()}:")
        print("-" * 50)
        
        subgroup_results[demo_col] = {}
        
        # Get unique groups
        groups = X_test[demo_col].unique()
        
        for group in tqdm(groups, desc=f"{demo_col} groups", leave=False):
            mask = X_test[demo_col] == group
            
            if mask.sum() < 10:  # Skip small groups
                continue
                
            y_group = y_test[mask]
            y_pred_group = y_pred[mask]
            
            # Calculate metrics
            acc = accuracy_score(y_group, y_pred_group)
            
            # Handle cases where there are no positive or negative cases
            if len(np.unique(y_group)) == 1:
                if y_group.iloc[0] == 1:  # All positive
                    sens = recall_score([1], [y_pred_group.iloc[0]]) if len(y_pred_group) > 0 else 0
                    spec = np.nan
                else:  # All negative  
                    sens = np.nan
                    spec = recall_score([0], [1 - y_pred_group.iloc[0]]) if len(y_pred_group) > 0 else 0
            else:
                try:
                    sens = recall_score(y_group, y_pred_group)
                except:
                    sens = np.nan
                try:
                    spec = recall_score(1 - y_group, 1 - y_pred_group)
                except:
                    spec = np.nan
            
            # Confidence intervals (Wilson)
            n = len(y_group)
            acc_ci = wilson_ci(np.sum(y_pred_group == y_group), n)
            
            if not np.isnan(sens) and np.sum(y_group) > 0:
                sens_ci = wilson_ci(np.sum((y_pred_group == 1) & (y_group == 1)), np.sum(y_group))
            else:
                sens_ci = (np.nan, np.nan, np.nan)
                
            if not np.isnan(spec) and np.sum(1 - y_group) > 0:
                spec_ci = wilson_ci(np.sum((y_pred_group == 0) & (y_group == 0)), np.sum(1 - y_group))
            else:
                spec_ci = (np.nan, np.nan, np.nan)
            
            subgroup_results[demo_col][group] = {
                'n': n,
                'accuracy': acc,
                'sensitivity': sens,
                'specificity': spec,
                'acc_ci': acc_ci,
                'sens_ci': sens_ci,
                'spec_ci': spec_ci
            }
            
            print(f"  {group:15s} (n={n:4d}): Acc={acc:.3f} [{acc_ci[1]:.3f}, {acc_ci[2]:.3f}], "
                  f"Sens={sens:.3f}, Spec={spec:.3f}")
        
        print()
    
    return subgroup_results


def plot_subgroup_performance(subgroup_results, metrics=['accuracy', 'sensitivity', 'specificity']):
    """
    Plot subgroup performance with confidence intervals
    
    Args:
        subgroup_results: Dictionary with subgroup results
        metrics: List of metrics to plot
    """
    fig, axes = plt.subplots(len(metrics), len(subgroup_results), 
                            figsize=(5 * len(subgroup_results), 4 * len(metrics)))
    
    if len(metrics) == 1:
        axes = axes.reshape(1, -1)
    if len(subgroup_results) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, metric in enumerate(metrics):
        for j, (demo_col, demo_results) in enumerate(subgroup_results.items()):
            ax = axes[i, j]
            
            groups = []
            values = []
            ci_lowers = []
            ci_uppers = []
            
            for group, group_results in demo_results.items():
                if group_results['n'] >= 10:  # Only include groups with sufficient samples
                    groups.append(group)
                    values.append(group_results[metric])
                    
                    # Get confidence intervals
                    if metric == 'accuracy':
                        ci_lower, ci_upper = group_results['acc_ci'][1], group_results['acc_ci'][2]
                    elif metric == 'sensitivity':
                        ci_lower, ci_upper = group_results['sens_ci'][1], group_results['sens_ci'][2]
                    elif metric == 'specificity':
                        ci_lower, ci_upper = group_results['spec_ci'][1], group_results['spec_ci'][2]
                    else:
                        ci_lower, ci_upper = np.nan, np.nan
                    
                    ci_lowers.append(ci_lower)
                    ci_uppers.append(ci_upper)
            
            if groups:  # Only plot if we have data
                x_pos = np.arange(len(groups))
                
                bars = ax.bar(x_pos, values, alpha=0.7, color='steelblue')
                
                # Add confidence intervals if available
                if not any(np.isnan([ci_lowers, ci_uppers])):
                    errors = [np.maximum(0, np.array(values) - np.array(ci_lowers)),
                             np.maximum(0, np.array(ci_uppers) - np.array(values))]
                    ax.errorbar(x_pos, values, yerr=errors, fmt='none', 
                               color='black', capsize=3)
                
                ax.set_xticks(x_pos)
                ax.set_xticklabels(groups, rotation=45, ha='right')
                ax.set_ylabel(metric.capitalize())
                ax.set_title(f'{metric.capitalize()} by {demo_col.capitalize()}')
                ax.grid(axis='y', alpha=0.3)
                ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()


def analyze_fairness_metrics(subgroup_results):
    """
    Analyze fairness across subgroups
    
    Args:
        subgroup_results: Dictionary with subgroup results
    
    Returns:
        str: Fairness analysis summary
    """
    insights = []
    insights.append("FAIRNESS ANALYSIS SUMMARY:")
    insights.append("=" * 35)
    
    for demo_col, demo_results in subgroup_results.items():
        insights.append(f"\n{demo_col.upper()} Fairness Assessment:")
        
        # Extract metrics for comparison
        groups_data = []
        for group, metrics in demo_results.items():
            if metrics['n'] >= 10:
                groups_data.append({
                    'group': group,
                    'accuracy': metrics['accuracy'],
                    'sensitivity': metrics['sensitivity'],
                    'specificity': metrics['specificity'],
                    'n': metrics['n']
                })
        
        if len(groups_data) < 2:
            insights.append("  • Insufficient groups for fairness comparison")
            continue
        
        # Compare accuracy across groups
        accuracies = [g['accuracy'] for g in groups_data]
        acc_range = max(accuracies) - min(accuracies)
        insights.append(f"  • Accuracy range: {acc_range:.3f} ({min(accuracies):.3f} to {max(accuracies):.3f})")
        
        # Compare sensitivity (TPR)
        sensitivities = [g['sensitivity'] for g in groups_data if not np.isnan(g['sensitivity'])]
        if len(sensitivities) > 1:
            sens_range = max(sensitivities) - min(sensitivities)
            insights.append(f"  • Sensitivity range: {sens_range:.3f} ({min(sensitivities):.3f} to {max(sensitivities):.3f})")
        
        # Compare specificity (TNR)
        specificities = [g['specificity'] for g in groups_data if not np.isnan(g['specificity'])]
        if len(specificities) > 1:
            spec_range = max(specificities) - min(specificities)
            insights.append(f"  • Specificity range: {spec_range:.3f} ({min(specificities):.3f} to {max(specificities):.3f})")
        
        # Identify best and worst performing groups
        best_acc_group = max(groups_data, key=lambda x: x['accuracy'])
        worst_acc_group = min(groups_data, key=lambda x: x['accuracy'])
        
        insights.append(f"  • Highest accuracy: {best_acc_group['group']} ({best_acc_group['accuracy']:.3f})")
        insights.append(f"  • Lowest accuracy: {worst_acc_group['group']} ({worst_acc_group['accuracy']:.3f})")
        
        # Fairness assessment
        if acc_range < 0.05:
            insights.append("  • Assessment: GOOD fairness (accuracy difference < 5%)")
        elif acc_range < 0.10:
            insights.append("  • Assessment: MODERATE fairness concerns (accuracy difference 5-10%)")
        else:
            insights.append("  • Assessment: SIGNIFICANT fairness concerns (accuracy difference > 10%)")
    
    # Overall fairness summary
    insights.append("\nOverall Fairness Recommendations:")
    insights.append("• Consider subgroup-specific thresholds if large performance gaps exist")
    insights.append("• Investigate data representation and feature engineering for underperforming groups")
    insights.append("• Monitor these metrics in production deployment")
    
    return "\n".join(insights)


def generate_threshold_recommendations(subgroup_results, overall_threshold):
    """
    Generate recommendations for threshold optimization
    
    Args:
        subgroup_results: Dictionary with subgroup results
        overall_threshold: Current fixed threshold
    
    Returns:
        str: Threshold recommendations
    """
    insights = []
    insights.append("THRESHOLD OPTIMIZATION RECOMMENDATIONS:")
    insights.append("=" * 45)
    
    insights.append(f"Current fixed threshold: {overall_threshold:.4f}")
    insights.append("\nSubgroup-specific considerations:")
    
    for demo_col, demo_results in subgroup_results.items():
        insights.append(f"\n{demo_col.upper()}:")
        
        for group, metrics in demo_results.items():
            if metrics['n'] >= 10:
                sens = metrics['sensitivity']
                spec = metrics['specificity']
                
                if not np.isnan(sens) and not np.isnan(spec):
                    if sens < 0.7:  # Low sensitivity
                        insights.append(f"  • {group}: Consider LOWER threshold (current sens={sens:.3f})")
                    elif spec < 0.8:  # Low specificity
                        insights.append(f"  • {group}: Consider HIGHER threshold (current spec={spec:.3f})")
                    else:
                        insights.append(f"  • {group}: Current threshold appears appropriate")
    
    insights.append("\nImplementation options:")
    insights.append("1. Single global threshold (current approach)")
    insights.append("2. Demographic-aware thresholds (requires careful validation)")
    insights.append("3. Post-hoc calibration by subgroup")
    insights.append("4. Fairness-constrained model retraining")
    
    return "\n".join(insights)
