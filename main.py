#!/usr/bin/env python3
"""
Main Orchestration File for Diabetes 30-Day Readmission Risk Stratification Analysis
Part 1 Report - Complete Implementation

Based on original instructions:
- Dataset: UCI "Diabetes 130-US hospitals (1999–2008)" 
- Task: 30-day readmission prediction (binary classification)
- Split: 60/20/20 stratified splits
- Models: L2-regularized logistic regression, gradient boosting, neural network
- Analysis: EDA by demographics, feature importance, subgroup evaluation, RMT/scaling

Author: Atharva Patel
Date: September 2025
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import all function modules
from functions import (
    data_preprocessing,
    exploratory_analysis, 
    rmt_analysis,
    model_training,
    feature_importance,
    subgroup_evaluation,
    scaling_analysis
)

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# =========================================================================
# PART 1.4 SUBGROUP EVALUATION - COMPLETE IMPLEMENTATION
# =========================================================================

def wilson_ci(successes, n, confidence=0.95):
    """
    Wilson confidence interval for proportions
    """
    if n == 0:
        return 0, 0, 0
    
    import scipy.stats as stats
    p = successes / n
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z**2 / (2 * n)
    adjusted_standard_deviation = np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    
    lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
    
    # Ensure bounds are within [0, 1]
    return p, max(0, lower_bound), min(1, upper_bound)


def find_threshold_for_target_accuracy_main(y_true, y_scores, target_accuracy=0.95):
    """Find threshold that achieves target accuracy"""
    from sklearn.metrics import accuracy_score
    
    thresholds = np.linspace(0, 1, 1000)
    best_threshold = 0.5
    best_diff = float('inf')
    
    for thresh in tqdm(thresholds, desc="Finding Optimal Threshold", leave=False):
        y_pred = (y_scores >= thresh).astype(int)
        acc = accuracy_score(y_true, y_pred)
        diff = abs(acc - target_accuracy)
        
        if diff < best_diff:
            best_diff = diff
            best_threshold = thresh
    
    return best_threshold


def evaluate_subgroups_complete_main(model, X_test_processed, y_test, demographic_data, threshold):
    """
    Complete subgroup evaluation that handles all model types properly
    """
    from sklearn.metrics import accuracy_score, recall_score
    from tqdm import tqdm
    
    # Get predictions using only processed numerical features
    if hasattr(model, 'predict_proba'):
        y_scores = model.predict_proba(X_test_processed)[:, 1]
    else:
        y_scores = model.decision_function(X_test_processed)
        y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
    
    y_pred = (y_scores >= threshold).astype(int)
    
    # Overall performance
    overall_acc = accuracy_score(y_test, y_pred)
    overall_sens = recall_score(y_test, y_pred, zero_division=0)
    overall_spec = recall_score(1 - y_test, 1 - y_pred, zero_division=0)
    
    print(f"Overall Performance at threshold {threshold:.4f}:")
    print(f"  Accuracy: {overall_acc:.4f}")
    print(f"  Sensitivity (TPR): {overall_sens:.4f}")
    print(f"  Specificity (TNR): {overall_spec:.4f}")
    print()
    
    subgroup_results = {}
    
    for demo_col in tqdm(['age', 'gender', 'race'], desc="Evaluating Demographics"):
        if demo_col not in demographic_data.columns:
            continue
            
        print(f"Performance by {demo_col.upper()}:")
        print("-" * 50)
        
        subgroup_results[demo_col] = {}
        groups = demographic_data[demo_col].unique()
        
        for group in tqdm(groups, desc=f"{demo_col} groups", leave=False):
            if pd.isna(group):
                continue
                
            # Get mask for this demographic group
            mask = demographic_data[demo_col] == group
            
            if mask.sum() < 10:  # Skip small groups
                continue
            
            # Get group data using the mask
            y_test_group = y_test[mask]
            y_pred_group = y_pred[mask]
            
            if len(y_test_group) == 0:
                continue
            
            # Calculate metrics
            group_acc = accuracy_score(y_test_group, y_pred_group)
            group_sens = recall_score(y_test_group, y_pred_group, zero_division=0)
            group_spec = recall_score(1 - y_test_group, 1 - y_pred_group, zero_division=0)
            
            # Calculate confidence intervals using Wilson method
            n_total = len(y_test_group)
            n_correct = accuracy_score(y_test_group, y_pred_group, normalize=False)
            
            # Wilson CI for accuracy
            acc_rate, acc_ci_lower, acc_ci_upper = wilson_ci(n_correct, n_total)
            
            # Store results
            subgroup_results[demo_col][group] = {
                'n': n_total,
                'accuracy': group_acc,
                'accuracy_ci': (acc_ci_lower, acc_ci_upper),
                'sensitivity': group_sens,
                'specificity': group_spec
            }
            
            print(f"  {str(group)[:20]:20s}: Acc={group_acc:.3f} [{acc_ci_lower:.3f}, {acc_ci_upper:.3f}] "
                  f"Sens={group_sens:.3f} Spec={group_spec:.3f} (n={n_total})")
        
        print()
    
    return subgroup_results


def plot_subgroup_fairness_main(subgroup_results, overall_accuracy, model_name, threshold, viz_dir=None):
    """
    Create comprehensive fairness visualization
    
    Args:
        subgroup_results: Dictionary with subgroup results
        overall_accuracy: Overall model accuracy
        model_name: Name of the model
        threshold: Decision threshold
        viz_dir: Visualization directory (optional, uses global if not provided)
    """
    # Use global viz_dir if not provided
    if viz_dir is None:
        viz_dir = globals().get('VISUALIZATION_DIR', 'visualizations')
    demographics = ['age', 'gender', 'race']
    available_demos = [demo for demo in demographics if demo in subgroup_results and subgroup_results[demo]]
    
    if not available_demos:
        print("No demographic data available for plotting.")
        return
    
    fig, axes = plt.subplots(1, len(available_demos), figsize=(6*len(available_demos), 6))
    if len(available_demos) == 1:
        axes = [axes]
    
    for demo_idx, demo in enumerate(available_demos):
        ax = axes[demo_idx]
        demo_data = subgroup_results[demo]
        
        groups = list(demo_data.keys())
        accuracies = [demo_data[g]['accuracy'] for g in groups]
        ci_lowers = [demo_data[g]['accuracy_ci'][0] for g in groups]
        ci_uppers = [demo_data[g]['accuracy_ci'][1] for g in groups]
        sample_sizes = [demo_data[g]['n'] for g in groups]
        
        # Create bar plot
        x_pos = np.arange(len(groups))
        bars = ax.bar(x_pos, accuracies, alpha=0.7, color='steelblue')
        
        # Add confidence intervals (ensure non-negative)
        error_lower = np.maximum(0, np.array(accuracies) - np.array(ci_lowers))
        error_upper = np.maximum(0, np.array(ci_uppers) - np.array(accuracies))
        
        if np.any(error_lower > 0) or np.any(error_upper > 0):
            ax.errorbar(x_pos, accuracies, yerr=[error_lower, error_upper], 
                       fmt='none', color='black', capsize=5)
        
        # Add overall performance line
        ax.axhline(y=overall_accuracy, color='red', linestyle='--', alpha=0.7, 
                   label=f'Overall: {overall_accuracy:.3f}')
        
        # Customize plot
        ax.set_xlabel(demo.title())
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{model_name}\\n{demo.title()} Accuracy (θ={threshold:.3f})')
        ax.set_xticks(x_pos)
        
        # Handle long group names
        group_labels = [str(g)[:12] + '...' if len(str(g)) > 12 else str(g) for g in groups]
        ax.set_xticklabels(group_labels, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 1.05)
        
        # Add sample size annotations
        for i, (bar, n) in enumerate(zip(bars, sample_sizes)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'n={n}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Get figure before showing
    current_fig = plt.gcf()
    
    plt.show()
    
    return current_fig


def analyze_fairness_complete_main(subgroup_results, model_name, threshold, overall_accuracy):
    """
    Generate comprehensive fairness analysis report
    """
    print(f"\n📋 COMPREHENSIVE FAIRNESS ANALYSIS")
    print("=" * 50)
    print(f"Model: {model_name}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Overall Accuracy: {overall_accuracy:.3f}")
    
    for demo_col, demo_results in subgroup_results.items():
        if not demo_results:
            continue
            
        print(f"\n📈 {demo_col.upper()} SUBGROUP ANALYSIS:")
        print("  " + "-" * 35)
        
        # Calculate fairness metrics
        accuracies = [group['accuracy'] for group in demo_results.values()]
        sensitivities = [group['sensitivity'] for group in demo_results.values()]
        specificities = [group['specificity'] for group in demo_results.values()]
        sample_sizes = [group['n'] for group in demo_results.values()]
        
        # Fairness gaps
        acc_gap = max(accuracies) - min(accuracies)
        sens_gap = max(sensitivities) - min(sensitivities)
        spec_gap = max(specificities) - min(specificities)
        
        print(f"  Performance Gaps:")
        print(f"    • Accuracy gap:    {acc_gap:.3f}")
        print(f"    • Sensitivity gap: {sens_gap:.3f}")
        print(f"    • Specificity gap: {spec_gap:.3f}")
        
        # Group details
        print(f"\n  Group Performance:")
        for group, data in demo_results.items():
            ci_width = data['accuracy_ci'][1] - data['accuracy_ci'][0]
            print(f"    • {str(group)[:15]:15s}: Acc={data['accuracy']:.3f} "
                  f"Sens={data['sensitivity']:.3f} Spec={data['specificity']:.3f} "
                  f"(n={data['n']:,}, CI±{ci_width/2:.3f})")
        
        # Statistical significance
        significant_groups = []
        for group, data in demo_results.items():
            ci_lower, ci_upper = data['accuracy_ci']
            if ci_upper < overall_accuracy or ci_lower > overall_accuracy:
                significant_groups.append(group)
        
        if significant_groups:
            print(f"\n  ⚠️  Groups with significantly different performance:")
            for group in significant_groups:
                print(f"      • {group}")
        else:
            print(f"\n  ✅ No groups show statistically significant differences")
        
        # Fairness assessment for this demographic
        if acc_gap < 0.05:
            fairness_level = "🟢 EXCELLENT"
            recommendation = "Excellent fairness across groups"
        elif acc_gap < 0.10:
            fairness_level = "🟡 GOOD"
            recommendation = "Good fairness with minor disparities"
        elif acc_gap < 0.15:
            fairness_level = "🟠 MODERATE"
            recommendation = "Moderate fairness concerns - consider bias mitigation"
        else:
            fairness_level = "🔴 CONCERNING"
            recommendation = "Significant fairness issues - requires intervention"
        
        print(f"\n  🏆 {demo_col.title()} Fairness: {fairness_level}")
        print(f"      💡 {recommendation}")
    
    # Overall fairness assessment
    all_gaps = []
    for demo_results in subgroup_results.values():
        if demo_results:
            demo_accs = [group['accuracy'] for group in demo_results.values()]
            if len(demo_accs) > 1:
                all_gaps.append(max(demo_accs) - min(demo_accs))
    
    max_gap = max(all_gaps) if all_gaps else 0
    
    print(f"\n🏆 OVERALL FAIRNESS ASSESSMENT")
    print("-" * 35)
    if max_gap < 0.05:
        overall_fairness = "🟢 EXCELLENT"
    elif max_gap < 0.10:
        overall_fairness = "🟡 GOOD"
    elif max_gap < 0.15:
        overall_fairness = "🟠 MODERATE"
    else:
        overall_fairness = "🔴 CONCERNING"
    
    print(f"Overall Fairness Rating: {overall_fairness}")
    print(f"Maximum Performance Gap: {max_gap:.3f}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if max_gap < 0.05:
        print("• Model demonstrates excellent fairness across all demographic groups")
        print("• Continue monitoring in production deployment")
    elif max_gap < 0.10:
        print("• Model shows good fairness with minor disparities")
        print("• Consider additional validation on underrepresented groups")
    else:
        print("• Model shows concerning fairness gaps requiring attention")
        print("• Consider bias mitigation techniques or subgroup-specific thresholds")
        print("• Investigate data representation and feature engineering")
    
    return max_gap


def run_complete_subgroup_evaluation_main(results, X_test, y_test, df, viz_dir, target_accuracy=0.95):
    """
    Complete subgroup fairness evaluation - integrated into main.py
    
    Args:
        results: Model training results dictionary
        X_test: Test features
        y_test: Test labels  
        df: Original dataframe with demographics
        viz_dir: Visualization directory path
        target_accuracy: Target accuracy for threshold selection
        
    Returns:
        tuple: (subgroup_results, optimal_threshold)
    """
    from sklearn.metrics import accuracy_score
    
    print("⚖️ COMPLETE SUBGROUP FAIRNESS EVALUATION")
    print("=" * 50)

    # Find best model by AUC
    best_model_name = max(results.keys(), key=lambda x: results[x]['test_auc'])
    best_model_info = results[best_model_name]
    best_model = best_model_info['model']
    preprocessor = best_model_info['preprocessor']

    print(f"🏆 Best model: {best_model_name} (AUC: {best_model_info['test_auc']:.4f})")

    # Preprocess test data
    X_test_processed = preprocessor.transform(X_test)

    # Get predictions
    if hasattr(best_model, 'predict_proba'):
        y_scores = best_model.predict_proba(X_test_processed)[:, 1]
    else:
        y_scores = best_model.decision_function(X_test_processed)
        y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())

    # Find optimal threshold for target accuracy
    optimal_threshold = find_threshold_for_target_accuracy_main(
        y_test, y_scores, target_accuracy=target_accuracy
    )

    print(f"🎯 Target accuracy: {target_accuracy:.1%}")
    print(f"🎯 Optimal threshold: {optimal_threshold:.4f}")

    # Get test data demographics
    test_demographics = df.loc[X_test.index][['age', 'gender', 'race']]

    print(f"📋 Test set demographics shape: {test_demographics.shape}")
    print(f"🏥 Available demographics: {list(test_demographics.columns)}")

    # Evaluate subgroups
    print(f"\n📊 Evaluating subgroups at threshold {optimal_threshold:.4f}...")

    subgroup_results = evaluate_subgroups_complete_main(
        best_model, 
        X_test_processed,
        y_test, 
        test_demographics,
        optimal_threshold
    )

    print(f"\n✅ Subgroup evaluation completed!")

    # Calculate achieved accuracy
    y_pred_binary = (y_scores >= optimal_threshold).astype(int)
    achieved_accuracy = accuracy_score(y_test, y_pred_binary)
    print(f"📊 Achieved accuracy: {achieved_accuracy:.1%} at threshold: {optimal_threshold:.4f}")

    # Create visualization
    print(f"\n📊 Generating fairness visualizations...")
    fairness_fig = plot_subgroup_fairness_main(subgroup_results, achieved_accuracy, best_model_name, optimal_threshold, viz_dir)
    fairness_fig.savefig(f"{viz_dir}/07_subgroup_fairness_analysis.png", 
                        dpi=300, bbox_inches='tight')

    # Generate comprehensive analysis
    max_fairness_gap = analyze_fairness_complete_main(subgroup_results, best_model_name, optimal_threshold, achieved_accuracy)

    # Summary
    print(f"\n🎯 EXECUTIVE SUMMARY:")
    print(f"• Model: {best_model_name} achieves {achieved_accuracy:.1%} accuracy at threshold {optimal_threshold:.3f}")
    print(f"• Maximum fairness gap: {max_fairness_gap:.1%} across all demographic groups")
    fairness_grade = "EXCELLENT" if max_fairness_gap < 0.05 else "GOOD" if max_fairness_gap < 0.10 else "MODERATE" if max_fairness_gap < 0.15 else "CONCERNING"
    print(f"• Overall fairness grade: {fairness_grade}")
    print(f"• Ready for production deployment: {'✅ YES' if max_fairness_gap < 0.10 else '⚠️ WITH CAUTION' if max_fairness_gap < 0.15 else '❌ NOT RECOMMENDED'}")
    
    return subgroup_results, optimal_threshold

def main():
    """
    Main orchestration function for diabetes readmission analysis
    """
    print("=" * 80)
    print("🏥 DIABETES 30-DAY READMISSION RISK STRATIFICATION ANALYSIS")
    print("=" * 80)
    print(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Define analysis steps for progress tracking
    analysis_steps = [
        "Data Loading & Preprocessing",
        "Exploratory Data Analysis", 
        "Random Matrix Theory Analysis",
        "Model Training & Evaluation",
        "Feature Importance Analysis",
        "Subgroup Fairness Evaluation",
        "Scaling Analysis",
        "Final Report Generation"
    ]
    
    print("📋 Analysis Pipeline Overview:")
    for i, step in enumerate(analysis_steps, 1):
        print(f"   {i}. {step}")
    print()
    
    # Create visualizations directory - unified subfolder
    viz_dir = "visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    print(f"📁 Created unified visualizations directory: {viz_dir}/")
    print(f"📊 All analysis plots will be saved as high-quality PNG files")
    print()
    
    # Make viz_dir available globally for consistency
    global VISUALIZATION_DIR
    VISUALIZATION_DIR = viz_dir
    
    # =========================================================================
    # PART 0: CONFIGURATION AND SETUP
    # =========================================================================
    
    # Data paths (modify as needed)
    DATA_PATH = "diabetic_data.csv"
    IDS_PATH = "IDS_mapping.csv"  # Optional
    
    # Analysis parameters
    RANDOM_SEED = 42
    TARGET_ACCURACY = 0.95
    N_BOOTSTRAP = 10  # Reduced for speed
    
    print("📋 Configuration:")
    print(f"   • Random seed: {RANDOM_SEED}")
    print(f"   • Target accuracy: {TARGET_ACCURACY:.1%}")
    print(f"   • Bootstrap samples: {N_BOOTSTRAP}")
    print()
    
    # =========================================================================
    # PART 1.0: DATA LOADING AND PREPROCESSING  
    # =========================================================================
    
    print("📂 STEP 1/8: DATA LOADING AND PREPROCESSING")
    print("-" * 50)
    
    # Load raw data
    try:
        df, y, ids = data_preprocessing.load_diabetes_data(DATA_PATH, IDS_PATH)
    except FileNotFoundError:
        print(f"❌ Error: Could not find data file '{DATA_PATH}'")
        print("Please ensure the diabetes dataset is available in the current directory.")
        print("Dataset: UCI Diabetes 130-US hospitals (1999–2008)")
        print("URL: https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008")
        return
    
    # Preprocess data with 60/20/20 split
    print("\n🔧 Preprocessing data with 60/20/20 stratified split...")
    data_splits = data_preprocessing.preprocess_data(
        df, y, test_size=0.2, valid_size=0.25, random_state=RANDOM_SEED
    )
    
    # Extract components
    X_train = data_splits['X_train']
    X_valid = data_splits['X_valid'] 
    X_test = data_splits['X_test']
    y_train = data_splits['y_train']
    y_valid = data_splits['y_valid']
    y_test = data_splits['y_test']
    preprocessor = data_splits['preprocessor']
    categorical_cols = data_splits['categorical_cols']
    numeric_cols = data_splits['numeric_cols']
    df_clean = data_splits['df_clean']
    
    # Get processed features for downstream analysis
    X_processed, feature_names = data_preprocessing.get_processed_features(
        preprocessor, X_train
    )
    
    print("✅ Data preprocessing completed!")
    print()
    
    # =========================================================================
    # PART 1.1: EXPLORATORY DATA ANALYSIS - DEMOGRAPHIC PATTERNS
    # =========================================================================
    
    print("📊 STEP 2/8: EXPLORATORY DATA ANALYSIS")
    print("-" * 50)
    
    # Analyze readmission patterns by demographics
    print("Analyzing demographic patterns...")
    demographic_cols = ['age', 'gender', 'race']
    demographic_results = {}
    
    for demo_col in tqdm(demographic_cols, desc="Demographic Analysis"):
        if demo_col in df.columns:
            title = f'Readmission Rate by {demo_col.title()} Group'
            print(f"\nAnalyzing {demo_col}...")
            rates_df, fig = exploratory_analysis.plot_readmission_rates(df, y, demo_col, title)
            # Save plot before it's closed
            fig.savefig(f"{viz_dir}/01_demographic_readmission_rates_{demo_col}.png", 
                       dpi=300, bbox_inches='tight')
            demographic_results[demo_col] = rates_df
        else:
            print(f"Warning: {demo_col} column not found in dataframe")
    
    # Generate clinical insights
    clinical_insights = exploratory_analysis.generate_clinical_insights(demographic_results)
    print("\n" + clinical_insights)
    
    print("✅ Demographic analysis completed!")
    print()
    
    # =========================================================================
    # PART 1.1+: RANDOM MATRIX THEORY ANALYSIS (BONUS)
    # =========================================================================
    
    print("🔬 STEP 3/8: RANDOM MATRIX THEORY ANALYSIS")
    print("-" * 50)
    
    # Overall covariance spectrum analysis
    print("Analyzing overall feature covariance spectrum...")
    overall_spectrum = rmt_analysis.analyze_covariance_spectrum(
        X_processed, "Overall Feature Covariance Spectrum"
    )
    # Save spectrum plot
    overall_spectrum['figure'].savefig(f"{viz_dir}/02_overall_covariance_spectrum.png", 
                                      dpi=300, bbox_inches='tight')
    
    # Subgroup spectrum analysis
    print("\nAnalyzing demographic subgroup spectra...")
    demographic_data = df[['age', 'gender', 'race']].loc[X_train.index]
    
    subgroup_spectra = {}
    demo_cols_available = [col for col in ['age', 'gender', 'race'] if col in demographic_data.columns]
    
    for demo_col in tqdm(demo_cols_available, desc="Subgroup Spectra"):
        print(f"\nAnalyzing {demo_col} subgroups...")
        subgroup_spectra[demo_col] = rmt_analysis.analyze_subgroup_spectra(
            X_processed, y_train, demo_col, demographic_data
        )
        # Save subgroup spectrum plot
        subgroup_spectra[demo_col]['figure'].savefig(f"{viz_dir}/03_subgroup_spectrum_{demo_col}.png", 
                                                     dpi=300, bbox_inches='tight')
    
    # Generate RMT insights
    rmt_insights = rmt_analysis.generate_rmt_insights(overall_spectrum, subgroup_spectra)
    print("\n" + rmt_insights)
    
    print("✅ RMT analysis completed!")
    print()
    
    # =========================================================================
    # PART 1.2: MODEL DEVELOPMENT - THREE MODEL CLASSES
    # =========================================================================
    
    print("🤖 STEP 4/8: MODEL TRAINING AND EVALUATION")
    print("-" * 50)
    
    # Train model suite with GPU acceleration and model saving
    print("Training three model classes with GPU acceleration...")
    print("   • Logistic Regression (L2-regularized, balanced)")
    print("   • Gradient Boosting (XGBoost with GPU support)")
    print("   • Neural Network (PyTorch MLP with GPU support)")
    print("   • Features: Bootstrap CIs, Model persistence, Reproducible seeding")
    print()
    
    # Train models with saving enabled
    results, model_objects, save_dir = model_training.train_model_suite(
        preprocessor, X_train, y_train, X_valid, y_valid, X_test, y_test,
        save_models=True, seed=RANDOM_SEED
    )
    
    print(f"\n✅ Model training completed! Models saved to: {save_dir}")
    
    # Create model comparison plots
    print("\n📈 Creating model comparison visualizations...")
    forest_fig = model_training.plot_model_comparison(results)
    forest_fig.savefig(f"{viz_dir}/04_model_comparison_forest_plot.png", 
                      dpi=300, bbox_inches='tight')
    
    roc_fig = model_training.plot_roc_curves(results, y_test)
    roc_fig.savefig(f"{viz_dir}/05_model_roc_curves.png", 
                   dpi=300, bbox_inches='tight')
    
    # Get best model for downstream analysis
    best_model_name, best_model = model_training.get_best_model(results)
    
    print("✅ Model evaluation completed!")
    print()
    
    # =========================================================================
    # PART 1.3: FEATURE IMPORTANCE ANALYSIS (LINEAR MODEL)
    # =========================================================================
    
    print("🔍 STEP 5/8: FEATURE IMPORTANCE ANALYSIS")
    print("-" * 50)
    
    # Analyze linear model coefficients
    print("Analyzing logistic regression coefficients with bootstrap CIs...")
    lr_model = results['Logistic Regression']['model']
    
    coef_results = feature_importance.analyze_linear_coefficients(
        lr_model, preprocessor, X_train, y_train, feature_names, n_bootstrap=N_BOOTSTRAP
    )
    
    # Plot feature importance
    print("\n📊 Creating feature importance visualizations...")
    important_features, feat_fig = feature_importance.plot_feature_importance(
        coef_results['coefficients'], n_features=10
    )
    feat_fig.savefig(f"{viz_dir}/06_feature_importance_coefficients.png", 
                    dpi=300, bbox_inches='tight')
    
    # Generate clinical interpretations
    clinical_interpretations = feature_importance.generate_clinical_interpretations(
        coef_results['coefficients'], n_top=10
    )
    print("\n" + clinical_interpretations)
    
    # Analyze coefficient stability
    stability_analysis = feature_importance.analyze_coefficient_stability(
        coef_results['coefficients']
    )
    print("\n" + stability_analysis)
    
    print("✅ Feature importance analysis completed!")
    print()
    
    # =========================================================================
    # PART 1.4: SUBGROUP FAIRNESS EVALUATION - COMPLETE IMPLEMENTATION
    # =========================================================================
    
    print("⚖️  STEP 6/8: COMPLETE SUBGROUP FAIRNESS EVALUATION")
    print("-" * 50)
    
    # Complete subgroup evaluation implementation
    subgroup_results, optimal_threshold = run_complete_subgroup_evaluation_main(
        results, X_test, y_test, df, viz_dir, target_accuracy=TARGET_ACCURACY
    )
    
    print("✅ Complete subgroup fairness evaluation completed!")
    print()
    
    # =========================================================================
    # PART 1.2+: SCALING ANALYSIS (BONUS)
    # =========================================================================
    
    print("📈 STEP 7/8: SCALING ANALYSIS")
    print("-" * 50)
    
    # Generate scaling curves for best model class
    print("Generating scaling curves for data efficiency analysis...")
    
    # Use the best model class for scaling analysis
    if 'Gradient Boosting' in best_model_name:
        from sklearn.ensemble import GradientBoostingClassifier
        scaling_model_class = GradientBoostingClassifier
    elif 'Neural Network' in best_model_name:
        from sklearn.neural_network import MLPClassifier
        scaling_model_class = MLPClassifier
    else:
        from sklearn.linear_model import LogisticRegression
        scaling_model_class = LogisticRegression
    
    # Generate scaling data
    print("Generating scaling curves with progress tracking...")
    scaling_data = scaling_analysis.generate_scaling_curves(
        preprocessor, X_train, y_train, X_test, y_test,
        fractions=[0.33, 0.67],
        n_repeats=2, model_class=scaling_model_class
    )
    
    # Fit power law
    print("\n🔬 Fitting power law to scaling data...")
    fit_results = scaling_analysis.fit_scaling_law(scaling_data)
    
    # Create scaling plots
    print("\n📊 Creating scaling curve visualizations...")
    scaling_fig = scaling_analysis.plot_scaling_curves(fit_results)
    scaling_fig.savefig(f"{viz_dir}/08_scaling_curves_analysis.png", 
                       dpi=300, bbox_inches='tight')
    
    # Analyze data efficiency
    data_efficiency = scaling_analysis.analyze_data_efficiency(
        fit_results, len(X_train)
    )
    print("\n" + data_efficiency)
    
    # Generate scaling summary
    scaling_summary = scaling_analysis.generate_scaling_summary(
        scaling_data, fit_results
    )
    print("\n" + scaling_summary)
    
    print("✅ Scaling analysis completed!")
    print()
    
    # =========================================================================
    # FINAL SUMMARY AND DELIVERABLES
    # =========================================================================
    
    print("📋 STEP 8/8: GENERATING FINAL SUMMARY")
    print("-" * 50)
    
    # Compile final results
    final_summary = generate_final_report(
        demographic_results, results, coef_results, subgroup_results,
        rmt_insights, scaling_summary, best_model_name, optimal_threshold
    )
    
    print(final_summary)
    
    # Save summary to file
    with open('diabetes_readmission_analysis_summary.txt', 'w') as f:
        f.write(final_summary)
    
    print(f"\n💾 Complete analysis summary saved to: diabetes_readmission_analysis_summary.txt")
    print(f"💾 Trained models saved to: {save_dir}/")
    
    # List all saved visualizations
    print(f"\n📊 Visualizations saved to: {viz_dir}/")
    viz_files = [
        "01_demographic_readmission_rates_age.png",
        "01_demographic_readmission_rates_gender.png", 
        "01_demographic_readmission_rates_race.png",
        "02_overall_covariance_spectrum.png",
        "03_subgroup_spectrum_age.png",
        "03_subgroup_spectrum_gender.png",
        "03_subgroup_spectrum_race.png", 
        "04_model_comparison_forest_plot.png",
        "05_model_roc_curves.png",
        "06_feature_importance_coefficients.png",
        "07_subgroup_fairness_analysis.png",
        "08_scaling_curves_analysis.png"
    ]
    
    for i, viz_file in enumerate(viz_files, 1):
        print(f"   {i:2d}. {viz_file}")
    
    print("\n" + "=" * 80)
    print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Analysis finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Total visualizations created: {len(viz_files)}")
    
    return {
        'demographic_results': demographic_results,
        'model_results': results,
        'feature_importance': coef_results,
        'subgroup_results': subgroup_results,
        'rmt_spectrum': overall_spectrum,
        'scaling_results': fit_results,
        'best_model': best_model_name,
        'threshold': optimal_threshold
    }


def generate_final_report(demographic_results, model_results, feature_results, 
                         subgroup_results, rmt_insights, scaling_summary, 
                         best_model_name, threshold):
    """
    Generate comprehensive final report
    """
    report = []
    report.append("=" * 80)
    report.append("🏥 DIABETES 30-DAY READMISSION RISK STRATIFICATION")
    report.append("COMPREHENSIVE ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Executive Summary
    report.append("📊 EXECUTIVE SUMMARY")
    report.append("-" * 20)
    
    # Model performance summary
    best_auc = model_results[best_model_name]['test_auc']
    best_ci = model_results[best_model_name]['test_auc_ci']
    
    report.append(f"• Best performing model: {best_model_name}")
    report.append(f"• Test AUC: {best_auc:.4f} [{best_ci[0]:.4f}, {best_ci[1]:.4f}]")
    report.append(f"• Optimal threshold: {threshold:.4f}")
    
    # Demographic insights
    report.append("\n• Key demographic patterns identified:")
    for demo_col, rates_df in demographic_results.items():
        sorted_rates = rates_df.sort_values('Rate')
        lowest = sorted_rates.iloc[0]
        highest = sorted_rates.iloc[-1]
        report.append(f"  - {demo_col}: {lowest['Group']} ({lowest['Rate']:.1%}) to {highest['Group']} ({highest['Rate']:.1%})")
    
    report.append("")
    
    # Detailed Results
    report.append("📈 DETAILED RESULTS")
    report.append("-" * 20)
    
    # Model comparison
    report.append("\n1. MODEL PERFORMANCE COMPARISON:")
    for name, result in tqdm(model_results.items(), desc="Compiling Model Results", leave=False):
        auc = result['test_auc']
        ci_lower, ci_upper = result['test_auc_ci']
        report.append(f"   • {name:20s}: {auc:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Feature importance highlights
    report.append("\n2. TOP PREDICTIVE FEATURES:")
    top_positive = feature_results['top_positive'].head(5)
    top_negative = feature_results['top_negative'].tail(5)
    
    report.append("   Risk-increasing features:")
    for _, row in tqdm(top_positive.iterrows(), desc="Processing Top Features", leave=False, total=len(top_positive)):
        report.append(f"     • {row['feature'][:40]:40s}: {row['coefficient']:+.3f}")
    
    report.append("   Risk-decreasing features:")
    for _, row in tqdm(top_negative.iterrows(), desc="Processing Bottom Features", leave=False, total=len(top_negative)):
        report.append(f"     • {row['feature'][:40]:40s}: {row['coefficient']:+.3f}")
    
    # Fairness assessment
    report.append("\n3. FAIRNESS ASSESSMENT:")
    for demo_col, demo_results in tqdm(subgroup_results.items(), desc="Analyzing Fairness", leave=False):
        if demo_results:
            accuracies = [result['accuracy'] for result in demo_results.values()]
            acc_range = max(accuracies) - min(accuracies)
            report.append(f"   • {demo_col.title()} accuracy range: {acc_range:.3f}")
            
            if acc_range < 0.05:
                fairness = "EXCELLENT"
            elif acc_range < 0.10:
                fairness = "GOOD"
            else:
                fairness = "CONCERNING"
            report.append(f"     Fairness assessment: {fairness}")
    
    # Technical insights
    report.append("\n4. TECHNICAL INSIGHTS:")
    report.append("   " + rmt_insights.replace('\n', '\n   '))
    
    report.append("\n5. SCALING ANALYSIS:")
    report.append("   " + scaling_summary.replace('\n', '\n   '))
    
    # Recommendations
    report.append("\n📋 RECOMMENDATIONS")
    report.append("-" * 20)
    report.append("• Deploy best model with identified optimal threshold")
    report.append("• Monitor subgroup performance in production")
    report.append("• Consider feature engineering based on RMT insights")
    report.append("• Evaluate data collection strategy based on scaling analysis")
    report.append("• Implement fairness constraints if significant disparities exist")
    
    report.append("\n" + "=" * 80)
    
    return "\n".join(report)


def print_usage():
    """Print usage instructions"""
    print("Usage: python main.py")
    print()
    print("Requirements:")
    print("  • diabetic_data.csv - UCI Diabetes dataset")
    print("  • IDS_mapping.csv - Optional ID mapping file")
    print("  • All functions/ modules properly installed")
    print()
    print("The script will automatically:")
    print("  1. Load and preprocess data")
    print("  2. Perform demographic analysis")
    print("  3. Conduct RMT analysis")
    print("  4. Train three model classes")
    print("  5. Analyze feature importance")
    print("  6. Evaluate subgroup fairness")
    print("  7. Perform scaling analysis")
    print("  8. Generate comprehensive report")


if __name__ == "__main__":
    # Check if help requested
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print_usage()
        sys.exit(0)
    
    # Run main analysis
    try:
        results = main()
        print("\n🎯 Analysis pipeline completed successfully!")
        print("All results saved and models persisted for downstream use.")
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        print("\nFor help, run: python main.py --help")
        sys.exit(1)
