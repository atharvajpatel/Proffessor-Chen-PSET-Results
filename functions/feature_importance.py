"""
Feature Importance Analysis Module
Linear model coefficients with bootstrap confidence intervals
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from matplotlib.patches import Patch
from tqdm import tqdm


def analyze_linear_coefficients(model, preprocessor, X_train, y_train, feature_names, n_bootstrap=100):
    """
    Analyze linear model coefficients with bootstrap confidence intervals
    
    Args:
        model: Trained logistic regression model (not pipeline)
        preprocessor: Fitted preprocessor used for training
        X_train, y_train: Training data (raw, will be preprocessed internally)
        feature_names: List of feature names after preprocessing
        n_bootstrap: Number of bootstrap samples
    
    Returns:
        dict: Dictionary with coefficient analysis results
    """
    # Get coefficients from trained model (direct model, not pipeline)
    coefs = model.coef_[0]
    
    # Bootstrap for confidence intervals
    bootstrap_coefs = []
    
    print("Computing bootstrap confidence intervals for coefficients...")
    for i in tqdm(range(n_bootstrap), desc="Bootstrap Coefficients", leave=False):
        # Bootstrap sample (use smaller sample size for speed)
        sample_size = min(10000, len(X_train))  # Cap at 10K samples
        indices = np.random.choice(len(X_train), sample_size, replace=True)
        X_boot = X_train.iloc[indices]
        y_boot = y_train.iloc[indices] if hasattr(y_train, 'iloc') else y_train[indices]
        
        # Preprocess bootstrap sample
        X_boot_processed = preprocessor.transform(X_boot)
        
        # Fit model on bootstrap sample (preprocessed) - reduced iterations for speed
        model_boot = LogisticRegression(
            max_iter=100,  # Reduced from 1000 to 100
            class_weight='balanced',
            C=1.0,
            random_state=42,
            solver='liblinear'  # Faster solver for smaller datasets
        )
        
        try:
            model_boot.fit(X_boot_processed, y_boot)
            bootstrap_coefs.append(model_boot.coef_[0])
        except:
            continue
    
    bootstrap_coefs = np.array(bootstrap_coefs)
    
    # Calculate confidence intervals
    coef_cis = np.percentile(bootstrap_coefs, [2.5, 97.5], axis=0)
    
    # Debug: Check array lengths
    print(f"Number of coefficients: {len(coefs)}")
    print(f"Number of feature names: {len(feature_names)}")
    print(f"Shape of coef_cis: {coef_cis.shape}")
    
    # Ensure all arrays have the same length
    n_features = len(coefs)
    if len(feature_names) != n_features:
        print(f"Warning: feature_names length ({len(feature_names)}) doesn't match coefficients ({n_features})")
        # Use generic feature names if mismatch
        feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # Create DataFrame with results
    coef_df = pd.DataFrame({
        'feature': feature_names[:n_features],  # Ensure exact length match
        'coefficient': coefs,
        'ci_lower': coef_cis[0],
        'ci_upper': coef_cis[1],
        'abs_coef': np.abs(coefs)
    })
    
    print(f"Completed bootstrap analysis with {len(bootstrap_coefs)} successful samples")
    
    # Return top positive and negative predictors
    coef_df_sorted = coef_df.sort_values('coefficient', ascending=False)
    
    return {
        'coefficients': coef_df,
        'top_positive': coef_df_sorted.head(10),
        'top_negative': coef_df_sorted.tail(10)
    }


def plot_feature_importance(coef_df, n_features=10):
    """
    Plot top positive and negative coefficients
    
    Args:
        coef_df: DataFrame with coefficient results
        n_features: Number of top features to show in each direction
    
    Returns:
        pd.DataFrame: Important features
    """
    # Sort by coefficient value
    coef_sorted = coef_df.sort_values('coefficient')
    
    # Get top negative and positive
    top_negative = coef_sorted.head(n_features)
    top_positive = coef_sorted.tail(n_features)
    
    # Combine and sort by absolute value
    important_features = pd.concat([top_negative, top_positive]).sort_values('abs_coef', ascending=True)
    
    # Create larger, cleaner plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    y_pos = np.arange(len(important_features))
    colors = ['red' if coef < 0 else 'blue' for coef in important_features['coefficient']]
    
    # Plot bars with better styling
    bars = ax.barh(y_pos, important_features['coefficient'], color=colors, alpha=0.7, 
                   edgecolor='black', linewidth=0.5)
    
    # Add confidence intervals (ensure non-negative error bars)
    error_lower = np.maximum(0, important_features['coefficient'] - important_features['ci_lower'])
    error_upper = np.maximum(0, important_features['ci_upper'] - important_features['coefficient'])
    errors = [error_lower, error_upper]
    
    ax.errorbar(important_features['coefficient'], y_pos, xerr=errors, 
                fmt='none', color='black', capsize=4, capthick=1.5, linewidth=1.5)
    
    # Customize plot with better formatting
    ax.set_yticks(y_pos)
    # Clean up feature names for display
    clean_features = [feat[:50] + '...' if len(feat) > 50 else feat for feat in important_features['feature']]
    ax.set_yticklabels(clean_features, fontsize=11)
    ax.set_xlabel('Coefficient Value (Log-Odds)', fontsize=14, fontweight='bold')
    ax.set_title(f'Top {n_features} Risk-Increasing & Risk-Decreasing Features\n(Logistic Regression with 95% Bootstrap CIs)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
    ax.grid(axis='x', alpha=0.3)
    ax.tick_params(labelsize=10)
    
    # Add legend
    legend_elements = [Patch(facecolor='blue', alpha=0.7, label='Increases Readmission Risk'),
                      Patch(facecolor='red', alpha=0.7, label='Decreases Readmission Risk')]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    # Get figure before showing
    current_fig = plt.gcf()
    
    plt.show()
    
    return important_features, current_fig


def generate_clinical_interpretations(coef_df, n_top=10):
    """
    Generate clinical interpretations for top features
    
    Args:
        coef_df: DataFrame with coefficient results
        n_top: Number of top features to interpret
    
    Returns:
        str: Clinical interpretations
    """
    # Clinical interpretation mapping
    clinical_interpretations = {
        'number_inpatient': 'Prior inpatient visits indicate chronic/severe conditions',
        'time_in_hospital': 'Longer stays suggest complex cases requiring more care',
        'number_emergency': 'Emergency visits reflect acute episodes and instability',
        'num_medications': 'Polypharmacy indicates multiple comorbidities',
        'number_diagnoses': 'Multiple diagnoses suggest complex medical conditions',
        'age': 'Advanced age associated with higher readmission risk',
        'discharge_disposition_id': 'Discharge destination affects follow-up care quality',
        'admission_type_id': 'Emergency admissions have higher readmission risk',
        'num_lab_procedures': 'More lab work may indicate diagnostic uncertainty',
        'num_procedures': 'Multiple procedures suggest complex interventions',
        'diabetesMed': 'Diabetes medication management affects outcomes',
        'change': 'Medication changes may indicate treatment optimization',
        'A1Cresult': 'HbA1c levels reflect long-term glucose control',
        'max_glu_serum': 'Peak glucose levels indicate acute glycemic control',
        'insulin': 'Insulin use indicates more severe diabetes'
    }
    
    insights = []
    insights.append("CLINICAL INTERPRETATION OF KEY FEATURES:")
    insights.append("=" * 50)
    
    # Top risk-increasing features
    insights.append("\nFeatures INCREASING readmission risk:")
    top_positive = coef_df.nlargest(n_top, 'coefficient')
    for _, row in top_positive.iterrows():
        feature_base = row['feature'].split('_')[0] if '_' in row['feature'] else row['feature']
        interpretation = clinical_interpretations.get(feature_base, 'Clinical significance requires domain expertise')
        insights.append(f"  • {row['feature'][:40]:40s}: {row['coefficient']:+.3f} - {interpretation}")
    
    # Top protective features
    insights.append("\nFeatures DECREASING readmission risk:")
    top_negative = coef_df.nsmallest(n_top, 'coefficient')
    for _, row in top_negative.iterrows():
        feature_base = row['feature'].split('_')[0] if '_' in row['feature'] else row['feature']
        interpretation = clinical_interpretations.get(feature_base, 'Clinical significance requires domain expertise')
        insights.append(f"  • {row['feature'][:40]:40s}: {row['coefficient']:+.3f} - {interpretation}")
    
    # Statistical significance notes
    insights.append("\nStatistical Notes:")
    significant_features = coef_df[
        (coef_df['ci_lower'] > 0) | (coef_df['ci_upper'] < 0)
    ]
    insights.append(f"• {len(significant_features)} features have confidence intervals not crossing zero")
    insights.append(f"• {len(coef_df) - len(significant_features)} features have uncertain effect direction")
    
    return "\n".join(insights)


def analyze_coefficient_stability(coef_df):
    """
    Analyze coefficient stability and variance
    
    Args:
        coef_df: DataFrame with coefficient results
    
    Returns:
        str: Stability analysis
    """
    insights = []
    insights.append("COEFFICIENT STABILITY ANALYSIS:")
    insights.append("=" * 35)
    
    # Calculate coefficient variance (CI width)
    coef_df['ci_width'] = coef_df['ci_upper'] - coef_df['ci_lower']
    
    # Most stable coefficients (narrow CIs)
    most_stable = coef_df.nsmallest(5, 'ci_width')
    insights.append("\nMost stable coefficients (narrow CIs):")
    for _, row in most_stable.iterrows():
        insights.append(f"  • {row['feature'][:30]:30s}: {row['coefficient']:+.3f} ± {row['ci_width']/2:.3f}")
    
    # Least stable coefficients (wide CIs)
    least_stable = coef_df.nlargest(5, 'ci_width')
    insights.append("\nLeast stable coefficients (wide CIs):")
    for _, row in least_stable.iterrows():
        insights.append(f"  • {row['feature'][:30]:30s}: {row['coefficient']:+.3f} ± {row['ci_width']/2:.3f}")
    
    # Overall stability metrics
    mean_ci_width = coef_df['ci_width'].mean()
    insights.append(f"\nOverall coefficient stability:")
    insights.append(f"• Mean CI width: {mean_ci_width:.3f}")
    insights.append(f"• {len(coef_df[coef_df['ci_width'] > 0.1])} features have CI width > 0.1 (potentially unstable)")
    
    return "\n".join(insights)
