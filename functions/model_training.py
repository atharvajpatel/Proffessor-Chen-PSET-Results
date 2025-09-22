"""
Model Training Module
Three model classes with bootstrap confidence intervals
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import pickle
import os
import random

# GPU-accelerated imports
import xgboost as xgb
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from skorch.dataset import ValidSplit


def set_all_seeds(seed=42):
    """
    Set seeds for reproducible results across all libraries
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # XGBoost (handled via random_state parameter)
    # Sklearn (handled via random_state parameter)
    
    print(f"All seeds set to {seed} for reproducible results")


class MLPNet(nn.Module):
    """
    PyTorch MLP for GPU acceleration
    """
    def __init__(self, input_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Binary classification
        )
    
    def forward(self, x):
        return self.network(x.float())


def bootstrap_auc_ci(y_true, y_scores, n_bootstrap=1000, confidence=0.95):
    """
    Bootstrap confidence intervals for AUC
    
    Args:
        y_true: True binary labels
        y_scores: Predicted scores
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
    
    Returns:
        tuple: (mean_auc, lower_ci, upper_ci)
    """
    # Convert to numpy arrays to avoid pandas indexing issues
    y_true = np.asarray(y_true).flatten()  # Ensure numpy array and flatten
    y_scores = np.asarray(y_scores).flatten()  # Ensure numpy array and flatten
    
    # Verify they're numpy arrays
    assert isinstance(y_true, np.ndarray), f"y_true is {type(y_true)}, not numpy array"
    assert isinstance(y_scores, np.ndarray), f"y_scores is {type(y_scores)}, not numpy array"
    
    n = len(y_true)
    bootstrap_aucs = []
    
    for _ in tqdm(range(n_bootstrap), desc="Bootstrap CI", leave=False, disable=n_bootstrap<100):
        # Bootstrap sample - use numpy indexing
        indices = np.random.choice(n, n, replace=True)
        y_boot = y_true[indices]  # Now guaranteed to be numpy array indexing
        scores_boot = y_scores[indices]  # Now guaranteed to be numpy array indexing
        
        # Skip if all same class
        if len(np.unique(y_boot)) < 2:
            continue
            
        auc_boot = roc_auc_score(y_boot, scores_boot)
        bootstrap_aucs.append(auc_boot)
    
    # Calculate confidence intervals
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_aucs, 100 * alpha / 2)
    upper = np.percentile(bootstrap_aucs, 100 * (1 - alpha / 2))
    
    return np.mean(bootstrap_aucs), lower, upper


def save_trained_models(model_objects, model_results, save_dir="weights"):
    """
    Save trained models and results for downstream use
    
    Args:
        model_objects: Dictionary of trained model objects
        model_results: Dictionary of model performance results
        save_dir: Directory to save models (default: "weights")
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nSaving trained models to {save_dir}/...")
    
    # Define clean model names for file saving
    model_name_map = {
        'Logistic Regression': 'logistic',
        'Gradient Boosting': 'xgboost', 
        'Neural Network': 'mlp'
    }
    
    for name, model in model_objects.items():
        # Use clean filename from mapping
        filename = model_name_map.get(name, name.replace(" ", "_").lower())
        model_path = os.path.join(save_dir, f"{filename}_weights.pkl")
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ Saved {name} weights to {model_path}")
    
    # Save results
    results_path = os.path.join(save_dir, "model_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(model_results, f)
    print(f"✅ Saved model results to {results_path}")
    
    # Save metadata
    first_result = model_results[list(model_results.keys())[0]]
    metadata = {
        'seed': 42,
        'models': list(model_objects.keys()),
        'timestamp': pd.Timestamp.now().isoformat(),
        'gpu_used': torch.cuda.is_available(),
        'training_data_shape': first_result.get('X_train_shape', 'Unknown'),
        'model_filenames': {
            'Logistic Regression': 'logistic_weights.pkl',
            'Gradient Boosting': 'xgboost_weights.pkl', 
            'Neural Network': 'mlp_weights.pkl'
        }
    }
    metadata_path = os.path.join(save_dir, "metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✅ Saved metadata to {metadata_path}")


def load_trained_models(save_dir="weights"):
    """
    Load previously trained models
    
    Args:
        save_dir: Directory containing saved models
        
    Returns:
        tuple: (model_objects, model_results, metadata)
    """
    model_objects = {}
    
    # Load models
    for filename in os.listdir(save_dir):
        if filename.endswith("_model.pkl"):
            model_name = filename.replace("_model.pkl", "").replace("_", " ").title()
            model_path = os.path.join(save_dir, filename)
            
            with open(model_path, 'rb') as f:
                model_objects[model_name] = pickle.load(f)
    
    # Load results
    results_path = os.path.join(save_dir, "model_results.pkl")
    with open(results_path, 'rb') as f:
        model_results = pickle.load(f)
    
    # Load metadata
    metadata_path = os.path.join(save_dir, "metadata.pkl")
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    return model_objects, model_results, metadata


def train_model_suite(preprocessor, X_train, y_train, X_valid, y_valid, X_test, y_test, save_models=True, seed=42):
    """
    Train three model classes and evaluate with confidence intervals
    
    Args:
        preprocessor: Sklearn preprocessor
        X_train, y_train: Training data
        X_valid, y_valid: Validation data
        X_test, y_test: Test data
        save_models: Whether to save trained models
        seed: Random seed for reproducibility
    
    Returns:
        tuple: (model_results, model_objects) if save_models=False
               (model_results, model_objects, save_dir) if save_models=True
    """
    # Set all seeds for reproducibility
    set_all_seeds(seed)
    
    # Check CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
    
    # Preprocess the full training data to get correct feature dimensions
    print("Preprocessing training data to determine feature dimensions...")
    X_train_processed = preprocessor.fit_transform(X_train)
    n_features = X_train_processed.shape[1]
    print(f"Features after preprocessing: {n_features}")
    
    # Define models with GPU acceleration
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, 
            class_weight='balanced',
            C=1.0,
            random_state=42
        ),
        
        'Gradient Boosting': xgb.XGBClassifier(
            tree_method='hist',
            device=device,  # 'cuda' if available, 'cpu' otherwise
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            eval_metric='logloss',
            verbosity=0  # Suppress XGBoost output
        ),
        
        'Neural Network': NeuralNetClassifier(
            MLPNet,
            module__input_dim=n_features,
            module__hidden_dim=128,
            module__dropout=0.2,
            max_epochs=100,
            lr=0.01,
            batch_size=256,
            device=device,
            iterator_train__shuffle=True,
            iterator_valid__shuffle=False,
            train_split=ValidSplit(0.1),  # Use 10% for validation
            verbose=0  # Suppress skorch output
            # Note: PyTorch seeding handled by set_all_seeds() function
        )
    }
    
    # Train models and evaluate
    results = {}
    model_objects = {}
    
    # Preprocess validation and test data
    X_valid_processed = preprocessor.transform(X_valid)
    X_test_processed = preprocessor.transform(X_test)
    
    print("Training models and computing AUC with confidence intervals...")
    print("=" * 60)
    
    for name, model in tqdm(models.items(), desc="Training Models", unit="model"):
        print(f"\nTraining {name}...")
        
        # Fit model on preprocessed data
        model.fit(X_train_processed, y_train)
        model_objects[name] = model
        
        # Predictions on validation and test (preprocessed data)
        y_valid_pred = model.predict_proba(X_valid_processed)[:, 1]
        y_test_pred = model.predict_proba(X_test_processed)[:, 1]
        
        # AUC scores
        valid_auc = roc_auc_score(y_valid, y_valid_pred)
        test_auc = roc_auc_score(y_test, y_test_pred)
        
        # Bootstrap confidence intervals
        test_auc_mean, test_ci_lower, test_ci_upper = bootstrap_auc_ci(y_test, y_test_pred)
        
        results[name] = {
            'model': model,
            'preprocessor': preprocessor,  # Save preprocessor with each model
            'valid_auc': valid_auc,
            'test_auc': test_auc,
            'test_auc_ci': (test_ci_lower, test_ci_upper),
            'y_test_pred': y_test_pred,
            'X_train_shape': X_train_processed.shape,  # Track training data dimensions
            'seed_used': 42
        }
        
        print(f"  Validation AUC: {valid_auc:.4f}")
        print(f"  Test AUC: {test_auc:.4f} [{test_ci_lower:.4f}, {test_ci_upper:.4f}]")
    
    print("\n" + "=" * 60)
    print("Model Comparison Summary:")
    for name, result in results.items():
        auc = result['test_auc']
        ci_lower, ci_upper = result['test_auc_ci']
        print(f"{name:20s}: {auc:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Save models if requested
    if save_models:
        save_dir = "weights"
        save_trained_models(model_objects, results, save_dir)
        return results, model_objects, save_dir
    else:
        return results, model_objects


def plot_model_comparison(results):
    """
    Create forest plot for model AUC comparison
    
    Args:
        results: Dictionary with model results
    """
    model_names = list(results.keys())
    aucs = [results[name]['test_auc'] for name in model_names]
    ci_lowers = [results[name]['test_auc_ci'][0] for name in model_names]
    ci_uppers = [results[name]['test_auc_ci'][1] for name in model_names]
    
    # Sort by AUC
    sorted_indices = np.argsort(aucs)[::-1]
    model_names = [model_names[i] for i in sorted_indices]
    aucs = [aucs[i] for i in sorted_indices]
    ci_lowers = [ci_lowers[i] for i in sorted_indices]
    ci_uppers = [ci_uppers[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(model_names))
    
    # Plot points and error bars
    ax.errorbar(aucs, y_pos, 
                xerr=[np.array(aucs) - np.array(ci_lowers), 
                      np.array(ci_uppers) - np.array(aucs)],
                fmt='o', markersize=8, capsize=5, capthick=2, linewidth=2)
    
    # Customize plot with better formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names, fontsize=12, fontweight='bold')
    ax.set_xlabel('Test AUC', fontsize=14, fontweight='bold')
    ax.set_title('Model Performance Comparison\n(with 95% Bootstrap CIs)', fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Random Classifier')
    ax.tick_params(labelsize=11)
    
    # Add AUC values as text with better positioning
    for i, (auc, ci_lower, ci_upper) in enumerate(zip(aucs, ci_lowers, ci_uppers)):
        ax.text(auc + 0.015, i, f'{auc:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]', 
                va='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    ax.legend()
    plt.tight_layout()
    
    # Get figure before showing
    current_fig = plt.gcf()
    
    plt.show()
    
    return current_fig


def plot_roc_curves(results, y_test):
    """
    Plot ROC curves for all models
    
    Args:
        results: Dictionary with model results
        y_test: True test labels
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (name, result) in enumerate(results.items()):
        y_pred = result['y_test_pred']
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        auc = result['test_auc']
        
        ax.plot(fpr, tpr, color=colors[i], linewidth=2, 
                label=f'{name} (AUC = {auc:.3f})')
    
    # Plot diagonal line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves Comparison')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    # Get figure before showing
    current_fig = plt.gcf()
    
    plt.show()
    
    return current_fig


def get_best_model(results):
    """
    Get the best performing model based on validation AUC
    
    Args:
        results: Dictionary with model results
    
    Returns:
        tuple: (best_model_name, best_model_object)
    """
    best_name = max(results.keys(), key=lambda x: results[x]['valid_auc'])
    best_model = results[best_name]['model']
    
    print(f"Best model: {best_name} (Validation AUC: {results[best_name]['valid_auc']:.4f})")
    
    return best_name, best_model
