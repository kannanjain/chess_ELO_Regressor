

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, auc, average_precision_score
)
from sklearn.preprocessing import label_binarize
import joblib
from pathlib import Path
import matplotlib.pyplot as plt


def load_features(csv_path):
    print(f"Loading {csv_path}.")
    df = pd.read_csv(csv_path)
    
    
    feature_cols = [f'feature_{i}' for i in range(768)]
    
    available_features = [col for col in feature_cols if col in df.columns]
    if len(available_features) != 768:
        print(f"Warning: Expected 768 features, found {len(available_features)}")
        print("Running feature extraction first...")
        return None
    
    X = df[feature_cols].values
    y = df['OpeningFamily'].values
    
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    print(f"Opening families: {len(np.unique(y))}")
    
    return X, y


def precision_at_k(y_true, y_pred_proba, classes, k=3):
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    y_true_idx = np.array([label_to_idx[label] for label in y_true])
   
    top_k_indices = np.argsort(y_pred_proba, axis=1)[:, -k:][:, ::-1]
    
    hits = []
    for i, true_idx in enumerate(y_true_idx):
        if true_idx in top_k_indices[i]:
            hits.append(1.0)
        else:
            hits.append(0.0)
    
    return np.mean(hits)


def recall_at_k(y_true, y_pred_proba, classes, k=3):
    return precision_at_k(y_true, y_pred_proba, classes, k)


def map_at_k(y_true, y_pred_proba, classes, k=3):
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    y_true_idx = np.array([label_to_idx[label] for label in y_true])
    
    map_scores = []
    for i, true_idx in enumerate(y_true_idx):
        top_k_indices = np.argsort(y_pred_proba[i])[-k:][::-1]
        
        if true_idx in top_k_indices:
            rank = np.where(top_k_indices == true_idx)[0][0] + 1
            ap = 1.0 / rank
        else:
            ap = 0.0
        map_scores.append(ap)
    
    return np.mean(map_scores)


def plot_pr_curve(y_test, y_test_proba, classes, output_path):
    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = len(classes)
    
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(classes):
        precision, recall, _ = precision_recall_curve(
            y_test_bin[:, i], y_test_proba[:, i]
        )
        ap = average_precision_score(y_test_bin[:, i], y_test_proba[:, i])
        plt.plot(recall, precision, label=f'{class_name} (AP={ap:.3f})', linewidth=2)
    

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"PR curve saved to {output_path}")
    plt.close()


def plot_roc_curve(y_test, y_test_proba, classes, output_path):
    
    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = len(classes)
    

    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_test_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC={roc_auc:.3f})', linewidth=2)
    
 
    fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_test_proba.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro,
             label=f'Micro-average (AUC={roc_auc_micro:.3f})',
             linewidth=3, linestyle='--', color='black')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve (One-vs-Rest)', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"ROC curve saved to {output_path}")
    plt.close()


def train_random_forest(X_train, y_train, X_test, y_test):
    print("\n" + "="*60)
    print("="*60)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    print("Fitting model...")
    model.fit(X_train, y_train)
    
 
    print("\nEvaluating on test set...")
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
   
    y_test_proba = model.predict_proba(X_test)
    model_classes = model.classes_
    
    
    print("\n" + "="*60)
    print("Top-K Metrics (Test Set):")
    print("="*60)
    prec_at_3 = precision_at_k(y_test, y_test_proba, model_classes, k=3)
    rec_at_3 = recall_at_k(y_test, y_test_proba, model_classes, k=3)
    map_at_3 = map_at_k(y_test, y_test_proba, model_classes, k=3)
    
    print(f"Precision@3: {prec_at_3:.4f}")
    print(f"Recall@3: {rec_at_3:.4f}")
    print(f"MAP@3: {map_at_3:.4f}")
    
    
    print("\n" + "="*60)
    print("Classification Report (Test Set):")
    print("="*60)
    report_dict = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
    
    
    print(f"{'Class':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 66)
    
    
    for class_name in model_classes:
        if class_name in report_dict:
            metrics = report_dict[class_name]
            print(f"{class_name:<30} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1-score']:<12.4f}")
    
 
    print("-" * 66)
   
    print(f"Accuracy: {report_dict['accuracy']:<12.4f}")
    
  
    feature_importance = model.feature_importances_
    print(f"\nTop 10 most important features:")
    top_indices = np.argsort(feature_importance)[-10:][::-1]
    for idx in top_indices:
        print(f"  Feature {idx}: {feature_importance[idx]:.6f}")
   
    script_dir = Path(__file__).parent
    plot_pr_curve(y_test, y_test_proba, model_classes, script_dir / 'pr_curve.png')
    plot_roc_curve(y_test, y_test_proba, model_classes, script_dir / 'roc_curve.png')
    
    return model


def main():
    script_dir = Path(__file__).parent
    

    X_train, y_train = load_features(script_dir / 'train_features.csv')
    if X_train is None:
        print("Error: Could not load training features. Run features.py first.")
        return
    
    
    X_test, y_test = load_features(script_dir / 'test_features.csv')
    if X_test is None:
        print("Error: Could not load test features. Run features.py first.")
        return
    
 
    model = train_random_forest(X_train, y_train, X_test, y_test)
    
  
    model_path = script_dir / 'opening_classifier.pkl'
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)


if __name__ == "__main__":
    main()

