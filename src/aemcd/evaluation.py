import numpy as np
from sklearn.metrics import precision_recall_fscore_support , confusion_matrix ,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score

# Point-adjusted evaluation if any point in true anomaly segment is detected, the whole segment is counted as detected ,recall will increase
def point_adjust(y_true, y_pred):
   
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    adjusted_pred = y_pred.copy()
    
   
    # find segments in y_true
    in_segment = False
    start = 0
    
    for i, label in enumerate(y_true):
        if label == 1 and not in_segment:
            in_segment = True
            start = i
        elif label == 0 and in_segment:
            end = i
            if y_pred[start:end].sum() > 0:
                adjusted_pred[start:end] = 1
            in_segment = False
    
    if in_segment:
        end = len(y_true)
        if y_pred[start:end].sum() > 0:
            adjusted_pred[start:end] = 1

    return adjusted_pred

def evaluate_and_report(y_true, y_pred, y_pred_adjusted=None, title="Results",y_score=None):
    """ Evaluate and print classification report and confusion matrix """
   
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    # original predictions
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, 
                                                   average="binary", 
                                                   zero_division=0)
    from sklearn.metrics import accuracy_score, roc_auc_score
    acc = accuracy_score(y_true, y_pred)
    auc = None
    ap = None
    if y_score is not None:
        y_score = np.asarray(y_score, dtype=float)
        try:
            auc = roc_auc_score(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
        except Exception:
            auc, ap = None, None

    print(f"\n original index:")
    print(f"  Accuracy  = {acc:.3f}")
    print(f"  Precision = {p:.3f}")
    print(f"  Recall    = {r:.3f}")
    print(f"  F1-Score  = {f1:.3f}")
    if auc is not None:
        print(f"  ROC-AUC   = {auc:.3f}   (from score)")
    if ap is not None:
        print(f"  PR-AUC    = {ap:.3f}   (from score)")
    
    # Point-Adjusted predictions
    if y_pred_adjusted is not None:
        p_adj, r_adj, f1_adj, _ = precision_recall_fscore_support(
            y_true, y_pred_adjusted, average="binary", zero_division=0
        )
        acc_adj = accuracy_score(y_true, y_pred_adjusted)
        
        try:
            auc_adj = roc_auc_score(y_true, y_pred_adjusted)
        except:
            auc_adj = None
        
        print(f"\n Point-Adjusted prdict index:")
        print(f"  Accuracy  = {acc_adj:.3f}  (Δ = {acc_adj - acc:+.3f})")
        print(f"  Precision = {p_adj:.3f}  (Δ = {p_adj - p:+.3f})")
        print(f"  Recall    = {r_adj:.3f}  (Δ = {r_adj - r:+.3f})")
        print(f"  F1-Score  = {f1_adj:.3f}  (Δ = {f1_adj - f1:+.3f})")
       
        try:
            auc_adj = roc_auc_score(y_true, y_pred_adjusted)
            print(f"  ROC-AUC   = {auc_adj:.3f}  (from PA 0/1 preds)")
        except:
            pass

    
    
    
    
    
    
    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n  confusion matrix  :")
    print(f"              predicted")
    print(f"            Normal  Anomaly")
    print(f"  Practical Normal   {tn:5d}   {fp:5d}")
    print(f"     Anomaly   {fn:5d}   {tp:5d}")
    
    print(f"\n statical summary:")
    print(f"  True Negatives  (TN) = {tn:5d}")
    print(f"  False Positives (FP) = {fp:5d}  ← False alarm")
    print(f"  False Negatives (FN) = {fn:5d}  ← Missed alarm")
    print(f"  True Positives  (TP) = {tp:5d}")
    
    # ---- PA confusion matrix  ----
    if y_pred_adjusted is not None:
        cm_pa = confusion_matrix(y_true, y_pred_adjusted)
        tn2, fp2, fn2, tp2 = cm_pa.ravel()
        print(f"\n confusion matrix (Point-Adjusted):")
        print(f"              預測")
        print(f"            Normal  Anomaly")
        print(f"  實際 Normal   {tn2:5d}   {fp2:5d}")
        print(f"     Anomaly   {fn2:5d}   {tp2:5d}")
        print(f"  Manual PA Recall = {tp2/(tp2+fn2+1e-12):.3f}")
    # plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'],
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix - {title}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"{'='*70}\n")
