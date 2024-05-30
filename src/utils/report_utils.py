from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix

def generate_reports(name, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    
    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr') if len(set(y_true)) > 2 else roc_auc_score(y_true, y_pred)
    else:
        auc = 0
    
    result = {
        "acc": acc,
        "f1": f1,
        "conf": confusion_matrix(y_true, y_pred),
        "auc": auc
    }
    print(f"========{name}=========")
    print(f"{acc} & {f1} & {auc}")
    return result