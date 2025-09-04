from sklearn.metrics import classification_report, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt

def report(y_true, y_pred):
    return classification_report(y_true, y_pred, target_names=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
def pr_curve(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true,y_pred)

    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

def roc_auc(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true,y_pred)

    plt.plot(fpr,tpr)
    plt.plot([0,1],[0,1], style = "--", color='black', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve:{auc(fpr, tpr)}')
    plt.show()

    