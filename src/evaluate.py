from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc, roc_auc_score
from visualisation import cm_reference_models

def evaluate_models(y_test, y_pred, model_name, best_model):

      print(classification_report(y_test, y_pred))

      cm_reference_models(y_test=y_test, y_pred=y_pred,labels=best_model.classes_,model_name=model_name)

