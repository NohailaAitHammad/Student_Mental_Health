import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc, roc_auc_score
def roc_auc_curve(y_test, y_test_bin, y_score, model_name, classes):

      # Calcul du ROC-AUC multiclasse
      
      roc_auc_macro = roc_auc_score(
            y_test,
            y_score,
            labels=classes,
            multi_class='ovr',
            average='macro'
      )
      
      # Courbes ROC
      plt.figure(figsize=(7, 5))

      auc_per_class = {}
      for i, class_name in enumerate(classes):
      
            fpr, tpr, thresholds = roc_curve(
                  y_test_bin[:, i],
                  y_score[:, i]
            )
      
            auc_value = auc(fpr, tpr)
      
            auc_per_class[class_name] = auc_value
      
            plt.plot(
                  fpr,
                  tpr,
                  label=f'{class_name} (AUC = {auc_value:.2f})'
            )
      
      # Ligne du classificateur aléatoire
      plt.plot(
            [0, 1],
            [0, 1],
            'k--',
            label='Random Guess'
      )
      
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title(f'ROC Curve - {model_name}')
      plt.legend(loc='lower right')
      plt.grid(alpha=0.3)
      # plt.show()

      plt.savefig(f'reports/roc_auc/ROC_Curve_{model_name}.png')
      plt.close()

def cm_reference_models(y_test, y_pred, labels, model_name):

      # Matrice de confusion
      ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            labels=labels,
            display_labels=labels
      )
      
      plt.title(f'Confusion Matrix - {model_name}')
      # plt.show()
      plt.savefig(f'reports/cm/Confusion Matrix_{model_name}.png')
      plt.close()
      

