import sklearn
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_roc(y_val, y_pred):
    fpr, tpr, _ = roc_curve(y_val, y_pred)
    roc_auc = sklearn.metrics.auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()


def plot_cm(true_labels, predicted_labels):

    cm = confusion_matrix(true_labels, predicted_labels)

    # Plot confusion matrix as heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True,
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.show()
    print('cm')
    print(cm)


def plot_feat_importance(cls, columns, K=20):
    feature_importance = cls.feature_importances_
    feature_importance_dict = dict(zip(columns, feature_importance))

    # Sort the dictionary by importance scores (descending order)
    sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    # Get the top 10 features with the highest importance scores
    top_features = sorted_feature_importance[:K]

    # Print the top 10 features with their importance scores
    for feature, importance in top_features:
        print(f"{feature}: {importance}")

    # Visualize the top 10 features
    top_features_names = [x[0] for x in top_features]
    top_importance_scores = [x[1] for x in top_features]

    matplotlib.rcParams['font.family'] = 'Heiti TC'
    plt.figure(figsize=(12, 6))  # Increase figure width to accommodate longer tick labels
    plt.barh(range(len(top_importance_scores)), top_importance_scores)
    plt.yticks(range(len(top_features_names)), top_features_names)
    plt.xlabel('Importance Score')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()  # Invert y-axis to display top features at the top
    plt.show()