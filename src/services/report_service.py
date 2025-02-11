import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold


class ReportService:
    def __init__(self, model, X_train, y_train, X_val, y_val):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def run_cross_validation(self):
        """
        Perform K-Fold Cross Validation to get validation accuracy for different data splits.
        """
        kf = StratifiedKFold(n_splits=5)
        validation_accuracies = []

        for train_idx, val_idx in kf.split(self.X_train, self.y_train):
            X_train_fold, X_val_fold = self.X_train[train_idx], self.X_train[val_idx]
            y_train_fold, y_val_fold = self.y_train[train_idx], self.y_train[val_idx]

            self.model.fit(X_train_fold, y_train_fold)
            y_val_pred = self.model.predict(X_val_fold)
            val_accuracy = accuracy_score(y_val_fold, y_val_pred)
            validation_accuracies.append(val_accuracy)

        avg_val_accuracy = np.mean(validation_accuracies)
        print(f"Avg Cross-Validation Accuracy: {avg_val_accuracy:.4f}")
        return avg_val_accuracy

    def plot_learning_curves(self):
        """
        Plot the learning curves (train vs validation loss) for models that support it.
        """
        # Only plot for tree-based models (those with get_evals_result)
        if hasattr(self.model, "get_evals_result"):
            try:
                # Get eval results (train and validation losses)
                eval_result = self.model.get_evals_result()

                # Plot training and validation losses
                plt.figure(figsize=(10, 6))

                # Extract and plot training and validation losses
                train_loss = eval_result["learn"]["MultiClass"]
                val_loss = eval_result["validation"]["MultiClass"]

                plt.plot(train_loss, label="Train Loss")
                plt.plot(val_loss, label="Validation Loss")

                # Labels and title
                plt.xlabel("Iterations")
                plt.ylabel("Loss")
                plt.title("Learning Curves (Train vs Validation Loss)")
                plt.legend()
                plt.show()

            except Exception as e:
                print(f"Error in plotting learning curves: {e}")
        else:
            print(
                "Learning curves not available for non-tree-based models like Logistic Regression."
            )

    def generate_classification_report(self):
        """
        Generate classification report for model evaluation.
        """
        y_val_pred = self.model.predict(self.X_val)
        report = classification_report(self.y_val, y_val_pred)
        print("Classification Report:\n", report)
        return report

    def generate_report(self):
        """
        Generate full report including cross-validation, learning curves, and classification report.
        """
        # Cross-validation performance
        avg_val_accuracy = self.run_cross_validation()

        # Learning curve plot (only for tree-based models)
        self.plot_learning_curves()

        # Classification report
        classification_report_str = self.generate_classification_report()

        report = {
            "avg_val_accuracy": avg_val_accuracy,
            "classification_report": classification_report_str,
        }

        return report
