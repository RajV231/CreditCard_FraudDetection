import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score, f1_score, accuracy_score)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

class FraudDetectionModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_and_explore_data(self, file_path=None):
        self.df = pd.read_csv(file_path)
        print("Dataset Shape:", self.df.shape)
        return self.df

    def preprocess_data(self):
        feature_cols = [col for col in self.df.columns if col.startswith('V')]
        if 'Time' in self.df.columns: feature_cols.append('Time')
        if 'Amount' in self.df.columns: feature_cols.append('Amount')

        X = self.df[feature_cols]
        y = self.df['Class']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test

    def handle_class_imbalance(self, method='smote'):
        print(f"Handling imbalance with: {method}")
        if method == 'smote':
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_res, y_res = smote.fit_resample(self.X_train_scaled, self.y_train)
            return X_res, y_res
        else:
            return self.X_train_scaled, self.y_train

    def train_models(self, X_train, y_train, use_gridsearch=False):
        print("Training Logistic Regression...")
        lr = LogisticRegression(random_state=42, max_iter=500, class_weight='balanced')
        lr.fit(X_train, y_train)
        self.models['Logistic Regression'] = lr

        print("Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=30,
            max_depth=8,
            min_samples_split=5,
            n_jobs=-1,
            random_state=42
        )
        rf.fit(X_train, y_train)
        self.models['Random Forest'] = rf

        if use_gridsearch:
            print("Training Random Forest with Grid Search (optimized)...")
            param_grid = {
                'n_estimators': [30],
                'max_depth': [8, 10],
                'min_samples_split': [5]
            }
            rf_grid = GridSearchCV(
                RandomForestClassifier(random_state=42, n_jobs=-1),
                param_grid,
                cv=2,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            rf_grid.fit(X_train, y_train)
            self.models['Random Forest (Tuned)'] = rf_grid.best_estimator_
            print(f"Grid Search Best Params: {rf_grid.best_params_}")

        print("Model training completed!")
        return self.models

    def evaluate_models(self):
        results = {}
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test_scaled)
            y_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            results[name] = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1_score': f1_score(self.y_test, y_pred),
                'auc': roc_auc_score(self.y_test, y_proba),
                'y_pred': y_pred,
                'y_pred_proba': y_proba
            }
            print(f"\n{name} Results:")
            print(classification_report(self.y_test, y_pred))
            print("Confusion Matrix:")
            print(confusion_matrix(self.y_test, y_pred))
        return results

    def plot_results(self, results):
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        for name, res in results.items():
            fpr, tpr, _ = roc_curve(self.y_test, res['y_pred_proba'])
            plt.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve')
        plt.legend()

        plt.subplot(1, 2, 2)
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
        for i, metric in enumerate(metric_names):
            values = [results[m][metric] for m in results]
            plt.bar(np.arange(len(values)) + i*0.15, values, width=0.15, label=metric)
        plt.xticks(np.arange(len(results)) + 0.3, list(results.keys()), rotation=45)
        plt.ylim(0, 1.1)
        plt.legend()
        plt.title('Model Performance Metrics')
        plt.tight_layout()
        plt.show()

    def feature_importance(self):
        model = self.models.get('Random Forest')
        if model and hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            features = [col for col in self.df.columns if col.startswith('V')]
            indices = np.argsort(importances)[::-1][:10]
            plt.bar(range(len(indices)), importances[indices])
            plt.xticks(range(len(indices)), [features[i] for i in indices], rotation=45)
            plt.title("Top 10 Feature Importances")
            plt.show()


def main():
    fraud_detector = FraudDetectionModel()
    fraud_detector.load_and_explore_data('creditcard.csv')

    if 'Class' not in fraud_detector.df.columns:
        print("No 'Class' column found.")
        return

    X_train, X_test, y_train, y_test = fraud_detector.preprocess_data()
    X_resampled, y_resampled = fraud_detector.handle_class_imbalance(method='smote')

    fraud_detector.train_models(X_resampled, y_resampled, use_gridsearch=False)
    results = fraud_detector.evaluate_models()
    fraud_detector.plot_results(results)
    fraud_detector.feature_importance()

    best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
    print(f"\nBest Model: {best_model[0]}")
    print(f"F1 Score: {best_model[1]['f1_score']:.4f}")
    print(f"Precision: {best_model[1]['precision']:.4f}")
    print(f"Recall: {best_model[1]['recall']:.4f}")
    print(f"AUC-ROC: {best_model[1]['auc']:.4f}")

if __name__ == "__main__":
    main()
