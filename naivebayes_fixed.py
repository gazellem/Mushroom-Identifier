import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df = pd.read_csv('mushrooms.csv')

target_mapping = {'e': 0, 'p': 1}
df['class'] = df['class'].map(target_mapping)

X = df.drop('class', axis=1)
y = df['class']

X_train_orig, X_test_orig, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_oh = pd.get_dummies(X_train_orig)
X_test_oh = pd.get_dummies(X_test_orig)

X_train_oh, X_test_oh = X_train_oh.align(X_test_oh, join='outer', axis=1, fill_value=0)

X_train_ord = X_train_orig.copy()
X_test_ord = X_test_orig.copy()
for col in X_train_orig.columns:
    X_train_ord[col] = X_train_orig[col].astype('category').cat.codes
    categories = X_train_orig[col].unique()
    X_test_ord[col] = pd.Categorical(X_test_orig[col], categories=categories).codes

X_train_tgt = X_train_orig.copy()
X_test_tgt = X_test_orig.copy()
global_mean = y_train.mean()
target_encoding_maps = {}
for col in X_train_orig.columns:
    mapping = X_train_orig.join(y_train, rsuffix='_target').groupby(col)['class'].mean().to_dict()
    target_encoding_maps[col] = mapping
    X_train_tgt[col] = X_train_orig[col].map(mapping)
    X_test_tgt[col] = X_test_orig[col].map(mapping).fillna(global_mean)

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.priors = {}
        self.means = {}
        self.vars = {}

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classes = np.unique(y)
        for cls in self.classes:
            X_c = X[y == cls]
            self.priors[cls] = X_c.shape[0] / X.shape[0]
            self.means[cls] = np.mean(X_c, axis=0)
            self.vars[cls] = np.var(X_c, axis=0)

    def _calculate_log_probability(self, x, mean, var):
        eps = 1e-9
        var = var + eps
        return -0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2) / (2 * var)

    def predict(self, X):
        X = np.array(X)
        predictions = []
        for x in X:
            log_probs = {}
            for cls in self.classes:
                log_prob = np.log(self.priors[cls])
                log_prob += np.sum(self._calculate_log_probability(x, self.means[cls], self.vars[cls]))
                log_probs[cls] = log_prob
            predictions.append(max(log_probs, key=log_probs.get))
        return np.array(predictions)

def rfe(X_train, y_train, X_test, y_test, n_features_to_select):
    features = list(X_train.columns)
    while len(features) > n_features_to_select:
        best_acc = -1
        feature_to_remove = None
        for feature in features:
            current_features = [f for f in features if f != feature]
            X_train_subset = X_train[current_features]
            X_test_subset = X_test[current_features]
            model_temp = GaussianNaiveBayes()
            model_temp.fit(X_train_subset, y_train)
            y_pred_temp = model_temp.predict(X_test_subset)
            acc = np.mean(y_pred_temp == np.array(y_test))
            if acc > best_acc:
                best_acc = acc
                feature_to_remove = feature
        features.remove(feature_to_remove)
        print("Removed:", feature_to_remove, "Accuracy:", best_acc, "Remaining features:", len(features))
    return features

model_oh = GaussianNaiveBayes()
model_oh.fit(X_train_oh, y_train)
y_pred_oh = model_oh.predict(X_test_oh)
accuracy_oh = np.mean(y_pred_oh == np.array(y_test))
print("One-Hot Encoding - Accuracy:", accuracy_oh)
print("One-Hot Encoding - Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_oh))

model_ord = GaussianNaiveBayes()
model_ord.fit(X_train_ord, y_train)
y_pred_ord = model_ord.predict(X_test_ord)
accuracy_ord = np.mean(y_pred_ord == np.array(y_test))
print("\nOrdinal Encoding - Accuracy:", accuracy_ord)
print("Ordinal Encoding - Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_ord))

model_tgt = GaussianNaiveBayes()
model_tgt.fit(X_train_tgt, y_train)
y_pred_tgt = model_tgt.predict(X_test_tgt)
accuracy_tgt = np.mean(y_pred_tgt == np.array(y_test))
print("\nTarget Encoding - Accuracy:", accuracy_tgt)
print("Target Encoding - Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tgt))

selected_features_oh = rfe(X_train_oh, y_train, X_test_oh, y_test, 112)
X_train_rfe_oh = X_train_oh[selected_features_oh]
X_test_rfe_oh = X_test_oh[selected_features_oh]
model_rfe_oh = GaussianNaiveBayes()
model_rfe_oh.fit(X_train_rfe_oh, y_train)
y_pred_rfe_oh = model_rfe_oh.predict(X_test_rfe_oh)
accuracy_rfe_oh = np.mean(y_pred_rfe_oh == np.array(y_test))
print("\nFinal RFE One-Hot Encoding Model - Accuracy:", accuracy_rfe_oh)
print("Final RFE One-Hot Encoding Model - Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rfe_oh))

selected_features_ord = rfe(X_train_ord, y_train, X_test_ord, y_test, 20)
X_train_rfe_ord = X_train_ord[selected_features_ord]
X_test_rfe_ord = X_test_ord[selected_features_ord]
model_rfe_ord = GaussianNaiveBayes()
model_rfe_ord.fit(X_train_rfe_ord, y_train)
y_pred_rfe_ord = model_rfe_ord.predict(X_test_rfe_ord)
accuracy_rfe_ord = np.mean(y_pred_rfe_ord == np.array(y_test))
print("\nFinal RFE Ordinal Encoding Model - Accuracy:", accuracy_rfe_ord)
print("Final RFE Ordinal Encoding Model - Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rfe_ord))

selected_features_tgt = rfe(X_train_tgt, y_train, X_test_tgt, y_test, 20)
X_train_rfe_tgt = X_train_tgt[selected_features_tgt]
X_test_rfe_tgt = X_test_tgt[selected_features_tgt]
model_rfe_tgt = GaussianNaiveBayes()
model_rfe_tgt.fit(X_train_rfe_tgt, y_train)
y_pred_rfe_tgt = model_rfe_tgt.predict(X_test_rfe_tgt)
accuracy_rfe_tgt = np.mean(y_pred_rfe_tgt == np.array(y_test))
print("\nFinal RFE Target Encoding Model - Accuracy:", accuracy_rfe_tgt)
print("Final RFE Target Encoding Model - Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rfe_tgt))
