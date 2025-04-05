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

X_encoded = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

X_train_ordinal = X_train.copy()
X_test_ordinal = X_test.copy()

for col in X_train.columns:
    X_train_ordinal[col] = X_train[col].astype('category').cat.codes
    
    categories = X_train[col].unique()
    
    X_test_ordinal[col] = pd.Categorical(X_test[col], categories=categories).codes

X_train_target = X_train.copy()
X_test_target = X_test.copy()

global_mean = y_train.mean()

target_encoding_maps = {}

for col in X_train.columns:
    mapping = X_train.join(y_train, rsuffix='_target').groupby(col)['class'].mean().to_dict()

    target_encoding_maps[col] = mapping

    X_train_target[col] = X_train[col].map(mapping)

    X_test_target[col] = X_test[col].map(mapping).fillna(global_mean)

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

model = GaussianNaiveBayes()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = np.mean(y_pred == np.array(y_test))
print("One-Hot Encoding - Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("One-Hot Encoding - Confusion Matrix:")
print(cm)

model_ordinal = GaussianNaiveBayes()
model_ordinal.fit(X_train_ordinal, y_train)

y_pred_ordinal = model_ordinal.predict(X_test_ordinal)
accuracy_ordinal = np.mean(y_pred_ordinal == np.array(y_test))
print("\nOrdinal Encoding - Accuracy:", accuracy_ordinal)

cm_ordinal = confusion_matrix(y_test, y_pred_ordinal)
print("Ordinal Encoding - Confusion Matrix:")
print(cm_ordinal)

model_target = GaussianNaiveBayes()
model_target.fit(X_train_target, y_train)

y_pred_target = model_target.predict(X_test_target)
accuracy_target = np.mean(y_pred_target == np.array(y_test))
print("\nTarget Encoding - Accuracy:", accuracy_target)

cm_target = confusion_matrix(y_test, y_pred_target)
print("Target Encoding - Confusion Matrix:")
print(cm_target)

selected_features = rfe(X_train, y_train, X_test, y_test, 112)

X_train_rfe = X_train[selected_features]
X_test_rfe = X_test[selected_features]

model_rfe = GaussianNaiveBayes()
model_rfe.fit(X_train_rfe, y_train)

y_pred_rfe = model_rfe.predict(X_test_rfe)

accuracy_rfe = np.mean(y_pred_rfe == np.array(y_test))
print("\nFinal RFE One-Hot Encoding Model - Accuracy:", accuracy_rfe)

print("Final RFE One-Hot Encoding Model - Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rfe))

selected_features = rfe(X_train_ordinal, y_train, X_test_ordinal, y_test, 80)

X_train_rfe_ordinal = X_train_ordinal[selected_features]
X_test_rfe_ordinal = X_test_ordinal[selected_features]

model_rfe = GaussianNaiveBayes()
model_rfe.fit(X_train_rfe_ordinal, y_train)

y_pred_rfe = model_rfe.predict(X_test_rfe_ordinal)

accuracy_rfe = np.mean(y_pred_rfe == np.array(y_test))
print("\nFinal RFE Ordinal Encoding Model - Accuracy:", accuracy_rfe)

print("Final RFE Ordinal Encoding Model - Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rfe))

selected_features = rfe(X_train_target, y_train, X_test_target, y_test, 112)

X_train_rfe_target = X_train_target[selected_features]
X_test_rfe_target = X_test_target[selected_features]

model_rfe = GaussianNaiveBayes()
model_rfe.fit(X_train_rfe_target, y_train)

y_pred_rfe = model_rfe.predict(X_test_rfe_target)

accuracy_rfe = np.mean(y_pred_rfe == np.array(y_test))
print("\nFinal RFE Target Encoding Model - Accuracy:", accuracy_rfe)

print("Final RFE Target Encoding Model - Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rfe))


