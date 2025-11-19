from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train, y_train)

forest_clf = RandomForestClassifier(n_estimators=100)
forest_clf.fit(X_train, y_train)

tree_accuracy = tree_clf.score(X_test, y_test)
forest_accuracy = forest_clf.score(X_test, y_test)

print(f'Decision Tree Accuracy: {tree_accuracy:.3f}')
print(f'Random Forest Accuracy: {forest_accuracy:.3f}')

tree_pred = tree_clf.predict([X_test[0]])
forest_pred = forest_clf.predict([X_test[0]])

print(f"\nTrue Value: {y_test[0]}") # 1 - Benign (non-cancerous), 0 - Malignant (cancerous)
print(f"Decision Tree Predicted: {'Benign' if tree_pred[0] == 1 else 'Malignant'}")
print(f"Random Forest Predicted: {'Benign' if forest_pred[0] == 1 else 'Malignant'}")
