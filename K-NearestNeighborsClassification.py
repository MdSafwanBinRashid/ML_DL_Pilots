from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

data = load_iris()

print("Features:", data.feature_names)
print("Targets:", data.target_names)

x_train, x_test, y_train, y_test = train_test_split(np.array(data.data), np.array(data.target), test_size=0.2)

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])  # demo measurements
prediction = clf.predict(new_flower)

print(f"Predicted flower type: {data.target_names[prediction[0]]}")

