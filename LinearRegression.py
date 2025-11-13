import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

hours_studied = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 3, 5, 7, 9, 2, 4]).reshape(-1, 1)
scores = np.array([50, 58, 65, 70, 75, 78, 82, 85, 88, 90, 62, 72, 80, 86, 55, 68])
# features must be 2D (for scikit-learn), hence reshaping

time_train, time_test, score_train, score_test = train_test_split(hours_studied, scores, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(time_train, score_train)

accuracy = (model.score(time_test, score_test)) * 100

hours = float(input("Enter hours studied: "))
prediction = model.predict([[hours]])

plt.scatter(hours_studied, scores, color='blue', label='Actual Scores')
plt.plot(hours_studied, model.predict(hours_studied), color='red', label='Prediction Line')
plt.scatter(hours, prediction, color='green', s=100, label=f'Prediction: {prediction[0]:.1f}')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title(f'Time Spent vs Score (Accuracy: {accuracy:.2f})')
plt.legend()
plt.grid(True)
plt.show()

