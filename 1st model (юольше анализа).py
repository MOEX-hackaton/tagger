import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

processed_data = pd.read_csv('processed_train_dataset.csv')
processed_data['Текст инцидента'].fillna('', inplace=True)

tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(processed_data['Текст инцидента'])
y = processed_data['Тема']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

f1 = f1_score(y_test, predictions, average='weighted')
precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
recall = recall_score(y_test, predictions, average='weighted', zero_division=0)

print(classification_report(y_test, predictions))
print(f"F1 Score (Weighted): {f1}")
print(f"Precision (Weighted): {precision}")
print(f"Recall (Weighted): {recall}")

unique_labels = y_test.unique()
f1_scores = [f1_score(y_test, predictions, average='weighted', labels=[label]) for label in unique_labels]
precision_scores = [precision_score(y_test, predictions, average='weighted', labels=[label], zero_division=0) for label in unique_labels]
recall_scores = [recall_score(y_test, predictions, average='weighted', labels=[label], zero_division=0) for label in unique_labels]

plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.bar(np.arange(len(unique_labels)), f1_scores, color='blue', alpha=0.7)
plt.xlabel('Тема')
plt.ylabel('F1 Score')
plt.title('F1 Score по темам')
plt.xticks(np.arange(len(unique_labels)), unique_labels, rotation=45)

plt.subplot(1, 3, 2)
plt.bar(np.arange(len(unique_labels)), precision_scores, color='green', alpha=0.7)
plt.xlabel('Тема')
plt.ylabel('Precision')
plt.title('Precision по темам')
plt.xticks(np.arange(len(unique_labels)), unique_labels, rotation=45)

plt.subplot(1, 3, 3)
plt.bar(np.arange(len(unique_labels)), recall_scores, color='red', alpha=0.7)
plt.xlabel('Тема')
plt.ylabel('Recall')
plt.title('Recall по темам')
plt.xticks(np.arange(len(unique_labels)), unique_labels, rotation=45)

plt.tight_layout()
plt.show()

