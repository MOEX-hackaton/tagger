import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

processed_data = pd.read_csv('../DataWork/processed_train_dataset.csv')

processed_data['Текст инцидента'].fillna('', inplace=True)

tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(processed_data['Текст инцидента'])
y = processed_data['Тема']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
f1 = f1_score(y_test, predictions, average='weighted')
print(classification_report(y_test, predictions))
print(f"F1 Score (Weighted): {f1}")

predictions_df = pd.DataFrame({'ID': y_test.index, 'Тема': predictions})
predictions_df.to_csv('classification_predictions.csv', index=False)

plt.figure(figsize=(10, 6))
plt.bar(y_test.unique(), [f1_score(y_test, predictions, average='weighted', labels=[label]) for label in y_test.unique()])
plt.xlabel('Тема')
plt.ylabel('F1 Score')
plt.title('F1 Score по темам')
plt.xticks(rotation=45)
plt.show()
