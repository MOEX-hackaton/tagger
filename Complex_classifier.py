import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from tqdm import tqdm

# Загрузка обработанного датасета
processed_data_path = '../DataWork/processed_train_dataset.csv'
processed_data = pd.read_csv(processed_data_path)

# Замена NaN значений пустой строкой
processed_data['Текст инцидента'].fillna('', inplace=True)

# Предобработка данных
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(processed_data['Текст инцидента'])
y_theme = processed_data['Тема']
y_group = processed_data['Группа тем']

# Разделение данных
X_train_theme, X_test_theme, y_train_theme, y_test_theme = train_test_split(X, y_theme, test_size=0.2, random_state=42)
X_train_group, X_test_group, y_train_group, y_test_group = train_test_split(X, y_group, test_size=0.2, random_state=42)

# Настройка GridSearchCV для RandomForestClassifier
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    # Дополнительные параметры
}
grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=3, scoring='f1_weighted')
grid_search_rf.fit(X_train_group, y_train_group)
best_model_group = grid_search_rf.best_estimator_

# Обучение GradientBoostingClassifier для темы
model_theme = GradientBoostingClassifier(random_state=42)
model_theme.fit(X_train_theme, y_train_theme)

# Оценка модели для темы
predictions_theme = model_theme.predict(X_test_theme)
f1_theme = f1_score(y_test_theme, predictions_theme, average='weighted')
print("Classification Report for Theme:")
print(classification_report(y_test_theme, predictions_theme))
print(f"F1 Score (Weighted) for Theme: {f1_theme}")

# Оценка лучшей модели для группы тем
predictions_group = best_model_group.predict(X_test_group)
f1_group = f1_score(y_test_group, predictions_group, average='weighted')
print("\nClassification Report for Group:")
print(classification_report(y_test_group, predictions_group))
print(f"F1 Score (Weighted) for Group: {f1_group}")

# Сохранение предсказаний в новый файл с прогресс баром
predictions_df = pd.DataFrame()
for i, (idx, row) in tqdm(enumerate(processed_data.iterrows()), total=processed_data.shape[0], desc="Predicting and Saving"):
    if i < len(y_train_theme):  # Skip training data
        continue
    theme_prediction = model_theme.predict(tfidf.transform([row['Текст инцидента']]))[0]
    group_prediction = best_model_group.predict(tfidf.transform([row['Текст инцидента']]))[0]
    predictions_df = predictions_df.append({'ID': idx, 'Тема': theme_prediction, 'Группа тем': group_prediction}, ignore_index=True)
    if i % 1000 == 0:  # Save partial results
        predictions_df.to_csv('classification_predictions_partial.csv', index=False)

predictions_df.to_csv('classification_predictions.csv', index=False)

# Визуализация F1 Score
plt.figure(figsize=(10, 6))
plt.bar(y_test_theme.unique(), [f1_score(y_test_theme, predictions_theme, average='weighted', labels=[label]) for label in y_test_theme.unique()], alpha=0.5, label='Theme')
plt.bar(y_test_group.unique(), [f1_score(y_test_group, predictions_group, average='weighted', labels=[label]) for label in y_test_group.unique()], alpha=0.5, label='Group')
plt.xlabel('Categories')
plt.ylabel('F1 Score')
plt.title('F1 Score for Theme and Group Categories')
plt.xticks(rotation=45)
plt.legend()
plt.show()
