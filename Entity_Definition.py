import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_f1_score
from tqdm import tqdm

def word2features(tokens, i):
    word = tokens[i]
    features = {
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'is_first': i == 0,
        'is_last': i == len(tokens) - 1,
        'prev_word': '' if i == 0 else tokens[i - 1],
        'next_word': '' if i == len(tokens) - 1 else tokens[i + 1]
    }
    return features

def sentence_to_features(tokens):
    return [word2features(tokens, i) for i in range(len(tokens))]

tokenized_data_path = 'tokenized_with_entities.csv'
tokenized_data = pd.read_csv(tokenized_data_path)

X = []
y = []

for _, row in tqdm(tokenized_data.iterrows(), total=tokenized_data.shape[0], desc="Preparing Data"):
    tokens = ast.literal_eval(row['Tokenized Text'])
    entities = ast.literal_eval(row['Entities'])
    X.append(sentence_to_features(tokens))
    y.append([entity for _, entity in entities])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
crf.fit(X_train, y_train)

y_pred = crf.predict(X_test)
f1 = flat_f1_score(y_test, y_pred, average='weighted')
print(f"F1 Score (Weighted): {f1}")

predictions_df = pd.DataFrame({'ID': tokenized_data['ID'][len(y_train):].values, 'Predicted Entities': y_pred})
predictions_df.to_csv('itog_sych.csv', index=False)
