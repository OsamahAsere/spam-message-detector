import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB

df = pd.read_csv('spam_ham_dataset.csv')
df = df[['text', 'label_num']]
df.columns = ['message', 'label']

X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

vectorizer = CountVectorizer(stop_words='english', lowercase=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

nb_model = ComplementNB()
nb_model.fit(X_train_vec, y_train)

lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr_model.fit(X_train_vec, y_train)

nb_pred = nb_model.predict(X_test_vec)
lr_pred = lr_model.predict(X_test_vec)

print('================================================')
print('         EMAIL SPAM MODEL EVALUATION')
print('================================================')
print(f"Metric          | Complement NB | Logistic Reg")
print('------------------------------------------------')
print(f"Accuracy        | {accuracy_score(y_test, nb_pred)*100:.2f}%        | {accuracy_score(y_test, lr_pred)*100:.2f}%")
print(f"Precision       | {precision_score(y_test, nb_pred)*100:.2f}%        | {precision_score(y_test, lr_pred)*100:.2f}%")
print(f"Recall          | {recall_score(y_test, nb_pred)*100:.2f}%        | {recall_score(y_test, lr_pred)*100:.2f}%")
print(f"F1-Score        | {f1_score(y_test, nb_pred)*100:.2f}%        | {f1_score(y_test, lr_pred)*100:.2f}%")
print('================================================')

def test_message(text):
    vec = vectorizer.transform([text])
    
    nb_class = nb_model.predict(vec)[0]
    lr_class = lr_model.predict(vec)[0]

    nb_prob = nb_model.predict_proba(vec)[0][1] * 100
    lr_prob = lr_model.predict_proba(vec)[0][1] * 100

    nb_res = 'SPAM' if nb_class == 1 else 'HAM'
    lr_res = 'SPAM' if lr_class == 1 else 'HAM'

    print(f'\nText: "{text}"')
    print(f'-> Naive Bayes:        {nb_res} (Spam Score: {nb_prob:.1f}%)')
    print(f'-> Logistic Regression: {lr_res} (Spam Score: {lr_prob:.1f}%)\n')

while True:
    user_input = input('Enter a message to test (or Enter to quit): ')
    if not user_input.strip():
        break
    test_message(user_input)