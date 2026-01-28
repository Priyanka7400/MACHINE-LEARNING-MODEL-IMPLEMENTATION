import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv(
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
    sep="\t",
    header=None,
    names=["label", "message"]
)

print("Dataset Preview:")
print(data.head())


data['label'] = data['label'].map({'ham': 0, 'spam': 1})

X = data['message']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


model = MultinomialNB()
model.fit(X_train_vec, y_train)


y_pred = model.predict(X_test_vec)


print("\nModel Accuracy:")
print(accuracy_score(y_test, y_pred))


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


test_message = ["Congratulations! You have won a free prize. Click now!"]
test_message_vec = vectorizer.transform(test_message)
prediction = model.predict(test_message_vec)

print("\nCustom Message Prediction:")
print("Spam" if prediction[0] == 1 else "Not Spam")
