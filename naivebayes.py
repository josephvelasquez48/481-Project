import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / float(n_samples)

    def _gaussian_density(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _predict_single(self, x):
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            likelihood = np.sum(np.log(self._gaussian_density(idx, x)))
            posterior = prior + likelihood
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])


# Load the dataset (use error handling for missing file or sheet)
try:
    data = pd.read_excel("LargeDataSet.xlsx", sheet_name="mail_data")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Access the columns
emails = data['email'].values
labels = data['label'].values

# Convert labels to numerical format (0: ham, 1: spam)
label_number = np.where(labels == 'spam', 1, 0)

# Vectorize the email text
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(emails).toarray()
y = label_number

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Train the Naive Bayes classifier
classifier = NaiveBayes()
classifier.fit(x_train, y_train)

# Test the classifier
predictions = classifier.predict(x_test)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classify new emails
new_emails = ["Congratulations! You won a prize!", "Please review the attached report."]
if new_emails:
    new_x = vectorizer.transform(new_emails).toarray()
    new_predictions = classifier.predict(new_x)

    # Display predictions
    for email, label in zip(new_emails, new_predictions):
        print(f"Email: {email} -> {'Spam' if label == 1 else 'Not Spam'}")
else:
    print("No new emails to classify.")
