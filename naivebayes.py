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
        """
        Fit the Naive Bayes classifier to the data.
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Initialize means, variances, and priors
        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / float(n_samples)
    
    def _gaussian_density(self, class_idx, x):
        """
        Calculate the Gaussian likelihood of the data for a given class.
        """
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
    def _predict_single(self, x):
        """
        Predict the class label for a single sample.
        """
        posteriors = []
        
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            likelihood = np.sum(np.log(self._gaussian_density(idx, x)))
            posterior = prior + likelihood
            posteriors.append(posterior)
        
        return self.classes[np.argmax(posteriors)]
    
    def predict(self, X):
        """
        Predict the class labels for the given data.
        """
        return np.array([self._predict_single(x) for x in X])

#load the dataset. using only sheet name = emails from LargeDataSet.xlsx
data = pd.read_excel("LargeDataSet.xlsx", sheet_name="emails") 

#access the columns
emails = data['email'].values  
labels = data['label'].values

#vectorize the email text
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(emails).toarray()
y = np.array(labels)

#split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#train the Naive Bayes classifier
classifier = NaiveBayes()
classifier.fit(x_train, y_train)

#test the classifier
predictions = classifier.predict(x_test)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

#classify new emails. need to add email text.
new_emails = []
new_x = vectorizer.transform(new_emails).toarray()
new_predictions = classifier.predict(new_x)

#display predictions
for email, label in zip(new_emails, new_predictions):
    print(f"Email: {email} -> {'Spam' if label == 1 else 'Not Spam'}")

