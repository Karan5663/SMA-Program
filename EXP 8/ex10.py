import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Load data
file_path = "C:\\Users\\Karan\\OneDrive\\Desktop\\sma\\SMA_PROGRAMS\\EXP 8\\movie_reviews.csv"
df = pd.read_csv(file_path)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Preview the data
print(df.sample(3))
print(df.info())

# Check the value counts of the sentiment label
print(df["Sentiment"].value_counts(normalize=True))

# Generate the word cloud
reviews = " ".join(df["Review"])  # Use "Review" column
word_cloud = WordCloud(background_color="white",
                       stopwords=ENGLISH_STOP_WORDS,
                       width=800,
                       height=400).generate(reviews)

plt.figure(figsize=(12, 8))
plt.imshow(word_cloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Vectorize the text data
pattern = r"[a-zA-Z]+"
vect = TfidfVectorizer(token_pattern=pattern,
                        stop_words=ENGLISH_STOP_WORDS,
                        ngram_range=(1, 2),
                        max_features=50)
vect.fit(df["Review"])  # Use "Review" column
tokenized_features = vect.transform(df["Review"])

# Create features DataFrame
features = pd.DataFrame(data=tokenized_features.toarray(), columns=vect.get_feature_names_out())
features["char_count"] = df["Review"].str.count(r"\S")
features["word_count"] = df["Review"].str.count(pattern)
features["avg_word_length"] = features["char_count"] / features["word_count"]

# Define X and y
X = features
y = df["Sentiment"]  # Use "Sentiment" column

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train a random forest classifier
rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)

# Predict the labels
y_pred = rf.predict(X_test)

# Print classification metrics
print(classification_report(y_test, y_pred))

# Confusion matrix
ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test, normalize="all")
plt.title("Confusion Matrix")
plt.show()

# Feature importance
importance_df = pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_})
print(importance_df.sort_values(by="importance", ascending=False))
