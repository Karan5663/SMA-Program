# Importing necessary libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models import LdaModel
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download NLTK data (only required the first time you run it)
nltk.download('stopwords')
nltk.download('wordnet')

# Step 1: Load the scraped data
# Updated file path for your "Exp 5" directory
df = pd.read_csv(r'C:\Users\Karan\OneDrive\Desktop\sma\SMA_PROGRAMS\Exp 5\google.csv')  # Update to your file path

# Step 2: Preprocessing function for text cleaning
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Clean the review text by removing punctuation, lowering the text, and applying lemmatization."""
    # Remove punctuation, special characters, and convert to lowercase
    text = re.sub(r'\W', ' ', text.lower())  # \W removes anything that is not a word character
    # Tokenize the text into individual words
    tokens = text.split()
    # Remove stopwords and apply lemmatization (converting words to their base form)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing to the reviews
# Ensure 'd4r55' is the correct column name with your reviews data
df['cleaned_reviews'] = df['d4r55'].apply(preprocess_text)

# Step 3: Tokenizing the cleaned reviews
tokenized_reviews = [review.split() for review in df['cleaned_reviews']]

# Step 4: Create a dictionary (vocabulary) and bag-of-words representation of the reviews
dictionary = corpora.Dictionary(tokenized_reviews)
bow_corpus = [dictionary.doc2bow(review) for review in tokenized_reviews]

# Step 5: Apply LDA for topic modeling
# Choosing the number of topics (adjust this as needed)
num_topics = 5
lda_model = LdaModel(bow_corpus, num_topics=num_topics, id2word=dictionary, passes=15)

# Step 6: Display the topics
print("Topics discovered by LDA model:")
for idx, topic in lda_model.print_topics(-1):
    print(f'Topic {idx}: {topic}')

# Step 7: Visualizing Topics with Word Cloud
for i, topic in lda_model.show_topics(formatted=False, num_words=10):
    words = dict(topic)
    wordcloud = WordCloud(width=400, height=400, background_color='white').generate_from_frequencies(words)

    # Plot the Word Cloud
    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Topic {i}")
    plt.show()

# Step 8: Get the topic distribution for each review
print("\nTopic Distribution for each Review:")
for index, review in enumerate(bow_corpus):
    topic_distribution = lda_model.get_document_topics(review)
    print(f"Review {index}: {topic_distribution}")

# Optional: Saving the results to a CSV for further analysis
df['topic_distribution'] = [lda_model.get_document_topics(bow) for bow in bow_corpus]

# Save the output file to the same directory as the input file
df.to_csv(r'C:\Users\sujal\Desktop\SMA_PROGRAMS\Exp 5\dominos_reviews_with_topics_output.csv', index=False)

