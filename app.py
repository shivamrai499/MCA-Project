import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import streamlit as st
import os

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# CSV file path
file_path = 'flipkart_data.csv'

# Preprocessing function
def preprocess_reviews_stopwords(df):
    df['review'] = df['review'].astype(str).str.lower()
    df['review'] = df['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    df['sentiment'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)
    return df

# Load and preprocess data
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    df_cleaned = preprocess_reviews_stopwords(df)
else:
    df_cleaned = pd.DataFrame(columns=['review', 'rating', 'sentiment'])

# Initialize variables for modeling
model = None
accuracy = None
conf_matrix = None
labels = None

# Train model only if enough data
if not df_cleaned.empty and len(df_cleaned) > 5:
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df_cleaned['review'])
    y = df_cleaned['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    labels = model.classes_
    conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)

# --- Streamlit UI ---
st.set_page_config(page_title="Sentiment App", layout="wide")
st.title("🛒 Sentiment Product Analysis Review System")



st.markdown(f"**📊 Total Reviews in Dataset:** {len(df_cleaned)}")

# Show full dataset
st.subheader("📄 Full Review Dataset")
st.dataframe(df_cleaned[['review', 'rating', 'sentiment']])

if not df_cleaned.empty and model is not None:
    st.subheader("Sentiment Distribution")
    sentiment_counts = df_cleaned['sentiment'].value_counts().sort_index()
    sentiment_labels = ['Negative', 'Positive']
    fig, ax = plt.subplots()
    sns.barplot(x=sentiment_labels, y=sentiment_counts.values, palette='coolwarm', ax=ax)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.subheader("Word Cloud for Positive Reviews")
    positive_text = ' '.join(df_cleaned[df_cleaned['sentiment'] == 1]['review'])
    wordcloud = WordCloud(width=800, height=400).generate(positive_text)
    fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis('off')
    st.pyplot(fig_wc)

    st.subheader("Model Accuracy and Confusion Matrix")
    st.write(f"**Model Accuracy:** {accuracy:.2f}")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)
else:
    st.warning("⚠️ Not enough data to train the model. Please add more reviews.")

# Add new review section
st.subheader("📩 Add a New Review")
with st.form("review_form", clear_on_submit=True):
    review_text = st.text_area("Enter your review:")
    review_rating = st.slider("Select Rating (1 to 5):", 1, 5, 3)
    submitted = st.form_submit_button("Add Review")

    if submitted and review_text.strip() != "":
        new_data = pd.DataFrame({
            'review': [review_text],
            'rating': [review_rating]
        })
        new_data = preprocess_reviews_stopwords(new_data)
        new_data.to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path))
        st.session_state["review_added"] = True
        st.rerun()
# ✅ Show success message if review was just added
if st.session_state.get("review_added", False):
    st.success("✅ Review added successfully.")
    st.session_state["review_added"] = False
st.markdown("---")
st.caption("Created with ❤️ using Streamlit")
# ========updated===============

# import pandas as pd
# import nltk
# import matplotlib.pyplot as plt
# import seaborn as sns
# from wordcloud import WordCloud
# from nltk.corpus import stopwords
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix
# import streamlit as st
# import os

# # Download NLTK stopwords
# nltk.download('stopwords')
# stop_words = set(stopwords.words('english'))

# # CSV file path
# file_path = 'flipkart_data.csv'

# # Preprocessing function
# def preprocess_reviews_stopwords(df):
#     df['review'] = df['review'].astype(str).str.lower()
#     df['review'] = df['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
#     df['sentiment'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)
#     return df

# # Function to train model and return all necessary outputs
# def train_model(df_cleaned):
#     tfidf = TfidfVectorizer()
#     X = tfidf.fit_transform(df_cleaned['review'])
#     y = df_cleaned['sentiment']

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, stratify=y, test_size=0.2, random_state=42
#     )

#     model = DecisionTreeClassifier(random_state=42)
#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     labels = model.classes_
#     conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)

#     return model, accuracy, conf_matrix, labels, tfidf

# # --- Load and preprocess data ---
# def load_data():
#     if os.path.exists(file_path):
#         df = pd.read_csv(file_path)
#         df_cleaned = preprocess_reviews_stopwords(df)
#     else:
#         df_cleaned = pd.DataFrame(columns=['review', 'rating', 'sentiment'])
#     return df_cleaned

# df_cleaned = load_data()

# # Train model only if enough data
# model = None
# accuracy = None
# conf_matrix = None
# labels = None
# tfidf = None

# if not df_cleaned.empty and len(df_cleaned) > 5:
#     model, accuracy, conf_matrix, labels, tfidf = train_model(df_cleaned)

# # --- Streamlit UI ---
# st.set_page_config(page_title="Sentiment App", layout="wide")
# st.title("🛒 Sentiment Product Analysis Review System")

# st.markdown(f"**📊 Total Reviews in Dataset:** {len(df_cleaned)}")

# # Show full dataset
# st.subheader("📄 Full Review Dataset")
# st.dataframe(df_cleaned[['review', 'rating', 'sentiment']])

# if not df_cleaned.empty and model is not None:
#     st.subheader("Sentiment Distribution")
#     sentiment_counts = df_cleaned['sentiment'].value_counts().sort_index()
#     sentiment_labels = ['Negative', 'Positive']
#     fig, ax = plt.subplots()
#     sns.barplot(x=sentiment_labels, y=sentiment_counts.values, palette='coolwarm', ax=ax)
#     ax.set_xlabel("Sentiment")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

#     st.subheader("Word Cloud for Positive Reviews")
#     positive_text = ' '.join(df_cleaned[df_cleaned['sentiment'] == 1]['review'])
#     wordcloud = WordCloud(width=800, height=400).generate(positive_text)
#     fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
#     ax_wc.imshow(wordcloud, interpolation='bilinear')
#     ax_wc.axis('off')
#     st.pyplot(fig_wc)

#     st.subheader("Model Accuracy and Confusion Matrix")
#     st.write(f"**Model Accuracy:** {accuracy:.2f}")
#     fig_cm, ax_cm = plt.subplots()
#     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues",
#                 xticklabels=labels, yticklabels=labels, ax=ax_cm)
#     ax_cm.set_xlabel("Predicted")
#     ax_cm.set_ylabel("Actual")
#     st.pyplot(fig_cm)
# else:
#     st.warning("⚠️ Not enough data to train the model. Please add more reviews.")

# # Add new review section
# st.subheader("📩 Add a New Review")
# with st.form("review_form", clear_on_submit=True):
#     review_text = st.text_area("Enter your review:")
#     review_rating = st.slider("Select Rating (1 to 5):", 1, 5, 3)
#     submitted = st.form_submit_button("Add Review")

#     if submitted and review_text.strip() != "":
#         # Append new review to CSV
#         new_data = pd.DataFrame({
#             'review': [review_text],
#             'rating': [review_rating]
#         })
#         new_data = preprocess_reviews_stopwords(new_data)
        
#         # Append without header after first write
#         new_data.to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path))
        
#         st.success("✅ Review added successfully.")
#         st.experimental_rerun()

# st.markdown("---")
# st.caption("Created with ❤️ using Streamlit")
