import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# File paths
DATA_PATH = "job_postings_location.csv"
MODEL_PATH = "tfidf_vectorizer.pkl"

# Load data and model
@st.cache_data
def load_data(filepath):
    data = pd.read_csv(filepath)
    data.dropna(subset=["job_description"], inplace=True)
    return data

@st.cache_resource
def load_model(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

# Recommend Jobs
def recommend_jobs(user_input, tfidf_matrix, vectorizer, data):
    user_query_tfidf = vectorizer.transform([user_input])
    cosine_sim = cosine_similarity(user_query_tfidf, tfidf_matrix).flatten()
    data["similarity_score"] = cosine_sim
    return data.nlargest(5, "similarity_score")[["Cleaned Job Title", "Category", "country", "average_hourly_rate", "link"]]

# Streamlit interface
def main():
    st.title("Job Recommendation System")
    user_input = st.text_input("Enter job title/description:")

    if user_input:
        data = load_data(DATA_PATH)
        vectorizer = load_model(MODEL_PATH)
        tfidf_matrix = vectorizer.transform(data["job_description"])
        recommendations = recommend_jobs(user_input, tfidf_matrix, vectorizer, data)

        if not recommendations.empty:
            st.write("**Top Recommendations:**")
            for _, row in recommendations.iterrows():
                st.markdown(
                    f"**{row['Cleaned Job Title']}**  \n"
                    f"Category: {row['Category']}  \n"
                    f"Location: {row['country']}  \n"
                    f"Hourly Rate: ${row['average_hourly_rate']}  \n"
                    f"[Job Link]({row['link']})  \n"
                    "---"
                )
        else:
            st.write("No recommendations found.")
    else:
        st.write("Please enter a job title or description.")

if __name__ == "__main__":
    main()
    
