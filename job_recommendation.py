import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Set MLflow experiment
mlflow.set_experiment("Job Recommendation System")

def preprocess_data(filepath):
    data = pd.read_csv(filepath)
    return data.dropna(subset=["job_description"])

def train_tfidf_model(data, save_path=None):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(data["job_description"])
    
    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(vectorizer, f)
    
    return vectorizer, tfidf_matrix

def recommend_jobs(user_input, tfidf_matrix, vectorizer, data):
    user_query_tfidf = vectorizer.transform([user_input])
    cosine_sim = cosine_similarity(user_query_tfidf, tfidf_matrix).flatten()
    data["similarity_score"] = cosine_sim
    return data.nlargest(5, "similarity_score")[["Cleaned Job Title", "Category", "country", "average_hourly_rate", "link"]]

def track_with_mlflow(data, vectorizer):
    with mlflow.start_run():
        mlflow.log_param("Number of Job Listings", len(data))
        mlflow.sklearn.log_model(vectorizer, "TFIDF_Vectorizer")
        print("Experiment logged successfully!")

if __name__ == "__main__":
    # Data paths
    data_path = "job_posting_location.csv"
    model_save_path = "tfidf_vectorizer.pkl"

    # Preprocess data
    data = preprocess_data(data_path)
    print(f"Data loaded: {data.shape[0]} rows.")
    
    # Train TF-IDF
    vectorizer, tfidf_matrix = train_tfidf_model(data, save_path=model_save_path)
    print("TF-IDF model trained and saved.")
    
    # Log experiment
    track_with_mlflow(data, vectorizer)
    
    # Test recommendation
    test_input = "data scientist"
    recommendations = recommend_jobs(test_input, tfidf_matrix, vectorizer, data)
    print("Top Recommendations:")
    print(recommendations)