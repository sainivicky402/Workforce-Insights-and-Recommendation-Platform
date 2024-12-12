import streamlit as st
import pandas as pd
import plotly.express as px

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("job_posting_location.csv", parse_dates=["published_date"])

# Load and preprocess data
data = load_data()
data["published_date"] = data["published_date"].dt.tz_localize(None)
data["YearMonth"] = data["published_date"].dt.to_period("M")

# Streamlit Dashboard Title
st.title("Job Market Dynamics Dashboard")
st.markdown("Visualize trends in job postings, roles, and salaries over time.")

# Sidebar Filters
st.sidebar.header("Filter Options")
selected_category = st.sidebar.multiselect("Select Categories", options=data["Category"].unique(), default=data["Category"].unique())
selected_countries = st.sidebar.multiselect("Select Countries", options=data["country"].unique(), default=data["country"].unique())
selected_date_range = st.sidebar.date_input("Select Date Range", [data["published_date"].min().date(), data["published_date"].max().date()])

# Filter data based on selections
filtered_data = data[
    (data["Category"].isin(selected_category)) &
    (data["country"].isin(selected_countries)) &
    (data["published_date"].between(selected_date_range[0], selected_date_range[1]))
]

# Key Insights Section
st.subheader("Key Insights")
total_jobs = len(filtered_data)
average_salary = filtered_data["average_hourly_rate"].mean() if total_jobs > 0 else 0
top_category = filtered_data["Category"].value_counts().idxmax() if total_jobs > 0 else "N/A"

st.metric("Total Job Postings", f"{total_jobs:,}")
st.metric("Average Hourly Rate", f"${average_salary:.2f}")
st.metric("Most Popular Category", top_category)

# Trend Analysis Section
st.subheader("Trends Over Time")
job_trend = filtered_data.groupby("YearMonth").size().reset_index(name="Job Postings")
salary_trend = filtered_data.groupby("YearMonth")["average_hourly_rate"].mean().reset_index()

# Plotting trends
st.plotly_chart(px.line(job_trend, x="YearMonth", y="Job Postings", title="Job Posting Trends Over Time"))
st.plotly_chart(px.line(salary_trend, x="YearMonth", y="average_hourly_rate", title="Average Hourly Rate Trends"))

# Geographic Analysis
st.subheader("Geographic Analysis")
geo_avg_salary = filtered_data.groupby("country")["average_hourly_rate"].mean().reset_index()
st.plotly_chart(px.choropleth(geo_avg_salary, locations="country", locationmode="country names", color="average_hourly_rate", title="Average Hourly Rate by Country", color_continuous_scale="Viridis"))

# User Interaction: Download Filtered Data
st.subheader("Download Filtered Data")
csv = filtered_data.to_csv(index=False).encode("utf-8")
st.download_button(label="Download Filtered Data as CSV", data=csv, file_name="filtered_job_data.csv", mime="text/csv")

st.markdown("---")
st.caption("Dashboard powered by Streamlit and Plotly.")