import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import requests
import webbrowser
import io

st.set_page_config(page_title="Movie Recommendation System",layout="wide")

css = """
<style>
    /* CSS styles go here */
    .title {
        color: #00FFFF;
    }
    
</style>
"""
st.markdown(css, unsafe_allow_html=True)


new_df = pd.read_csv(r"C:\Users\divya.LAPTOP-0B1GN7G7\PycharmProjects\Movie Recommender\processed_data.csv")

# Load the similarity matrix from the file
with open('similarity_matrix.pkl', 'rb') as file:
    similarity = pickle.load(file)

df = pd.read_csv(r"C:\Users\divya.LAPTOP-0B1GN7G7\PycharmProjects\Movie Recommender\processed_data.csv")
movies_list= df['title_x'].tolist()


def get_movie_poster_url(movie_title):
    api_key = "150164a23d0b5281b1fb907564e97c03"
    base_url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": api_key,
        "query": movie_title
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    if "results" in data and len(data["results"]) > 0:
        poster_path = data["results"][0].get("poster_path")
        if poster_path:
            poster_url = f"https://image.tmdb.org/t/p/w500/{poster_path}"
            return poster_url
    return None







# st.title('Movie Recommendation System')
st.markdown('<h1 class="title">Movie Recommendation System</h1>', unsafe_allow_html=True)

options = movies_list
selected_movie = st.selectbox('',options)

button_clicked = st.button('Recommend')
if button_clicked:
    st.write('Movies Similar to:', selected_movie)

    def recommend(movie):
        movie_index = new_df[new_df['title_x'] == movie].index[0]
        distances = list(enumerate(similarity[movie_index]))
        movies_list = sorted(distances, reverse=True, key=lambda x: x[1])[1:6]

        recommended_movies = []
        for i in movies_list:
            recommended_movies.append(new_df.iloc[i[0]].title_x)

        return recommended_movies


    recommended_movies = recommend(selected_movie)

    movie_title_1 = recommended_movies[0]
    movie_title_2 = recommended_movies[1]
    movie_title_3 = recommended_movies[2]
    movie_title_4 = recommended_movies[3]
    movie_title_5 = recommended_movies[4]

    poster_url_1 = get_movie_poster_url(movie_title_1)
    poster_url_2 = get_movie_poster_url(movie_title_2)
    poster_url_3 = get_movie_poster_url(movie_title_3)
    poster_url_4 = get_movie_poster_url(movie_title_4)
    poster_url_5 = get_movie_poster_url(movie_title_5)

    print(poster_url_1)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.header(recommended_movies[0])
        st.image(poster_url_1 )

    with col2:
        st.header(recommended_movies[1])
        st.image(poster_url_2 )

    with col3:
        st.header(recommended_movies[2])
        st.image(poster_url_3 )

    with col4:
        st.header(recommended_movies[3])
        st.image(poster_url_4 )

    with col5:
        st.header(recommended_movies[4])
        st.image(poster_url_5 )
