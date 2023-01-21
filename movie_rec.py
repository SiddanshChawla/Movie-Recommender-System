import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
#from google.colab import drive
#drive.mount('/content/drive/')
def main():
    #movies = pd.read_csv('/content/drive/My Drive/movie_data/movies.csv')
    movies = pd.read_csv("~/Downloads/ml-latest-small/movies.csv")
        #df = pd.read_csv("~/Desktop/valenceArousalDataset.csv")

    movies = movies.join(movies.pop('genres').str.get_dummies('|'))

    genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    feature_matrix = movies[genres].values

    st.title("Movie recommender system")

    movie_name = movies['title'].tolist()
    target_movie = st.selectbox("Enter the movie name", movie_name)

    movie_similarity = cosine_similarity(feature_matrix)

    target_movie_index = movies[movies['title'] == target_movie].index[0]

    most_similar_movies = movie_similarity[target_movie_index].argsort()[::-1][1:]

    most_similar_movie_titles = movies.iloc[most_similar_movies]['title'].values

    # Print the most similar movies
    final_result = pd.DataFrame(most_similar_movie_titles, columns = ['Title'])
    st.write(final_result.sample(n=5))

if __name__ == '__main__':
    main()

