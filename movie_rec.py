import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
#from google.colab import drive
#drive.mount('/content/drive/')

uel = "https://htmlcolorcodes.com/assets/images/colors/sky-blue-color-solid-background-1920x1080.png"
url = ("https://preview.redd.it/4fxxbm4opjd31.jpg?auto=webp&s=f5b7d62076600a978d290a5e87f13140c47f5cd0")

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://preview.redd.it/4fxxbm4opjd31.jpg?auto=webp&s=f5b7d62076600a978d290a5e87f13140c47f5cd0");
             background-attachment: fixed;
             background-size: full
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

def main():
    st.write('<style>div.block-container{text-align: center; background-color: #87becb; border-radius: 4rem; padding-top: 2rem; padding-bottom: 2rem; opacity:0.98; margin-top:6rem;}</style>', unsafe_allow_html=True)
#    with open('style.css') as f:
#        st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)

    #movies = pd.read_csv('/content/drive/My Drive/movie_data/movies.csv')
    movies = pd.read_csv("~/Downloads/ml-latest-small/movies.csv")
        #df = pd.read_csv("~/Desktop/valenceArousalDataset.csv")

    movies = movies.join(movies.pop('genres').str.get_dummies('|'))

    genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    feature_matrix = movies[genres].values

    st.markdown('<h1 style="font-family:sans-serif; color:black; font-size: 2rem;">SimilarCinema.com</h1>', unsafe_allow_html=True)

    movie_name = movies['title'].tolist()
#    movie_name.insert(0, "Choose your movie")
    
    target_movie = st.selectbox("Enter the movie name", movie_name)
  
    movie_similarity = cosine_similarity(feature_matrix)

    target_movie_index = movies[movies['title'] == target_movie].index[0]

    most_similar_movies = movie_similarity[target_movie_index].argsort()[::-1][1:]

    most_similar_movie_titles = movies.iloc[most_similar_movies]['title'].values

    # Print the most similar movies
    final_result = pd.DataFrame(most_similar_movie_titles, columns = ['Title'])
    
    # CSS to inject contained in a string
    hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

# Inject CSS with Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)

    st.table(final_result.sample(n=5))

#    from streamlit_card import card
#
#    hasClicked = card(
#        title='',
#        text="Click to generate new suggestions",
#    )
    st.button('Generate new suggestions')

if __name__ == '__main__':
    add_bg_from_url()
    main()
    

