import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics.pairwise import cosine_similarity 

books = pd.read_csv("./data/Books.csv")
users = pd.read_csv("./data/Users.csv")
ratings = pd.read_csv("./data/Ratings.csv")

ratings_with_name = ratings.merge(books,on='ISBN')
ratings_with_name.drop(columns=["ISBN","Image-URL-S","Image-URL-M"],axis=1,inplace=True)
complete_df = ratings_with_name.merge(users.drop("Age", axis=1), on="User-ID")
complete_df['Location'] = complete_df['Location'].str.split(',').str[-1].str.strip()
num_rating_df = complete_df.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)
avg_rating_df = complete_df.groupby('Book-Title').mean()['Book-Rating'].reset_index()
avg_rating_df.rename(columns={'Book-Rating': 'avg_ratings'}, inplace=True)
popularity_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
top_50 = popularity_df[popularity_df['num_ratings']>=250].sort_values("avg_ratings",ascending=False)
final_top_50 = top_50.merge(books, on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Publisher','num_ratings','avg_ratings','Image-URL-M']].reset_index(drop=True)

# Collaborative filtering
x = complete_df.groupby('User-ID').count()['Book-Rating']>200
knowledgable_users = x[x].index
filtered_rating = complete_df[complete_df['User-ID'].isin(knowledgable_users)]

y = filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50
famous_books = y[y].index

final_ratings =  filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
pt = final_ratings.pivot_table(index='Book-Title',columns='User-ID'
                          ,values='Book-Rating')
pt.fillna(0,inplace=True)

similarity_score = cosine_similarity(pt)

def recommend(book_name):
    index = np.where(pt.index==book_name)[0][0]
    similar_books = sorted(list(enumerate(similarity_score[index])),key=lambda x:x[1], reverse=True)[1:6]
    
    data = []
    
    for i in similar_books:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        
        data.append(item)
    return data

# Streamlit App
import streamlit as st

st.title('Book Recommendation System')
st.text_input("Enter the name of the book here..", key="book_name")

if len(st.session_state.book_name) == 0:
    st.subheader("Our Books")
    st.write(final_top_50)

if len(st.session_state.book_name) != 0:
    params = recommend(st.session_state.book_name)
    
    for i in range(len(params)):
        url = params[i][2]
        book_name = "<h2 style='text-align: left; color: grey;'> 1. " + params[i][0] + "</h2>"
        st.markdown(book_name, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(url)

        author = "<h3 style='text-align: left; color: grey;'> Author: " + params[i][1] + "</h3>"
        st.markdown(author, unsafe_allow_html=True)
        st.write("")
    