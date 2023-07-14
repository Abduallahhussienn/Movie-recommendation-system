#!/usr/bin/env python
# coding: utf-8

# In[153]:


import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
import joblib

# In[154]:


reader = Reader()
ratings = pd.read_csv('C:\\Users\\Abduallah Hussien\\Desktop\\R&A project\\ratings_small.csv')
movie_md = pd.read_csv('C:\\Users\\Abduallah Hussien\\Desktop\\R&A project\\movies_metadata.csv')

# In[143]:


data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)


# In[144]:


# Use the famous SVD algorithm
#algo = SVD(random_state=42)

# Run 5-fold cross-validation and then print results
#cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


# In[145]:


#trainset = data.build_full_trainset()
#algo.fit(trainset)


# In[146]:




# In[147]:


#algo.predict(1, 302)


# In[148]:


# movie dataframe with votes more than 20
movie_md = movie_md[movie_md['vote_count']>20][['id','title']]

# IDs of movies with count more than 20
movie_ids = [int(x) for x in movie_md['id'].values]

# Select ratings of movies with more than 20 counts
ratings = ratings[ratings['movieId'].isin(movie_ids)]

# Reset Index
ratings.reset_index(inplace=True, drop=True)


# In[149]:


# In[150]:

loaded_model = joblib.load(open('recommendation_model','rb'))

def get_recommendations(data, movie_md, user_id, top_n, algo):
    
    # creating an empty list to store the recommended product ids
    recommendations = []
    r = []
    # creating an user item interactions matrix 
    user_movie_interactions_matrix = data.pivot(index='userId', columns='movieId', values='rating')
    # extracting those product ids which the user_id has not interacted yet
    non_interacted_movies = user_movie_interactions_matrix.loc[user_id][user_movie_interactions_matrix.loc[user_id].isnull()].index.tolist()
    
    # looping through each of the product ids which user_id has not interacted yet
    for item_id in non_interacted_movies:
        
        # predicting the ratings for those non interacted product ids by this user
        est = loaded_model.predict(user_id, item_id).est
        
        # appending the predicted ratings
        movie_name = movie_md[movie_md['id']==str(item_id)]['title'].values[0]
        recommendations.append((movie_name, est))
    # sorting the predicted ratings in descending order
    recommendations.sort(key=lambda x: x[1], reverse=True)
    for i in recommendations[:top_n]:
        r.append(i[0])
    return r # returing top n highest predicted rating products for this user


# In[151]:


#uid = int(input("Enter user id : "))

# import module
import streamlit as st

st.set_page_config(
    page_title="Home Page",
)



# Title
from PIL import Image
img = Image.open("streamlit.png")
 
# display image using streamlit
# width is used to set the width of an image
st.columns(3)[1].image(img, width=200)
st.markdown("<h1 style='text-align: center; color: white;'>Movie Recommendation System</h1>", unsafe_allow_html=True)

#st.title("Recommendation System")

id = st.selectbox("Choose user ID",ratings.userId.unique())
# In[ ]:

# Create a button, that when clicked, shows a text
if(st.button("Recommendation")):
    st.markdown("### Recommended movies")
    for i in get_recommendations(data=ratings,movie_md=movie_md, user_id=int(id), top_n=5, algo=loaded_model):
        st.write(i)
    


