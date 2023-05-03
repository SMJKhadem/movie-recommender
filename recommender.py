from utils import col_mean, MOVIES, model
import pandas as pd
import numpy as np

# query={
#     'The Lord of the Rings': 5,
#     'Avatar':2,
#     'Titanic': 3.5
# }

def recommend_nmf(query, model=model, k=10):
    """
    Filters and recommends the top k movies for any given input query based on a trained NMF model. 
    Returns a list of k movie ids.
    """
    
    
    recommendations = []
    # 1. candidate generation
    user_initial_ratings=query

    # 2. construct new_user-item dataframe given the query
    
    user_input=pd.DataFrame(user_initial_ratings, index=['new user'], columns=MOVIES)
    
    # 3. scoring
    
    user_input_imputed = user_input.fillna(value=col_mean)
    
    # calculate the score with the NMF model
    P_user =model.transform(user_input_imputed)
    P_user=pd.DataFrame(P_user, index=['new_user'])
    Q=model.components_
    R_user_hat = np.dot(P_user, Q) 
    R_user_hat=pd.DataFrame(R_user_hat, index=['new_user'], columns=MOVIES)
    
    # 4. ranking
    
    R_user_hat_transposed=R_user_hat.T.sort_values(by='new_user', ascending=False)
    
    Recommendables = list(R_user_hat_transposed.index)
    user_initial_rating_list = list(user_initial_ratings.keys())
                                                   
    # filter out movies already seen by the user and 
     # return the top-k highest rated movie ids or titles
    
    recommendations= [movie for movie in Recommendables if movie not in user_initial_rating_list ]
   
    
    #print(type(Recommendables))
    #print(recommendations)
    return recommendations
#recommend_nmf(query, model=model)





