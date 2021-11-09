import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from math import sqrt

u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('data/ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('data/ml-100k/u.data', sep='\t', names=r_cols, encoding='latin-1')

i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('data/ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')

n_users = ratings.user_id.unique().shape[0]
n_items = ratings.movie_id.unique().shape[0]

data_matrix = np.zeros((n_users, n_items))

for line in ratings.itertuples():
    data_matrix[line[1]-1, line[2]-1] = line[3]

user_similarity = np.corrcoef(data_matrix)
item_similarity = 1 - pairwise_distances(data_matrix.T, metric='cosine')

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) /\
               np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

user_prediction = predict(data_matrix, user_similarity, type='user')
item_prediction = predict(data_matrix, item_similarity, type='item')

def rmse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return sqrt(np.square(np.subtract(pred, actual)).mean())

def mae(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return abs(np.subtract(pred, actual)).mean()

def mse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return np.square(np.subtract(pred, actual)).mean()

print('User-based CF MSE: ' + str(mse(user_prediction, data_matrix)))
print('Item-based CF MSE: ' + str(mse(item_prediction, data_matrix)))

print('User-based CF RMSE: ' + str(rmse(user_prediction, data_matrix)))
print('Item-based CF RMSE: ' + str(rmse(item_prediction, data_matrix)))

print('User-based CF MAE: ' + str(mae(user_prediction, data_matrix)))
print('Item-based CF MAE: ' + str(mae(item_prediction, data_matrix)))

precision_list = []
recall_list = []

for index, user1_pred in enumerate(user_prediction):

    enum_user1_pred = list(enumerate(user1_pred, start=1))

    enum_user1_pred.sort(key=lambda tup: tup[1], reverse=True)

    print(enum_user1_pred[:20])

    rating_movie_ids = ratings['movie_id'].tolist()
    movie_ids = [i for i in enum_user1_pred if i[0] in rating_movie_ids]
    movie_ids_20 = movie_ids[:20]

    recommended_ids = [i[0] for i in movie_ids_20 if i[1] >= 2.5]
    ratings['id_rating_pair'] = ratings[['movie_id', 'rating']].apply(tuple, axis=1)
    list_tuples = ratings[ratings['user_id'] == (index + 1)]['id_rating_pair'].tolist()
    movie_ids_single = [i[0] for i in movie_ids_20]
    relevant_ids = [i[0] for i in list_tuples if (i[0] in movie_ids_single) & (i[1] >= 2.5)]

    recommended_relevant = list(set(relevant_ids).intersection(recommended_ids))

    if(len(recommended_ids) == 0):
        precision = 1
    else:
        precision = len(recommended_relevant) / len(recommended_ids)
    precision_list.append(precision)

    if(len(relevant_ids) == 0):
        recall = 1
    else:
        recall = len(recommended_relevant) / len(relevant_ids)
    recall_list.append(recall)

print(recall_list)
print(precision_list)