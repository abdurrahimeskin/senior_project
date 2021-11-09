import numpy as np
import pandas as pd

u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('data/ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')

i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('data/ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('data/ml-100k/u.data', sep='\t', names=r_cols, encoding='latin-1')

n_users = ratings.user_id.unique().shape[0]
n_items = ratings.movie_id.unique().shape[0]

all_ratings = list(zip(ratings['user_id'], ratings['movie_id'], ratings['rating']))

n_factors = 10  # number of factors
alpha = .0005  # learning rate
n_epochs = 10  # number of iteration of the SGD procedure

p = np.random.normal(0, .1, (n_users, n_factors))
p = np.add(p,abs(np.min(p)))

q = np.random.normal(0, .1, (n_items, n_factors))
q = np.add(q,abs(np.min(q)))


def SGD():
    # Optimization procedure
    for _ in range(n_epochs):
        for u, i, r_ui in all_ratings:
            err = r_ui - np.dot(p[u - 1], q[i - 1])
            # Update vectors p_u and q_i
            p[u - 1] += alpha * err * q[i - 1]
            q[i - 1] += alpha * err * p[u - 1]

def estimate(u, i):
    '''Estimate rating of user u for item i.'''
    return np.dot(p[u], q[i])

SGD()

data_matrix_2 = np.zeros((n_users, n_items))

for i in range(n_users):
    for j in range(n_items):
        data_matrix_2[i-1,j-1] = estimate(i, j)

precision_list = []
recall_list = []

for index, user1_pred in enumerate(data_matrix_2):

    if (index == 10):
        break

    enum_user1_pred = list(enumerate(user1_pred, start=1))

    enum_user1_pred.sort(key=lambda tup: tup[1], reverse=True)

    print(enum_user1_pred[:20])

    rating_movie_ids = ratings['movie_id'].tolist()
    movie_ids = [i for i in enum_user1_pred if i[0] in rating_movie_ids]
    movie_ids.sort(key=lambda tup: tup[1], reverse=True)
    movie_ids_20 = movie_ids[:20]

    recommended_ids = [i[0] for i in movie_ids_20 if i[1] >= 3.5]
    ratings['id_rating_pair'] = ratings[['movie_id', 'rating']].apply(tuple, axis=1)
    list_tuples = ratings[ratings['user_id'] == (index + 1)]['id_rating_pair'].tolist()
    movie_ids_single = [i[0] for i in movie_ids_20]
    relevant_ids = [i[0] for i in list_tuples if (i[0] in movie_ids_single) & (i[1] >= 3.5)]

    recommended_relevant = list(set(relevant_ids).intersection(recommended_ids))

    if (len(recommended_ids) == 0):
        precision = 1
    else:
        precision = len(recommended_relevant) / len(recommended_ids)
    precision_list.append(precision)

    if (len(relevant_ids) == 0):
        recall = 1
    else:
        recall = len(recommended_relevant) / len(relevant_ids)
    recall_list.append(recall)

print(recall_list)
print(precision_list)

recall_average = sum(recall_list) / len(recall_list)
print(recall_average)

precision_average = sum(precision_list) / len(precision_list)
print(precision_average)

f1_score = 2 * recall_average * precision_average / (recall_average + precision_average)
print(f1_score)