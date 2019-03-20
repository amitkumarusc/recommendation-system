#!/usr/bin/python
# coding: utf-8

# New optimal hyperparameters
# model        <__main__.AlternatingLeastSquare object at 0x1...
# n_factors                                                   40
# n_iter                                                      95
# reg                                                        0.1
# test_mse                                               10.0342
# train_mse                                              4.72275
# dtype: object
# Regularization: 1.0
# Regularization: 10.0
# Regularization: 100.0
# Factors: 80
# Regularization: 0.01
# Regularization: 0.1
# Regularization: 1.0
# Regularization: 10.0
# Regularization: 100.0

import pandas as pd
import numpy as np
import random as rn
import time
np.random.seed(0)
import requests

class AlternatingLeastSquare(object):
    USERS_MATRIX = 'users_matrix'
    MOVIES_MATRIX = 'movies_matrix'
    
    def __init__(self, characteristic_matrix, latent_feature_count=50, users_reg=0.0, movies_reg=0.0):
        self.characteristic_matrix = characteristic_matrix
        self.latent_feature_count = latent_feature_count
        self.users_reg = users_reg
        self.movies_reg = movies_reg
        self.total_users, self.total_movies = self.characteristic_matrix.shape
    
    @staticmethod
    def calculate_mean_squared_error(predicted, actual):
        # mse = (np.square(predicted - actual)).mean(axis=ax)
        predicted = predicted[actual.nonzero()].flatten()
        actual = actual[actual.nonzero()].flatten()
        return ((predicted - actual) ** 2).mean()


    def train(self, iters=50):
        self.users_matrix = np.random.random((self.total_users, self.latent_feature_count))
        self.movies_matrix = np.random.random((self.total_movies, self.latent_feature_count))
        self.train_continue(iters)
    
    def train_continue(self, iters=5):
        for i in range(iters):
            self.users_matrix = self.train_step(self.users_matrix, self.movies_matrix, AlternatingLeastSquare.USERS_MATRIX)
            self.movies_matrix = self.train_step(self.movies_matrix, self.users_matrix, AlternatingLeastSquare.MOVIES_MATRIX)
            
    def train_step(self, variable_matrix, constant_matrix, chance):
        
        if chance == AlternatingLeastSquare.USERS_MATRIX:
            y_dot_y = constant_matrix.T.dot(constant_matrix)
            lambda_i = np.identity(y_dot_y.shape[0]) * self.users_reg
            
            for user_index in range(variable_matrix.shape[0]):
                variable_matrix[user_index, :] = np.linalg.solve(y_dot_y + lambda_i,                                                                 self.characteristic_matrix[user_index, :].dot(constant_matrix))
            
        elif chance == AlternatingLeastSquare.MOVIES_MATRIX:
            x_dot_x = constant_matrix.T.dot(constant_matrix)
            lambda_i = np.identity(x_dot_x.shape[0]) * self.movies_reg
            
            for movie_index in range(variable_matrix.shape[0]):
                variable_matrix[movie_index, :] = np.linalg.solve(x_dot_x + lambda_i,                                                                 self.characteristic_matrix[:, movie_index].T.dot(constant_matrix))
        return variable_matrix
    
    def predict(self, user, movie):
        return self.users_matrix[user, :].dot(self.movies_matrix[movie, :].T)
    
    def predict_all(self):
        # predictions = np.zeros((self.total_users, self.total_movies))
        # for user in xrange(self.total_users):
        #     for movie in xrange(self.total_movies):
        #         predictions[user, movie] = self.predict(user, movie)

        predictions = self.users_matrix.dot(self.movies_matrix.T)
                
        return predictions
    
    def get_train_test_error(self, test_data, iters_list):
        test_error = []
        train_error = []
        iters_count_till_now = 0
        for iters in iters_list:
            start = time.time()
            remaining_iters = iters - iters_count_till_now
            if remaining_iters < 1:
                continue
            if iters_count_till_now == 0:
                self.train(iters=remaining_iters)
            else:
                self.train_continue(iters=remaining_iters)
            iters_count_till_now += remaining_iters
            predicted = self.predict_all()
            end = time.time()
            test_error.append(AlternatingLeastSquare.calculate_mean_squared_error(predicted, test_data))
            train_error.append(AlternatingLeastSquare.calculate_mean_squared_error(predicted, self.characteristic_matrix))
            print "After [%d] itertions, Time taken [%d] secs, Training error: [%f], Test error: [%f]"%(iters_count_till_now, end-start, train_error[-1], test_error[-1])
        return train_error, test_error

def loadDatasets():
    movies = pd.read_csv('data/movie.csv')
    ratings = pd.read_csv('data/rating.csv')
    return movies, ratings

def preprocessData(user_movie_rating, factor=0.5):
    size = user_movie_rating.shape[0]
    partition_index = int(size*factor)
    subset = user_movie_rating.iloc[:partition_index, :]
    subset = subset.dropna()
    subset = subset.drop('timestamp', axis=1)
    return subset

def train_test_split(characteristic_matrix):
    test = np.zeros(characteristic_matrix.shape)
    train = characteristic_matrix.copy()
    for user in xrange(characteristic_matrix.shape[0]):
        test_indexes = np.random.choice(characteristic_matrix[user, :].nonzero()[0], size=10, replace=False)
        train[user, test_indexes] = 0.0
        test[user, test_indexes] = characteristic_matrix[user, test_indexes]
        
    assert(np.all((train * test) == 0)) 
    return train, test

def getCharateristicMatrix(user_movie_rating):
    characteristic_df = user_movie_rating.pivot('userId', 'movieId', values='rating')
    characteristic_df = characteristic_df.fillna(0)
    characteristic_matrix = characteristic_df.as_matrix()
    characteristic_df.index.name = None
    
    movie_mapping_df = pd.DataFrame({'matrix_index': range(characteristic_df.shape[1]), 'movie_id': characteristic_df.columns})
    movie_mapping = dict(zip(movie_mapping_df.matrix_index, movie_mapping_df.movie_id))
    r_movie_mapping = dict(zip(movie_mapping_df.movie_id, movie_mapping_df.matrix_index))
    
    user_mapping_df = pd.DataFrame({'matrix_index': range(characteristic_df.shape[0]), 'user_id': characteristic_df.index.tolist()})
    user_mapping = dict(zip(user_mapping_df.matrix_index, user_mapping_df.user_id))
    r_user_mapping = dict(zip(user_mapping_df.user_id, user_mapping_df.matrix_index))
    mappings = {
        'user_mapping': user_mapping,
        'movie_mapping': movie_mapping,
        'r_user_mapping': r_user_mapping,
        'r_movie_mapping': r_movie_mapping
    }
    return characteristic_matrix, mappings

def getUserIdsFromMatrixIndexes(matrix_indexes, user_mapping, preserve_order=True):
    user_ids = []
    for index in matrix_indexes:
        user_ids.append(user_mapping[index])
    return user_ids

def getMovieIdsFromMatrixIndexes(matrix_indexes, movie_mapping, preserve_order=True):
    movie_ids = []
    for index in matrix_indexes:
        movie_ids.append(movie_mapping[index])
    return movie_ids

def getWatchedMovies(user_id, user_movie_rating_df):
    rated_movies = user_movie_rating_df[user_movie_rating_df['userId'] == user_id]['movieId'].values.tolist()
    return rated_movies

def filterWatchedMovies(user_id, ordered_movie_ids, user_movie_rating_df):
    all_movies = set(ordered_movie_ids)
    rated_movies = set(getWatchedMovies(user_id, user_movie_rating_df))
    print "Watched : ", len(rated_movies)
    not_watched = all_movies - rated_movies
    ordered_not_watched = [movie_id for movie_id in ordered_movie_ids if movie_id not in rated_movies]
    return ordered_not_watched

def displayMovies(movie_ids, movies_df):
    for movie_id in movie_ids:
        print movies_df[movies_df['movieId'] == movie_id].values.tolist()[0]

def findBestHyperParams():
    latent_factors = [5, 10, 20, 40, 80]
    regularizations = [0.01, 0.1, 1., 10., 100.]
    regularizations.sort()
    iter_array = list(range(5, 100, 5))

    best_params = {}
    best_params['n_factors'] = latent_factors[0]
    best_params['reg'] = regularizations[0]
    best_params['n_iter'] = 0
    best_params['train_mse'] = np.inf
    best_params['test_mse'] = np.inf
    best_params['model'] = None

    for fact in latent_factors:
        print 'Factors: {}'.format(fact)
        for reg in regularizations:
            print 'Regularization: {}'.format(reg)
            als = AlternatingLeastSquare(train, latent_feature_count=fact, users_reg=0.3, movies_reg=0.3)
            train_mse, test_mse = als.get_train_test_error(test, iter_array)
            min_idx = np.argmin(test_mse)
            if test_mse[min_idx] < best_params['test_mse']:
                best_params['n_factors'] = fact
                best_params['reg'] = reg
                best_params['n_iter'] = iter_array[min_idx]
                best_params['train_mse'] = train_mse[min_idx]
                best_params['test_mse'] = test_mse[min_idx]
                best_params['model'] = als
                print 'New optimal hyperparameters'
                print pd.Series(best_params)
    return best_params

def main():
    movies, all_ratings = loadDatasets()
    ratings = preprocessData(all_ratings, factor=0.001)
    characteristic_matrix, mappings = getCharateristicMatrix(ratings)
    train_matrix, test_matrix = train_test_split(characteristic_matrix)
    
    als = AlternatingLeastSquare(train_matrix, latent_feature_count=40, users_reg=0.1, movies_reg=0.1)
    als.train(iters=95)
    
    movie_ids = recommendMoviesTo(31, als, mappings, limit=300)
    print len(movie_ids)
    
    f_movie_ids = filterWatchedMovies(31, movie_ids, ratings)
    print len(f_movie_ids)
    
    displayMovies(f_movie_ids_anime, movies)

if __name__ == '__main__':
    main()

