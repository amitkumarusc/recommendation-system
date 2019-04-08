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

class AlternatingLeastSquare(object):
    USERS_MATRIX = 'users_matrix'
    MOVIES_MATRIX = 'movies_matrix'
    
    def __init__(self, characteristic_matrix, latent_feature_count=50, users_reg=0.0, movies_reg=0.0):
        np.random.seed(0)
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
                variable_matrix[user_index, :] = np.linalg.solve(y_dot_y + lambda_i, self.characteristic_matrix[user_index, :].dot(constant_matrix))
            
        elif chance == AlternatingLeastSquare.MOVIES_MATRIX:
            x_dot_x = constant_matrix.T.dot(constant_matrix)
            lambda_i = np.identity(x_dot_x.shape[0]) * self.movies_reg
            
            for movie_index in range(variable_matrix.shape[0]):
                variable_matrix[movie_index, :] = np.linalg.solve(x_dot_x + lambda_i, self.characteristic_matrix[:, movie_index].T.dot(constant_matrix))
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
    
    def get_train_test_error(self, test_data, iters_list, debug=False):
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
            if debug:
                print "After [%d] itertions, Time taken [%d] secs, Training error: [%f], Test error: [%f]"%(iters_count_till_now, end-start, train_error[-1], test_error[-1])
        return train_error, test_error


