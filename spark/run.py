#!/usr/bin/env python
# coding: utf-8


from __future__ import print_function

import findspark
findspark.init()

import sys
import pickle
import numpy as np
import pandas as pd
from model import BestModel, AlternatingLeastSquare
from numpy.random import rand
from numpy import matrix
from pyspark.sql import SparkSession

LAMBDA = 0.01
M = 0
U = 0
F = 0

def loadDatasets():
	print('Loading datasets')
	movies = pd.read_csv('data/movie.csv')
	ratings = pd.read_csv('data/rating.csv')
	print('Datasets loaded successfully')
	return movies, ratings


def preprocessData(user_movie_rating, factor=0.25):
	size = user_movie_rating.shape[0]
	partition_index = int(size*factor)
	subset = user_movie_rating.iloc[:partition_index, :]
	subset = subset.dropna()
	subset = subset.drop('timestamp', axis=1)
	return subset


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

def rmse(R, ms, us):
	diff = R - ms * us.T
	return np.sqrt(np.sum(np.power(diff, 2)) / (M * U))

def updateMovie(i, mat, movie_rows, m_offset):
	uu = mat.shape[0]
	ff = mat.shape[1]

	XtX = mat.T * mat
	Xty = mat.T * movie_rows[i%m_offset, :].T

	for j in range(ff):
		XtX[j, j] += LAMBDA * uu

	return np.linalg.solve(XtX, Xty)



def updateUser(i, mat, user_cols, u_offset):
	uu = mat.shape[0]
	ff = mat.shape[1]

	XtX = mat.T * mat
	Xty = mat.T * user_cols[i%u_offset, :].T

	for j in range(ff):
		XtX[j, j] += LAMBDA * uu

	return np.linalg.solve(XtX, Xty)

def trainModel():
	movies, ratings = loadDatasets()

	ratings_subset = preprocessData(ratings, factor=0.75)
	characteristic_matrix, mappings = getCharateristicMatrix(ratings_subset)

	spark = SparkSession.builder.appName("MovieRecommender").getOrCreate()

	sc = spark.sparkContext

	n_characteristic_matrix = np.mat(characteristic_matrix.T)

	global M, U, F
	M = n_characteristic_matrix.shape[0]
	U = n_characteristic_matrix.shape[1]
	F = 20
	ITERATIONS = 5
	partitions = 2

	R = n_characteristic_matrix
	ms = matrix(rand(M, F))
	us = matrix(rand(U, F))


	m_offset = 1500

	u_offset = 10000

	b_movies = []
	b_users = []

	index = 0
	while index*m_offset < M:
		b_movies.append(sc.broadcast(R[m_offset*index:(index+1)*m_offset]))
		index += 1

	index = 0
	while index*u_offset < U:
		b_users.append(sc.broadcast(R[:, u_offset*index:(index+1)*u_offset]))
		index += 1

	msb = sc.broadcast(ms)
	usb = sc.broadcast(us)

	for i in range(ITERATIONS):
		ms = sc.parallelize(range(M), partitions).map(lambda x: updateMovie(x, usb.value, b_movies[x/m_offset].value,  m_offset)).collect()
		ms = matrix(np.array(ms)[:, :, 0])
		msb = sc.broadcast(ms)

		us = sc.parallelize(range(U), partitions).map(lambda x: updateUser(x, msb.value,  b_users[x/u_offset].value.T, u_offset)).collect()
		us = matrix(np.array(us)[:, :, 0])
		usb = sc.broadcast(us)

		error = rmse(R, ms, us)
		print("Iteration %d:" % i)
		print("\nRMSE: %5.4f\n" % error)



	with open('data/1ms.bin', mode='wb') as model_binary:
		pickle.dump(ms, model_binary)


	with open('data/1us.bin', mode='wb') as model_binary:
		pickle.dump(us, model_binary)


	with open('1mappings.bin', mode='wb') as model_binary:
		pickle.dump(mappings, model_binary)


	als = AlternatingLeastSquare(np.array([[]]))
	als.users_matrix = np.array(us)
	als.movies_matrix = np.array(ms)
	best_model = BestModel(movies, als, ratings_subset, mappings)

	with open('data/1best_model.bin', mode='wb') as model_binary:
		pickle.dump(best_model, model_binary)

if __name__ == '__main__':
	trainModel()

