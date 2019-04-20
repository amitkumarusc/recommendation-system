import os
import pandas as pd
import numpy as np
import pickle
from als import AlternatingLeastSquare
from .best_model import BestModel

MODEL_PATH = 'data/_best_model.bin'

class ModelLoader(object):
	def __init__(self):
		np.random.seed(0)
		pass

	def load(self):
		if os.path.isfile(MODEL_PATH):
			print('Loading the model from disk')
			self.model = self.readModelFromDisk()
			print('Model successfully loaded')
		else:
			print('Retraining the model from scratch')
			self.model = self.trainModelFromScratch()
			print('Model successfully trained')
		return self.model

	def trainModelFromScratch(self):
		movies, ratings = self.loadDatasets()
		ratings_subset = self.preprocessData(ratings, factor=0.5)
		characteristic_matrix, mappings = self.getCharateristicMatrix(ratings_subset)
		train_matrix, test_matrix = self.train_test_split(characteristic_matrix)
		best_model = self.findBestHyperParams(train_matrix, test_matrix)
		model = BestModel(movies, best_model['model'], ratings_subset, mappings)
		self.writeModelToDisk(model)
		return model
		
	def writeModelToDisk(self, model):
		model_file_name = "model_latentFeatureCount_userReg_moviesReg_iterCount"
		with open(MODEL_PATH, mode='wb') as model_binary:
			pickle.dump(model, model_binary)

	def readModelFromDisk(self):
		with open(MODEL_PATH, 'rb') as model_binary:
			model = pickle.load(model_binary)
		return model

	def loadDatasets(self):
		print('Loading datasets')
		movies = pd.read_csv('data/movie.csv')
		ratings = pd.read_csv('data/rating.csv')
		print('Datasets loaded successfully')
		return movies, ratings

	def preprocessData(self, user_movie_rating, factor=0.25):
		size = user_movie_rating.shape[0]
		partition_index = int(size*factor)
		subset = user_movie_rating.iloc[:partition_index, :]
		subset = subset.dropna()
		subset = subset.drop('timestamp', axis=1)
		return subset

	def train_test_split(self, characteristic_matrix):
		test = np.zeros(characteristic_matrix.shape)
		train = characteristic_matrix.copy()
		for user in xrange(characteristic_matrix.shape[0]):
			test_indexes = np.random.choice(characteristic_matrix[user, :].nonzero()[0], size=10, replace=False)
			train[user, test_indexes] = 0.0
			test[user, test_indexes] = characteristic_matrix[user, test_indexes]
			
		assert(np.all((train * test) == 0)) 
		return train, test

	def getCharateristicMatrix(self, user_movie_rating):
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

	def findBestHyperParams(self, train, test):
		latent_features = [5, 10, 20, 40, 80]
		regularizations = [0.01, 0.1, 1., 10., 100.]
		regularizations.sort()
		iter_array = list(range(5, 100, 5))

		best_params = {}
		best_params['n_features'] = latent_features[0]
		best_params['reg'] = regularizations[0]
		best_params['iters'] = 0
		best_params['train_mse'] = np.inf
		best_params['test_mse'] = np.inf
		best_params['model'] = None

		for fact in latent_features:
			print 'Factors: {}'.format(fact)
			for reg in regularizations:
				print 'Regularization: {}'.format(reg)
				als = AlternatingLeastSquare(train, latent_feature_count=fact, users_reg=reg, movies_reg=reg)
				train_mse, test_mse = als.get_train_test_error(test, iter_array)
				min_idx = np.argmin(test_mse)
				if test_mse[min_idx] < best_params['test_mse']:
					best_params['n_features'] = fact
					best_params['reg'] = reg
					best_params['iters'] = iter_array[min_idx]
					best_params['train_mse'] = train_mse[min_idx]
					best_params['test_mse'] = test_mse[min_idx]
					best_params['model'] = als
					print 'New optimal hyperparameters'
					print pd.Series(best_params)
		return best_params

if __name__ == '__main__':
	model_loader = ModelLoader()
	model_loader.load()

