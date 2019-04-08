import numpy as np
from model_loader import ModelLoader

class Recommender(object):
    def __init__(self):
        np.random.seed(0)
        self.model_obj = ModelLoader().load()
        self.model = self.model_obj.model
        self.mappings = self.model_obj.mappings
        self.ratings = self.model_obj.ratings
        self.movies = self.model_obj.movies

    def getUserIdsFromMatrixIndexes(self, matrix_indexes, user_mapping, preserve_order=True):
        user_ids = []
        for index in matrix_indexes:
            user_ids.append(user_mapping[index])
        return user_ids

    def getMovieIdsFromMatrixIndexes(self, matrix_indexes, preserve_order=True):
        movie_ids = []
        movie_mapping = self.mappings['movie_mapping']
        for index in matrix_indexes:
            movie_ids.append(movie_mapping[index])
        return movie_ids

    def getWatchedMovies(self, user_id):
        rated_movies = self.ratings[self.ratings['userId'] == user_id]['movieId'].values.tolist()
        return rated_movies

    def recommendMoviesTo(self, user_id, limit=50):
        user_index = self.mappings['r_user_mapping'][user_id]
        user_vector = self.model.users_matrix[user_index, :]
        user_ratings = user_vector.dot(self.model.movies_matrix.T)
        highest_rated_movie_indexes = user_ratings.argsort()[::-1][:limit]
        movie_ids = self.getMovieIdsFromMatrixIndexes(highest_rated_movie_indexes)
        return movie_ids

    def displayMovies(self, movie_ids):
        movie_infos = []
        for movie_id in movie_ids:
            movie_info = self.movies[self.movies['movieId'] == movie_id].values.tolist()[0]
            movie_infos.append(movie_info)
        return movie_infos

    def filterWatchedMovies(self, user_id, ordered_movie_ids):
        all_movies = set(ordered_movie_ids)
        rated_movies = set(self.getWatchedMovies(user_id))
        not_watched = all_movies - rated_movies
        ordered_not_watched = [movie_id for movie_id in ordered_movie_ids if movie_id not in rated_movies]
        return ordered_not_watched

if __name__ == '__main__':
    recommender = Recommender()
    movie_ids = recommender.recommendMoviesTo(1, limit=300)
    print movie_ids
    print recommender.displayMovies(movie_ids)


