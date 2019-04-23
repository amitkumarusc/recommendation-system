import numpy as np
from model_loader import ModelLoader

RESP_SIZE = 30

class Recommender(object):
    def __init__(self):
        np.random.seed(0)
        self.model_obj = ModelLoader().load()
        self.model = self.model_obj.model
        self.mappings = self.model_obj.mappings
        self.ratings = self.model_obj.ratings
        self.movies = self.model_obj.movies
        self.preparePopularMovies()

    def preparePopularMovies(self):
        groups = self.ratings.groupby(by='movieId')['rating'].agg(['sum','count'])
        groups = groups.reset_index()
        groups['averageRating'] = groups['sum']/(1.0*groups['count'])
        average_rating = groups['averageRating'].mean()
        minimum_reviews = groups['count'].quantile(0.90)
        self.popularMovies = groups.copy().loc[groups['count'] >= minimum_reviews]
        self.popularMovies['score'] = self.popularMovies.apply(lambda x: self.bayesian_average(x, minimum_reviews, average_rating), axis=1)
        self.popularMovies = self.popularMovies.sort_values('score', ascending=False)

    def getPopularMovies(self, limit=RESP_SIZE):
        return list(self.popularMovies[:limit]['movieId'].values)

    def getRecentMovies(self):
        years = map(str, range(2015, 2020))
        recent_movies = self.movies[self.movies['title'].str.contains('|'.join(years)) & ~self.movies['title'].str.startswith('2')][:RESP_SIZE]['movieId'].values
        return list(recent_movies)

    def bayesian_average(self, row, minimum_reviews, average_rating):
        v = row['count']
        R = row['averageRating']
        return (v/(v+minimum_reviews) * R) + (minimum_reviews/(minimum_reviews+v) * average_rating)

    def getWatchedMovies(self, user_id, limit=RESP_SIZE):
        if limit:
            already_watched = self.ratings[self.ratings['userId'] == user_id].sort_values('rating', ascending=False)[:RESP_SIZE]['movieId'].values
        else:
            already_watched = self.ratings[self.ratings['userId'] == user_id].sort_values('rating', ascending=False)['movieId'].values
        return list(already_watched)

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

    def recommendMoviesTo(self, user_id, limit=RESP_SIZE):
        user_index = self.mappings['r_user_mapping'][user_id]
        user_vector = self.model.users_matrix[user_index, :]
        user_ratings = user_vector.dot(self.model.movies_matrix.T)
        user_ratings = user_ratings + np.asarray(self.model.user_bias).reshape(-1,)[user_index] + np.asarray(self.model.movie_bias).reshape(-1,)
        highest_rated_movie_indexes = user_ratings.argsort()[::-1]
        movie_ids = self.getMovieIdsFromMatrixIndexes(highest_rated_movie_indexes)
        movie_ids = self.filterWatchedMovies(user_id, movie_ids)
        return movie_ids[:limit]

    def displayMovies(self, movie_ids):
        movie_infos = []
        for movie_id in movie_ids:
            movie_info = self.movies[self.movies['movieId'] == movie_id].values.tolist()[0]
            movie_infos.append(movie_info)
        return movie_infos

    def filterWatchedMovies(self, user_id, ordered_movie_ids):
        all_movies = set(ordered_movie_ids)
        rated_movies = set(self.getWatchedMovies(user_id, limit = None))
        not_watched = all_movies - rated_movies
        ordered_not_watched = [movie_id for movie_id in ordered_movie_ids if movie_id not in rated_movies]
        return ordered_not_watched

if __name__ == '__main__':
    recommender = Recommender()
    movie_ids = recommender.recommendMoviesTo(1, limit=300)
    print movie_ids
    print recommender.displayMovies(movie_ids)


