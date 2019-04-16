class BestModel(object):
	def __init__(self, movies, model, ratings, mappings):
		self.ratings = ratings
		self.model = model
		self.mappings = mappings
		self.movies = movies