import redis
import random
from flask import Flask, render_template, request, jsonify, Response
from model import Recommender

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

app = Flask(__name__)

db = redis.Redis('redis')

recommender = Recommender()

@app.route("/")
def home():
	return render_template('home.html')

@app.route('/<user_id>')
def user(user_id):
	movies = [
				{
					'title': 'Somthing Somthing',
					'url': 'https://m.media-amazon.com/images/M/MV5BMWU4ZjNlNTQtOGE2MS00NDI0LWFlYjMtMmY3ZWVkMjJkNGRmXkEyXkFqcGdeQXVyNjE1OTQ0NjA@._V1_UY268_CR2,0,182,268_AL_.jpg'
				},
				{
					'title': 'Another Somthing',
					'url': 'https://m.media-amazon.com/images/M/MV5BMWU4ZjNlNTQtOGE2MS00NDI0LWFlYjMtMmY3ZWVkMjJkNGRmXkEyXkFqcGdeQXVyNjE1OTQ0NjA@._V1_UY268_CR2,0,182,268_AL_.jpg'
				}
		]
	movie_ids = recommender.recommendMoviesTo(1, limit=300)
	movies_info = recommender.displayMovies(movie_ids)
	movies = prepareMovies(movies_info)
	return render_template('movies.html', queryResponse=movies)

def prepareMovies(movies_info):
	movies = []
	for movie_info in movies_info:
		movie = {}
		movie['title'] = movie_info[1]
		movie['genre'] = movie_info[2]
		movie['url'] = ''
		movies.append(movie)
	return movies

if __name__ == "__main__":
	app.run('', port=5000, debug=True)

