import redis
import random
from flask import Flask, render_template, request, jsonify, Response
from model import Recommender

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
	return render_template('movies.html', queryResponse=movies)



if __name__ == "__main__":
	app.run('', port=5000, debug=True)

