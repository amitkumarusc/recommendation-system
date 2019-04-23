import psycopg2
import time, sys

class Database(object):
	def __init__(self):
		count = 10
		while count > 0:
			try:
				self.conn = psycopg2.connect("dbname='movie_recommender' user='postgres' host='postgres' port=5432 password=''")
				break
			except:
				print 'Unable to connect to postgres. Retrying after 1 second.'
				time.sleep(1)
				count -= 1
		print 'Connected to postgres'
		self.conn.autocommit = True
		cursor = self.conn.cursor()

	def checkConnectivity(self):
		sql = 'SELECT version()';
		cursor = self.conn.cursor()
		cursor.execute(sql)
		record = cursor.fetchone()
		return record != None

	def getMovieInfo(self, movie_id):
		sql = "select title, url, year, genres, rating, numvotes from new_movies where oldid='%s';"%(str(movie_id))
		cursor = self.conn.cursor()
		cursor.execute(sql)
		record = None
		for row in cursor:
			record = row
		cursor.close()
		movie = {}
		if record:
			movie = {'title' : record[0], 'url': record[1],\
				'year' : record[2], 'genres' : record[3].split(','), \
				'rating' : float(record[4])/2.0, 'votes': record[5]}
		return movie

	def getRatings(self, movie_id):
		sql = "select rating from rating_events where movieid='%s' limit 400;"%(str(movie_id))
		cursor = self.conn.cursor()
		cursor.execute(sql)
		times = {'ratings': [], 'created_at': []}
		for row in cursor:
			times['ratings'].append(float(row[0]))
		return times

	def getMovieUrl(self, old_movie_id):
		sql = "select url from new_movies where oldid='%s';"%(str(old_movie_id))
		cursor = self.conn.cursor()
		cursor.execute(sql)
		record = None
		for row in cursor:
			record = row
		cursor.close()
		url = record[0] if record != None else ''
		return url