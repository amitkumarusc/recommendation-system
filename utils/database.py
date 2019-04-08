import psycopg2
import time, sys

class Database(object):
	def __init__(self):
		while True:
			try:
				self.conn = psycopg2.connect("dbname='movie_recommender' user='postgres' host='postgres' port=5432 password=''")
				break
			except:
				print 'Unable to connect to postgres. Retrying after 1 second.'
				time.sleep(1)
		print 'Connected to postgres'
		self.conn.autocommit = True
		cursor = self.conn.cursor()

	def checkConnectivity(self):
		sql = 'SELECT version()';
		cursor = self.conn.cursor()
		cursor.execute(sql)
		record = cursor.fetchone()
		return record != None

	def getMovieUrl(self, old_movie_id):
		sql = "select url from idmapper where oldid='%s';"%(str(old_movie_id))
		cursor = self.conn.cursor()
		cursor.execute(sql)
		record = None
		for row in cursor:
			record = row
		cursor.close()
		url = record[0] if record != None else ''
		return url