import pandas as pd
from elasticsearch import Elasticsearch

INDEX_NAME = 'movies'

class Indexer(object):
	def __init__(self):
		self.es = Elasticsearch(host='elasticsearch_instance')

	def dropIndex(self):
		print('Dropping index')
		try:
			self.es.indices.delete(INDEX_NAME)
		except:
			print('No index is present')
		print ('Index dropped')

	def createIndex(self):
		self.dropIndex()
		print('Creating new index')
		movies = pd.read_csv('data/movie.csv')
		for i in range(1, movies.shape[0]):
			try:
				movie = {'id': int(movies.iloc[i][0]),'title': movies.iloc[i][1]}
				self.es.index(index=INDEX_NAME, doc_type='movie', body=movie)
			except Exception as ex:
				print('Indexing failed for i=',i, str(ex))
			if i%250 == 0:
				print(i)
		print('New index created')

	def search(self, keyword):
		resp = self.es.search(index=INDEX_NAME, body={'query': {'match': {'title': keyword}}})
		movies = []
		for doc in resp['hits']['hits']:
			movies.append({'id':doc['_source']['id'], 'title': doc['_source']['title']})
		return movies

if __name__ == '__main__':
	indexer = Indexer()
	indexer.createIndex()