import pandas as pd
import requests, json

INDEX_NAME = 'movies'

class Indexer(object):

	def dropIndex(self):
		print('Dropping index')
		try:
			resp = requests.delete('http://elasticsearch:9200/' + INDEX_NAME, headers={'content-type':'application/json'})
		except:
			print('No index is present')
		print ('Index dropped')

	def createIndex(self):
		query = '''{
			  "mappings": {
				"movie": {
				  "properties": {
					"title": {
					  "type": "completion"
					},
					"id": {
					  "type": "keyword"
					}
				  }
				}
			  }
			}'''

		resp = requests.put('http://elasticsearch:9200/' + INDEX_NAME, data=query, headers={'content-type':'application/json'})
		print('Resp text : ', resp.text)

	def getForms(self, text):
		ans = [text]
		text = text.split(' ')
		while len(text) > 0:
			temp = ' '.join(text[1:]).strip()
			if temp:
				ans.append(temp)
			text = text[1:]
		return ans


	def buildIndex(self):
		self.dropIndex()
		print('Creating new index')
		self.createIndex()
		movies = pd.read_csv('data/movie.csv')
		for i in range(1, movies.shape[0]):
			try:
				movie = {'id': str(movies.iloc[i][0]), 'title': { 'input' : self.getForms(movies.iloc[i][1]) }}
				movie = json.dumps(movie)
				resp = requests.post('http://elasticsearch:9200/' + INDEX_NAME + '/movie', data=movie, headers={'content-type':'application/json'}).json()

			except Exception as ex:
				print('Indexing failed for i=',i, str(ex))
			if i%250 == 0:
				print(i)
		print('New index created')

	def search(self, keyword):
		keyword = keyword.lower()
		query = '{"suggest": {"movie-suggest-fuzzy": {"prefix": "%s","completion": {"field": "title", "size": 200, "fuzzy": {"fuzziness": 1 }}}}}'%keyword
		resp = requests.post('http://elasticsearch:9200/' + INDEX_NAME + '/_search', data=query, headers={'content-type':'application/json'}).json()
		movies = []
		for doc in resp['suggest']['movie-suggest-fuzzy'][0]['options']:
			movies.append({'id':doc['_source']['id'], 'title': doc['_source']['title']['input'][0]})
		return movies

if __name__ == '__main__':
	indexer = Indexer()
	indexer.buildIndex()