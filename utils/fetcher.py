import requests
from bs4 import BeautifulSoup

def fetchImageUrl(movie_id):
    url = "https://www.imdb.com/title/tt%d/"%(movie_id)
    resp = requests.get(url)
    if resp.status_code >= 400:
        return ''
    img_url = ''
    try:
        bs = BeautifulSoup(resp.text)
        poster_div = bs.findAll("div", {"class": "poster"})[0]
        poster_img = poster_div.findAll('img')[0]
        img_url = poster_img['src']
    except:
        pass
    return img_url

if __name__ == '__main__':
    fetchImageUrl(8291224)

# https://www.imdb.com/title/tt8291224/?ref_=nv_sr_1