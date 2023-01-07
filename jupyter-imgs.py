from bs4 import BeautifulSoup as soup
import base64
with open('./main.html',encoding='utf-8') as file:
    bs = soup(file,'lxml')
    imgs = bs.find_all('img')
    for i,img in enumerate(imgs):
        data = img['src'].split(',')[1]
        with open('./imgs/'+str(i)+'.png','wb') as f:
            f.write(base64.b64decode(data))
            f.close()