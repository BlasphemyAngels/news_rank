
tar -zcvf news.tar.gz --exclude=Similarity* --exclude=*data.csv --exclude=__pycache__/ --exclude=*.txt --exclude=word2vec* --exclude=*.tar.gz --exclude=doc2vec* *

scp -r news.tar.gz neukg6@219.216.64.90:/home/neukg6/a
