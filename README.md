word2vec-api
============

Simple web service providing a word embedding API. The methods are based on Gensim Word2Vec implementation. Models are passed as parameters and must be in the Word2Vec text or binary format.
* Install Depenencies
```
pip2 install -r requirements.txt
```

* Launching the service
```
python word2vec-api --model path/to/the/model --second_model path/to/the/second_model [--host host --port 1234]
```
or
```
python word2vec-api.py --model /home/ubuntu/data/models/GoogleNews-vectors-negative300.bin --second_model /home/ubuntu/data/models/wiki.en.vec --binary BINARY --path /word2vec --host 0.0.0.0 --port 80
```


Note: The "model" method returns a base64 encoding of the vector. "model\_word\_set" returns a base64 encoded pickle of the model's vocabulary.

## Where to get a pretrained model

In case you do not have domain specific data to train, it can be convenient to use a pretrained model.
Please feel free to submit additions to this list through a pull request.


| Model file | Number of dimensions | Corpus (size)| Vocabulary size | Author | Architecture | Training Algorithm | Context window - size | Web page |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [Google News](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/) | 300 |Google News (100B) | 3M | Google | word2vec | negative sampling | BoW - ~5| [link](http://code.google.com/p/word2vec/) |
| [Freebase IDs](https://docs.google.com/file/d/0B7XkCwpI5KDYaDBDQm1tZGNDRHc/edit?usp=sharing) | 1000 | Gooogle News (100B) | 1.4M | Google | word2vec, skip-gram | ? | BoW - ~10 | [link](http://code.google.com/p/word2vec/) |
| [Freebase names](https://docs.google.com/file/d/0B7XkCwpI5KDYeFdmcVltWkhtbmM/edit?usp=sharing) | 1000 | Gooogle News (100B) | 1.4M | Google | word2vec, skip-gram | ? | BoW - ~10 | [link](http://code.google.com/p/word2vec/) |
| [Wikipedia+Gigaword 5](http://nlp.stanford.edu/data/glove.6B.zip) | 50 | Wikipedia+Gigaword 5 (6B) | 400,000 | GloVe | GloVe | AdaGrad | 10+10 | [link](http://nlp.stanford.edu/projects/glove/) |
| [Wikipedia+Gigaword 5](http://nlp.stanford.edu/data/glove.6B.zip) | 100 | Wikipedia+Gigaword 5 (6B) | 400,000 | GloVe | GloVe | AdaGrad | 10+10 | [link](http://nlp.stanford.edu/projects/glove/) |
| [Wikipedia+Gigaword 5](http://nlp.stanford.edu/data/glove.6B.zip) | 200 | Wikipedia+Gigaword 5 (6B) | 400,000 | GloVe | GloVe | AdaGrad | 10+10 | [link](http://nlp.stanford.edu/projects/glove/) |
| [Wikipedia+Gigaword 5](http://nlp.stanford.edu/data/glove.6B.zip) | 300 | Wikipedia+Gigaword 5 (6B) | 400,000 | GloVe | GloVe | AdaGrad | 10+10 | [link](http://nlp.stanford.edu/projects/glove/) |
| [Common Crawl 42B](http://nlp.stanford.edu/data/glove.42B.300d.zip) | 300 | Common Crawl (42B) | 1.9M | GloVe | GloVe | GloVe | AdaGrad | [link](http://nlp.stanford.edu/projects/glove/) |
| [Common Crawl 840B](http://nlp.stanford.edu/data/glove.840B.300d.zip) | 300 | Common Crawl (840B) | 2.2M | GloVe | GloVe | GloVe | AdaGrad | [link](http://nlp.stanford.edu/projects/glove/) |
| [Twitter (2B Tweets)](http://www-nlp.stanford.edu/data/glove.twitter.27B.zip) | 25 | Twitter (27B) | ? | GloVe | GloVe | GloVe | AdaGrad | [link](http://nlp.stanford.edu/projects/glove/) |
| [Twitter (2B Tweets)](http://www-nlp.stanford.edu/data/glove.twitter.27B.zip) | 50 | Twitter (27B) | ? | GloVe | GloVe | GloVe | AdaGrad | [link](http://nlp.stanford.edu/projects/glove/) |
| [Twitter (2B Tweets)](http://www-nlp.stanford.edu/data/glove.twitter.27B.zip) | 100 | Twitter (27B) | ? | GloVe | GloVe | GloVe | AdaGrad | [link](http://nlp.stanford.edu/projects/glove/) |
| [Twitter (2B Tweets)](http://www-nlp.stanford.edu/data/glove.twitter.27B.zip) | 200 | Twitter (27B) | ? | GloVe | GloVe | GloVe | AdaGrad | [link](http://nlp.stanford.edu/projects/glove/) |
| [Wikipedia dependency](http://u.cs.biu.ac.il/~yogo/data/syntemb/deps.words.bz2) | 300 | Wikipedia (?) | 174,015 | Levy \& Goldberg | word2vec modified | word2vec | syntactic dependencies | [link](https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/) |
| [DBPedia vectors (wiki2vec)](https://github.com/idio/wiki2vec/raw/master/torrents/enwiki-gensim-word2vec-1000-nostem-10cbow.torrent) | 1000 | Wikipedia (?) | ? | Idio | word2vec | word2vec, skip-gram | BoW, 10 | [link](https://github.com/idio/wiki2vec#prebuilt-models) |
