from flask import Flask
from word2vec_api.handlers import InvalidUsage, N_SimilarWords, Similarity_V2, Get_vectors, Similarity_batch, N_Similarity, Similarity, MostSimilar, Model, ModelWordSet
from flask.ext.restful import Api
from IPython import embed
import argparse

application = Flask(__name__)
api = Api(application)

host = '0.0.0.0'
port = 9000
path = '/word2vec'

print 'register handlers'
api.add_resource(N_Similarity, path+'/n_similarity')
api.add_resource(Similarity, path+'/similarity')
api.add_resource(MostSimilar, path+'/most_similar')
api.add_resource(Model, path+'/model')
api.add_resource(ModelWordSet, '/model_word_set')
api.add_resource(N_SimilarWords, path + '/n_most_similar')
api.add_resource(Similarity_V2, path+'/similarity_v2')
api.add_resource(Similarity_batch, path+'/similarity_batch')
api.add_resource(Get_vectors, path+'/vectors')
api_app = api.app

if __name__ == '__main__':
  api_app.run(host=host, port=port)
