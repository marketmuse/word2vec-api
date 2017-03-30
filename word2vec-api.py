'''
Simple web service wrapping a Word2Vec as implemented in Gensim
Example call: curl http://127.0.0.1:5000/wor2vec/n_similarity/ws1=Sushi&ws1=Shop&ws2=Japanese&ws2=Restaurant
@TODO: Add more methods
@TODO: Add command line parameter: path to the trained model
@TODO: Add command line parameters: host and port
'''

from flask import Flask, request, jsonify
from flask.ext.restful import Resource, Api, reqparse
from gensim.models.word2vec import Word2Vec as w
from gensim import utils, matutils
from numpy import exp, dot, zeros, outer, random, dtype, get_include, float32 as REAL,\
     uint32, seterr, array, uint8, vstack, argsort, fromstring, sqrt, newaxis, ndarray, empty, sum as np_sum
import cPickle
import argparse
import base64
import sys

import gensim
from gensim.models.wrappers import fasttext

from IPython import embed

# Proprietary classes
from dictionary import Dictionary

from werkzeug.exceptions import abort


# Flask hotfix from here: https://github.com/pallets/flask/issues/941
import flask
from werkzeug.exceptions import default_exceptions


parser = reqparse.RequestParser()

class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv



def filter_words(words):
    if words is None:
        return
    return [word for word in words if word in model.vocab]


class N_SimilarWords(Resource):
  def get(self):
      # PARAMS:
      # required: phrase
      # optional: min_gram, max_gram, topn
      parser = reqparse.RequestParser()
      parser.add_argument('phrase', type=str, required=True, help='Phrase parameter cannot be blank!')
      parser.add_argument('topn', type=int, default=100, help='topn must be int!')
      args = parser.parse_args()

      dic = Dictionary([args.phrase], word2vec_model=model)
      phrase = args.phrase.replace(' ', '_')
      key_obj = dic.vocab()[phrase]
      if (key_obj.is_vectorized == False):
        err_msg = 'Could not vectorize phrase because of the following words: %s' % key_obj.partial_keyword_fails
        return {'error': err_msg, 'words': key_obj.partial_keyword_fails}

      vec = key_obj.vector
      return {'result': model.similar_by_vector(vec, topn=args.topn)}


class Similarity_V2(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('w1', type=str, required=True, help="Word 1 cannot be blank!")
        parser.add_argument('w2', type=str, required=True, help="Word 2 cannot be blank!")
        #parser.add_argument('topn', type=int, default=100, help='topn must be int!')

        args = parser.parse_args()

        dic = Dictionary([args.w1, args.w2], word2vec_model=model)

        # check if every word got vectorized
        # TODO: do this more elegantly
        for k,v in dic.vocab().iteritems():
          if (v.is_vectorized == False):
            # Try the facebook model
            dic = Dictionary([args.w1, args.w2], word2vec_model=facebook_model)

            for k,v in dic.vocab().iteritems():
              if (v.is_vectorized == False):
                return 0

        vecs = [v.vector for k,v in dic.vocab().iteritems()]

        similarity = dot(matutils.unitvec(vecs[0]), matutils.unitvec(vecs[1]))

        return similarity


class Similarity_batch(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('main_keyword', type=str, required=True, help="Word 1 cannot be blank!")
        parser.add_argument('keywords', type=str, action='append', required=True, help="keywwords must be a list of words!")
        #parser.add_argument('topn', type=int, default=100, help='topn must be int!')

        args = parser.parse_args()

        keywords = [args.main_keyword]
        keywords.extend(args.keywords)

        dic = Dictionary(keywords, word2vec_model=model)

        # get not vectorized keyword
        unvectorized_keywords = [key for key, key_obj in dic.vocab().iteritems() if key_obj.vector == None]
        second_unvectorized_keywords = []

        # if the main topic cannot be detected, try the facebook model immediately
        if args.main_keyword in unvectorized_keywords:
          second_dic = Dictionary(keywords, word2vec_model=facebook_model)
          second_unvectorized_keywords = [key for key, key_obj in second_dic.vocab().iteritems() if key_obj.vector == None]

        elif len(unvectorized_keywords) > 0:
          # try with facebook
          second_dic_keywords = [key for key in unvectorized_keywords]

          if args.main_keyword not in unvectorized_keywords:
            second_dic_keywords.append(args.main_keyword)

          second_dic = Dictionary(second_dic_keywords, word2vec_model=facebook_model)
          second_unvectorized_keywords = [key for key, key_obj in second_dic.vocab().iteritems() if key_obj.vector == None]

        # check here if the main topic could be vectorized, if not, return with error
        if args.main_keyword in unvectorized_keywords:
          dic_ok = False
        else:
          dic_ok = True

        if args.main_keyword in second_unvectorized_keywords:
          second_dic_ok = False
        else:
          second_dic_ok = True

        # If none of the models has the vector of the main topic, abort
        if dic_ok == False and second_dic_ok == False:
          #raise InvalidUsage('No vector could be build for the main keyword: %s' % args.main_keyword, status_code=400)
          #raise werkzeug.exceptions.NotFound('nah')
          #abort(404)
          return ('', 204)

        vecs = {}

        if dic_ok == True:
          vecs['dic'] = {}
          for key, key_obj in dic.vocab().iteritems():
            if key_obj.vector != None:
              vecs['dic'][key] = key_obj.vector


        if 'second_dic' in locals() and second_dic_ok == True:
          vecs['second_dic'] = {}
          for key, key_obj in second_dic.vocab().iteritems():
              if key_obj.vector != None:
                vecs['second_dic'][key] = key_obj.vector


        result = { 'semantic_similarity_scores': {},
                   'main_keyword': args.main_keyword,
                   'fails': second_unvectorized_keywords
                 }

        for dic, dic_data in vecs.iteritems():
          for key, vector in dic_data.iteritems():
            if key != args.main_keyword:
              sim = dot(matutils.unitvec(vector), matutils.unitvec(dic_data[args.main_keyword]))
              result['semantic_similarity_scores'][key] = sim

        return jsonify(result)


class N_Similarity(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('ws1', type=str, required=True, help="Word set 1 cannot be blank!", action='append')
        parser.add_argument('ws2', type=str, required=True, help="Word set 2 cannot be blank!", action='append')
        args = parser.parse_args()
        return model.n_similarity(filter_words(args['ws1']),filter_words(args['ws2']))


class Similarity(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('w1', type=str, required=True, help="Word 1 cannot be blank!")
        parser.add_argument('w2', type=str, required=True, help="Word 2 cannot be blank!")
        args = parser.parse_args()
        return model.similarity(args['w1'], args['w2'])


class MostSimilar(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('positive', type=str, required=False, help="Positive words.", action='append')
        parser.add_argument('negative', type=str, required=False, help="Negative words.", action='append')
        parser.add_argument('topn', type=int, required=False, help="Number of results.")
        args = parser.parse_args()
        pos = filter_words(args.get('positive', []))
        neg = filter_words(args.get('negative', []))
        t = args.get('topn', 10)
        pos = [] if pos == None else pos
        neg = [] if neg == None else neg
        t = 10 if t == None else t
        print "positive: " + str(pos) + " negative: " + str(neg) + " topn: " + str(t)
        try:
            res = model.most_similar_cosmul(positive=pos,negative=neg,topn=t)
            return res
        except Exception, e:
            print e
            print res


class Model(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('word', type=str, required=True, help="word to query.")
        args = parser.parse_args()
        try:
            res = model[args['word']]
            res = base64.b64encode(res)
            return res
        except Exception, e:
            print e
            return

class ModelWordSet(Resource):
    def get(self):
        try:
            res = base64.b64encode(cPickle.dumps(set(model.index2word)))
            return res
        except Exception, e:
            print e
            return

app = Flask(__name__)
api = Api(app)


#app.config['TRAP_HTTP_EXCEPTIONS']=True
#
#@app.errorhandler(Exception)
#def handle_error(e):
    #try:
        #if e.code == 400:
            #print '400 error code'
            #response = jsonify(error.to_dict())
            #response.status_code = error.status_code
            #return response
        #elif e.code == 404:
            #return make_error_page("Page Not Found", "The page you're looking for was not found"), 404
        #raise e
    #except:
        #print '500 error'
        #return flask.Response.force_type(e, flask.request.environ)

#@app.errorhandler(InvalidUsage)
#def handle_invalid_usage(error):
    #print 'inside handle_invalid_usage'
    #response = jsonify(error.to_dict())
    #response.status_code = error.status_code
    #return response
#
#@app.errorhandler(404)
#def pageNotFound(error):
    #return "page not found"
#
#@app.errorhandler(500)
#def raiseError(error):
    #print 'inside raiseError'
    #return error

def _handle_http_exception(error):
    print 'enters _handle_http_exception'
    if flask.request.is_json:
        return jsonify({
            'status_code': error.code,
            'message': str(error),
            'description': error.description
        }), error.code
    raise error.get_response()


# Flask fix: https://github.com/pallets/flask/issues/941
for code, ex in default_exceptions.iteritems():
    app.errorhandler(code)(_handle_http_exception)


if __name__ == '__main__':
    global model

    #app.config['TRAP_HTTP_EXCEPTIONS']=True
    #app.register_error_handler(Exception, defaultHandler)

    #----------- Parsing Arguments ---------------
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="Path to the trained model")
    p.add_argument("--second_model", help="Path to the second trained model")
    p.add_argument("--binary", help="Specifies the loaded model is binary")
    p.add_argument("--host", help="Host name (default: localhost)")
    p.add_argument("--port", help="Port (default: 5000)")
    p.add_argument("--path", help="Path (default: /word2vec)")
    args = p.parse_args()

    model_path = args.model if args.model else "./model.bin.gz"
    binary = True if args.binary else False
    host = args.host if args.host else "localhost"
    path = args.path if args.path else "/word2vec"
    port = int(args.port) if args.port else 5000
    if not args.model:
        print "Usage: word2vec-apy.py --model path/to/the/model --second_model path/to/the/model [--host host --port 1234]"

    if not args.second_model:
        print "Usage: word2vec-apy.py --model path/to/the/model --second_model path/to/the/model [--host host --port 1234]"


    model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=binary)
    #model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False) #dev
    # Load facebook model as fallback
    print 'loading facebook model...take a cup of coffee or two...really!'
    facebook_model = gensim.models.KeyedVectors.load_word2vec_format(args.second_model, binary=False)
    #facebook_model = gensim.models.KeyedVectors.load_word2vec_format(args.second_model, binary=False)  # dev

    api.add_resource(N_Similarity, path+'/n_similarity')
    api.add_resource(Similarity, path+'/similarity')
    api.add_resource(MostSimilar, path+'/most_similar')
    api.add_resource(Model, path+'/model')
    api.add_resource(ModelWordSet, '/word2vec/model_word_set')
    api.add_resource(N_SimilarWords, path + '/n_most_similar')
    api.add_resource(Similarity_V2, path+'/similarity_v2')
    api.add_resource(Similarity_batch, path+'/similarity_batch')

    app.run(host=host, port=port)
