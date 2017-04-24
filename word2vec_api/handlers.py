from flask.ext.restful import Resource, reqparse
from flask import jsonify
import logging
import gensim
from dictionary import Dictionary
from IPython import embed
from gensim import utils, matutils
from numpy import exp, dot, zeros, outer, random, dtype, get_include, float32 as REAL,\
     uint32, seterr, array, uint8, vstack, argsort, fromstring, sqrt, newaxis, ndarray, empty, sum as np_sum


# create different loggers
formatter = logging.Formatter('%(message)s')

failedKeywordLogger = logging.getLogger('failed_keywords')
failedKeywordLoggerHandler = logging.FileHandler('/home/ubuntu/logs/failed_keywords.log', mode='w')
failedKeywordLoggerHandler.setFormatter(formatter)
failedKeywordLogger.setLevel(logging.DEBUG)
failedKeywordLogger.addHandler(failedKeywordLoggerHandler)

failedMainKeywordLogger = logging.getLogger('failed_main_keywords')
failedMainKeywordLoggerHandler = logging.FileHandler('/home/ubuntu/logs/failed_main_keywords.log', mode='w')
failedMainKeywordLoggerHandler.setFormatter(formatter)
failedMainKeywordLogger.setLevel(logging.DEBUG)
failedMainKeywordLogger.addHandler(failedMainKeywordLoggerHandler)


# use for ative development, since the other models take over 20 min to load
# model_path = '/home/ubuntu/data/models/glove.6B.50d.txt'
# second_model = '/home/ubuntu/data/models/glove.6B.50d.txt'

model_path = '/home/ubuntu/data/models/GoogleNews-vectors-negative300.bin'
second_model = '/home/ubuntu/data/models/wiki.en.vec'


model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
#model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False) #dev
# Load facebook model as fallback
print 'loading facebook model...take a cup of coffee or two...really!'
facebook_model = gensim.models.KeyedVectors.load_word2vec_format(second_model, binary=False)
#facebook_model = gensim.models.KeyedVectors.load_word2vec_format(second_model, binary=False)  # dev




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


class Get_vectors(Resource):
    def post(self):
      parser = reqparse.RequestParser()
      parser.add_argument('keywords', type=str, action='append', required=True, help="keywwords must be a list of words!")
      args = parser.parse_args()
      result = {}
      for key in args.keywords:
        try:
          result[key] = model[key].tolist()
        except KeyError:
          result[key] = None

      return result



class Similarity_batch(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('main_keyword', type=str, required=True, help="Word 1 cannot be blank!")
        parser.add_argument('keywords', type=str, action='append', required=True, help="keywwords must be a list of words!")
        #parser.add_argument('topn', type=int, default=100, help='topn must be int!')

        args = parser.parse_args()

        keywords = [args.main_keyword]
        keywords.extend(args.keywords)

        # check if main_keyword is also inside the keywords
        addMainKeywordToSimMap = False
        if args.main_keyword in args.keywords:
          addMainKeywordToSimMap = True

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
          failedMainKeywordLogger.debug(args.main_keyword)
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


        second_unvectorized_keywords.extend(unvectorized_keywords)

        second_unvectorized_keywords = list(set(second_unvectorized_keywords))

        result = { 'semantic_similarity_scores': {},
                   'main_keyword': args.main_keyword
                 }

        for _, dic_data in vecs.iteritems():
          for key, vector in dic_data.iteritems():
            if key != args.main_keyword:
              sim = dot(matutils.unitvec(vector), matutils.unitvec(dic_data[args.main_keyword]))
              result['semantic_similarity_scores'][key] = sim

        # Add main_keyword if it was found in the keywords field
        if addMainKeywordToSimMap == True:
          result['semantic_similarity_scores'][args.main_keyword] = 1.0


        for fail in second_unvectorized_keywords:
          failedKeywordLogger.debug(fail)
          result['semantic_similarity_scores'][fail] = 0.0
        return result


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
