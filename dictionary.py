# Dictionary was copied from the cluser-api and is a total overkill
# but we reuse it to save time

from __future__ import division
import numpy as np
from collections import Mapping
# from sklearn.preprocessing import StandardScaler
from itertools import combinations
import kw
reload(kw)
from kw import Keyword
# from annoy import AnnoyIndex
from pprint import pprint
import spacy
import re
from operator import itemgetter


nlp = spacy.en.English()

class Dictionary(Mapping):
  def __init__(self, keywords, word2vec_model=None,  adjective_weights=False, batch_size=90000):
    assert type(keywords) is list, 'keywords must be a list of keyword strings'

    keywords = [re.sub(' +',' ', key).replace(" ", "_") for key in keywords]

    self.nlp = nlp
    self.batch_size = batch_size
    self.total_population = len(keywords)
    # Remove duplicates
    self.unique_keywords = list(set(keywords))
    self.direct_hits_counter = 0
    self.total_population_after_removing_duplicates = len(self.unique_keywords)

    self.id2keyword = {}
    self.str2keyword = {}

    self.word2vec_model = word2vec_model
    #self.stds = StandardScaler()
    self.adjective_weights = adjective_weights

    # debug
    self.vectors = []

    self.add_keywords(self.unique_keywords)
    self.vectorized_keywords_population = self.number_of_vectorized_keywords()
    self.print_population_stats()


  def __getitem__(self, keyword):
    return self.str2keyword[keyword]

  def __iter__(self):
    return iter(self.str2keyword.keys())

  def __len__(self):
    return len(self.id2keyword)

  def vocab(self):
    return self.str2keyword


  def mark_as_clustered(self, strings):
    for string in strings:
      self.str2keyword[string].is_clustered = True

  def mark_as_not_clustered(self, strings):
    for string in strings:
      self.str2keyword[string].is_clustered = False


  def number_of_unclustered_keyword(self):
    counter = 0
    for idx, keyword in self.id2keyword.iteritems():
      if(keyword.is_vectorized == True and keyword.is_clustered == False):
        counter += 1
    return counter

  # Convenience method for input to clustering.find_best_eps_by_cluster_distribution
  # and clustering.find_best_eps_by_rating
  def unclustered_key_vecs(self):
    keys = []
    vecs = []
    for idx, keyword in self.id2keyword.iteritems():
      if (keyword.is_vectorized == True and keyword.is_clustered == False):
        keys.append(keyword.keyword)
        vecs.append(keyword.vector)
    return keys, self.stds.fit_transform(vecs)

  def prepare_for_reclustering(self, keywords):
    vectors = []
    for keyword in keywords:
      self.str2keyword[keyword].is_clustered = False
      vectors.append(self.str2keyword[keyword].vector)
    return keywords, self.stds.fit_transform(vectors)


  def number_of_vectorized_keywords(self):
    ans = []
    for idx, keyword in self.id2keyword.iteritems():
      if (keyword.is_vectorized == True):
        ans.append(keyword)
    return len(ans)


  def print_population_stats(self):
    print 'Total keywords: %i' % self.total_population
    print 'Total keywords after removing duplicates: %i' % self.total_population_after_removing_duplicates
    print 'Total keywords: %i' % self.number_of_vectorized_keywords()
    print 'Total direct hits: %i' % self.direct_hits_counter
    print 'Total keywords to cluster: %i' % self.number_of_unclustered_keyword()


  def add_keywords(self, keywords):
    # First ceate all vectors, so that we can standardize the vectors later
    # for idx, keyword in enumerate(keywords):
      # vector, success = self.create_vector(keyword)
      # if (success == True):
        # self.vectors.append(vector)

    # Compute the mean and std to be used for later scaling
    #print 'old way'
    #print self.stds.fit_transform(self.vectors)[0][0].__class__
    #print self.stds.fit_transform(self.vectors)[0][0]
    #print 'new way'
    #print self.stds.fit(self.vectors).transform(self.vectors)[0][0].__class__
    #print self.stds.fit(self.vectors).transform(self.vectors)[0][0]
    #print self.stds.fit(self.vectors).transform([self.vectors[0]])[0][0]
    # self.stds.fit(self.vectors)

    self.fails = []
    for idx, keyword in enumerate(keywords):
      #print idx
      vector, success, direct_hit, partial_keywords, partial_fails = self.create_vector(keyword)
      if (success != True):
        vector = None
        self.fails.append(keyword)

        #print 'test_stds'
        #print test_stds.transform(vector)[0].__class__
        #print 'no reshape'
        #print(self.stds.transform(vector)[0].__class__)
        #print(np.float64(self.stds.transform(vector.reshape(1,-1))[0][0]).__class__)
        #print self.stds.transform([vector])[0][0].__class__
        #print self.stds.transform([vector])[0][0]

      keyword_obj = Keyword(keyword, vector, partial_keywords, direct_hit, partial_fails, is_vectorized=success)
      self.id2keyword[idx] = keyword_obj
      self.str2keyword[keyword] = keyword_obj
    #pprint(self.fails)


  # Returns vectors, success (Boolean)
  def create_vector(self, keyword):
    #print keyword
    #print self.word2vec_model.vocab[keyword]
    #self.stds.fit_transform(self.word2vec_model.vocab[keyword])
    direct_hit = False
    partial_keywords = {}
    fails = []
    try:
      #print 'direct'
      #print self.word2vec_model[keyword][0].__class__
      #print self.word2vec_model[keyword][0]
      vector = self.word2vec_model[keyword]
      partial_keywords[keyword] = vector
      direct_hit = True
      self.direct_hits_counter += 1
      return vector, True, direct_hit, partial_keywords
    except KeyError:
      try:
        # adjective_weights is broken and should be removed
        if (self.adjective_weights != False):
          uni_vectors = []
          unigrams = keyword.split('_')
          tagged_unigrams = nltk.pos_tag(unigrams)
          for uni_tag in tagged_unigrams:
            self.word2vec_model.vocab[uni_tag[0]]
            vec = self.word2vec_model[uni_tag[0]]
            if (uni_tag[1] in ['JJ', 'VBN', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']):
              vec = vec * adjective_weights
            uni_vectors.append(vec)

          vector = sum(uni_vectors) / len(tagged_unigrams)
          return vector, True
        else:
          unigrams = keyword.split('_')
          keyword_length = len(unigrams)
          repeat = keyword_length
          vecs = {}
          while (repeat > 0):
            for grams in combinations(unigrams, r=repeat):
              # only accept adjacent combinations
              if ('_'.join(grams) in keyword):
                try:
                  idx_string = '_'.join([str(unigrams.index(uni)) for uni in grams])
                  vecs[idx_string] = self.word2vec_model['_'.join(grams)]
                except KeyError:
                  try:
                    print grams
                    doc = self.nlp( unicode(' '.join(grams)) )
                    vecs[idx_string] = self.word2vec_model[str(doc.sents.next().lemma_)]
                    #print 'success lemma'

                    #{
                      #'vector': google[str(doc.sents.next().lemma_).replace(' ', '_')],
                      #'ngram': '_'.join(grams),
                      #'count': google.vocab[str(doc.sents.next().lemma_).replace(' ', '_')].count
                    #}
                  except KeyError:
                    if (repeat == 1):
                      #print 'grams: %s' % grams
                      fails.append(grams[0])
                    pass

                  except UnicodeDecodeError:
                    print grams
            repeat = repeat - 1

          # check if there was a vector found for each word
          if (keyword_length != len(list(set([int(idx) for idx_string in vecs.keys() for idx in idx_string.split('_')])))):
            return None, False, direct_hit, partial_keywords, fails

          # Calculate weighted average of vectors
          # start with longest ngrams found
          keys = vecs.keys()
          keys.sort(key = lambda x: len(x.split('_')), reverse=True)
          final_vecs = []
          vec_weights = []
          taken_indices = []
          total_len = len(unigrams)
          single_weight = 1 / total_len
          for key in keys:
            indices = key.split('_')
            int_indices  = [int(idx) for idx in indices]
            if (len(list(set(indices) & set(taken_indices))) == 0):
              final_vecs.append(vecs[key])
              vec_weights.append(single_weight * len(indices))
              taken_indices = taken_indices + indices
              #print 'keyword: %s' % keyword
              #print 'int_indices: %s' % int_indices
              #print itemgetter(*int_indices)(keyword.split('_'))
              #print '_'.join(itemgetter(*int_indices)(keyword.split('_')))
              partial_keyword = itemgetter(*int_indices)(keyword.split('_'))
              partial_keywords[partial_keyword] = vecs[key]

          weighted_middle_vector = np.average(np.array(final_vecs), axis=0, weights=vec_weights)
          return weighted_middle_vector, True, direct_hit, partial_keywords, fails
      except KeyError:
        return None, False, direct_hit, partial_keywords, fails