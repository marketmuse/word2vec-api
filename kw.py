class Keyword():
  def __init__(self, keyword, vector, partial_keywords, direct_hit, is_vectorized=False):
    assert type(keyword) is str, 'keyword must be a string'
    self.keyword = keyword
    self.vector = vector
    self.is_vectorized = is_vectorized
    self.is_clustered = False
    self.partial_keywords = partial_keywords
    self.direct_hit = direct_hit