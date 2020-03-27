class Dictionary:
    def __init__(self):
        pass

class Document:
    def __init__(self):
        pass

class ClusterFeatureVector:
    def __init__(self):
        self.cluster_features = []

    def add(self):
        self.cluster_features = []



class ClusterFeature:
    def __init__(self):
        # cluster id
        self.__id = 0 
        # frequency of the word 'v' in cluster 'z'
        self.__n_zv = {}
        # number of documents in cluster 'z'
        self.__m_z = 0
        # number of words in cluster 'z'
        self.__n_z = 0

    def add(self, doc):
        """
        Add the given document to this vector of cluster feature.
        """
        pass

    def sub(self, doc):
        """
        Subtract or remove the given document from this vector of cluster feature.
        """
        pass


class MStream:
    def __init__(self, filename):
        """
        Initialize the model by reading it from the specified file.
        """
        pass


    def __init__(self, dictionary, iter_num, alpha0, beta, words_topic_num):
        self.__K = 0
        self.__dictionary = dictionary
        self.__iter_num = iter_num
        self.__alpha0 = alpha0
        self.__beta = beta
        self.__words_topic_num = words_topic_num

    def __first_pass(self)


    def process(self, batch):
        first_pass()
        gibbs_sample()


    def save_model(self):
        pass

    def __gibbs_sample(self):
        pass


    def __sample_cluster(self):
        pass

    def __estimate_posterior(self):
        pass

    def __beta0(self):
        pass
