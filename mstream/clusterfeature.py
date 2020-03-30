import numpy as np

class ClusterFeatureVector:
    def __init__(self):
        self.__cluster_features = {}
        self.__ids = []
        self.__K = 0

    def add(self, doc, cluster_id):
        self.__cluster_features[cluster_id].add(doc)

    def sub(self, doc, cluster_id):
        self.__cluster_features[cluster_id].sub(doc)
        if self.__cluster_features[cluster_id].empty():
            self.__cluster_features[cluster_id].pop(cluster_id)
            self.__ids.append(cluster_id)

    def sample_and_add(self, doc):
        cluster_id = self.sample_cluster(doc)
        self.add(doc, cluster_id)
        return cluster_id

    def sample_cluster(self, doc):
        values, cluster_dist = self.__cluster_dist(doc)
        sampled_cluster = np.random.choice(values, p=cluster_dist)
        return sampled_cluster

    def __get_new_cluster_id(self):
        if len(self.__ids) == 0:
            self.K = K + 1
            return self.K - 1
        else:
            return self.__ids[0]

    def __cluster_dist(self, doc):
        values = np.array(
            list(self.__cluster_features.keys) + [self.__get_new_cluster_id()], dtype=int
        )
        dist = np.zeros(len(self.__clusters) + 1)
        for i, cluster in enumerate(self.__clusters):
            dist[i] = _old_cluster_prob(
                doc, cluster, len(self.__vocabulary), D, self.__alpha, self.__beta
            )
        dist[len(self.clusters)] = __new_cluster_prob(
            doc, len(self.__vocabulary), D, self.__alpha, self.__beta
        )
        return values, dist


class ClusterFeature:
    def __init__(self):
        # cluster id
        self.__id = 0
        # frequency of the word 'w' in cluster 'z'
        self.__n_zw = {}
        # number of documents in cluster 'z'
        self.__m_z = 0
        # number of words in cluster 'z'
        self.__n_z = 0

    def add(self, doc):
        """
        Add the given document to this vector of cluster feature.
        """
        for w in doc:
            self.n_zw += doc.a
        self.__m_z += 1
        self.__n_z += doc.n

    def sub(self, doc):
        """
        Subtract or remove the given document from this vector of cluster feature.
        """
        for w in doc:
            self.__n_zw -= doc.w
        self.__m_z -= 1
        self.__n_z -= doc.n

    def empty(self):
        return self.num_docs() == 0

    def num_docs(self):
        return self.__m_z

    def num_words(self):
        return self.__n_z

    def word_freq(self, word):
        return self.__n_zw[word]



def _old_cluster_prob(doc, cluster, V, D, alpha, beta):
    """
    Probability of a document being part an existing cluster.
    """
    p_cluster = cluster.m() / (D - 1 + alpha * D)
    for w, freq in doc:
        numerator = 1
        for j in range(freq):
            numerator *= cluster.n_v(w) + beta + j - 1
        denominator = 1
        for i in range(len(doc)):
            denominator *= cluster.n() + V * beta + i - 1
    p_sim = numerator / denominator
    return p_cluster * p_sim


def _new_cluster_prob(doc, V, D, alpha, beta):
    """
    Probability of a document being part of a new cluster.
    """
    p_cluster = alpha * D / (D - 1 + alpha * D)
    numerator = 1
    for w, freq in doc:
        for j in range(freq):
            numerator *= beta + j - 1
    denominator = 1
    for i in range(len(doc)):
        denominator *= V * beta + i - 1
    p_sim = numerator / denominator
    return p_cluster * p_sim
