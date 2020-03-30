import numpy as np


class ClusterFeatureVector:
    def __init__(self, alpha, beta, vocabulary):
        self.__cluster_features = {}
        self.__ids = []
        self.__K = 0
        self.__vocabulary = vocabulary
        self.__num_total_docs = 0
        self.__alpha = alpha
        self.__beta = beta

    def sub(self, doc, cluster_id):
        self.__cluster_features[cluster_id].sub(doc)
        if self.__cluster_features[cluster_id].empty():
            self.__cluster_features[cluster_id].pop(cluster_id)
            self.__ids.append(cluster_id)
        self.__num_total_docs -= 1

    def sample_and_add(self, doc):
        if self.__num_total_docs == 0:
            self.__add(doc, self.__get_new_cluster_id())
            return

        cluster_id = self.__sample_cluster(doc)
        self.__add(doc, cluster_id)
        return cluster_id

    def __sample_cluster(self, doc):
        values, cluster_dist = self.__cluster_dist(doc)
        sampled_cluster = np.random.choice(values, p=cluster_dist)
        return sampled_cluster

    def __add(self, doc, cluster_id):
        cluster_feature = self.__cluster_features.setdefault(
            cluster_id, ClusterFeature()
        )
        cluster_feature.add(doc)
        self.__num_total_docs += 1

    def __get_new_cluster_id(self):
        if len(self.__ids) == 0:
            self.__K += 1
            return self.__K - 1
        else:
            return self.__ids[0]

    def __cluster_dist(self, doc):
        values = np.array(
            list(self.__cluster_features.keys()) + [self.__get_new_cluster_id()],
            dtype=int,
        )
        dist = np.zeros(len(self.__cluster_features) + 1)
        for i, cluster_id in enumerate(self.__cluster_features):
            dist[i] = _old_cluster_prob(
                doc,
                self.__cluster_features[cluster_id],
                len(self.__vocabulary),
                self.__num_total_docs,
                self.__alpha,
                self.__beta,
            )
        dist[len(self.__cluster_features)] = _new_cluster_prob(
            doc,
            len(self.__vocabulary),
            self.__num_total_docs,
            self.__alpha,
            self.__beta,
        )
        factor = 1.0 / np.sum(dist)
        return values, dist * factor


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
        for w, f in doc:
            if w not in self.__n_zw:
                self.__n_zw[w] = 0
            self.__n_zw[w] += f
        self.__m_z += 1
        self.__n_z += len(doc)

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
        if word not in self.__n_zw:
            return 0
        
        return self.__n_zw[word]


def _old_cluster_prob(doc, cluster, vocab_size, num_total_docs, alpha, beta):
    """
    Probability of a document being part an existing cluster.
    """
    p_cluster = cluster.num_docs() / (num_total_docs - 1 + alpha * num_total_docs)
    for w, freq in doc:
        numerator = 1
        for j in range(1, freq + 1):
            numerator *= cluster.word_freq(w) + beta + j - 1
    denominator = 1
    for i in range(1, len(doc) + 1):
        denominator *= cluster.num_words() + vocab_size * beta + i - 1
    p_sim = numerator / denominator
    return p_cluster * p_sim


def _new_cluster_prob(doc, V, D, alpha, beta):
    """
    Probability of a document being part of a new cluster.
    """
    p_cluster = alpha * D / (D - 1 + alpha * D)
    numerator = 1
    for w, freq in doc:
        for j in range(1, freq + 1):
            numerator *= beta + j - 1
    denominator = 1
    for i in range(1, len(doc) + 1):
        denominator *= V * beta + i - 1
    p_sim = numerator / denominator
    return p_cluster * p_sim
