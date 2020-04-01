import numpy as np


class IdPool:
    def __init__(self):
        self.__k = 0
        self.__free_ids = []

    def acquire(self):
        if len(self.__free_ids) != 0:
            return self.__free_ids.pop()
        else:
            self.__k += 1
            return self.__k - 1

    def release(self, id_):
        self.__free_ids.append(id_)


class ClusterFeatureVector:
    def __init__(self, alpha, beta, vocabulary):
        self.__cluster_features = {}
        self.__idpool = IdPool()
        self.__vocabulary = vocabulary
        self.__num_total_docs = 0
        self.__alpha = alpha
        self.__beta = beta

    def sub(self, doc, cluster_id):
        self.__cluster_features[cluster_id].sub(doc)
        if self.__cluster_features[cluster_id].empty():
            self.__cluster_features.pop(cluster_id)
            self.__idpool.release(cluster_id)
        self.__num_total_docs -= 1

    def sample_and_add(self, doc):
        if self.__num_total_docs == 0:
            cluster_id = self.__idpool.acquire()
        else:
            cluster_id = self.__sample_cluster(doc)
        self.__add(doc, cluster_id)
        return cluster_id

    def pick_max_and_add(self, doc):
        if self.__num_total_docs == 0:
            cluster_id = self.__idpool.acquire()
        else:
            cluster_id = self.__pick_cluster_max_prob(doc)
        self.__add(doc, cluster_id)
        return cluster_id

    def __sample_cluster(self, doc):
        values, cluster_dist = self.__cluster_dist(doc)
        # from IPython.core.debugger import set_trace; set_trace()
        sampled_cluster = np.random.choice(values, p=cluster_dist)
        if sampled_cluster != values[-1]:
            self.__idpool.release(values[-1])
        return sampled_cluster

    def __pick_cluster_max_prob(self, doc):
        values, cluster_dist = self.__cluster_dist(doc)
        max_prob_cluster = values[np.argmax(cluster_dist)]
        if max_prob_cluster != values[-1]:
            self.__idpool.release(values[-1])
        return max_prob_cluster

    def __add(self, doc, cluster_id):
        cluster_feature = self.__cluster_features.setdefault(
            cluster_id, ClusterFeature()
        )
        cluster_feature.add(doc)
        self.__num_total_docs += 1

    def __cluster_dist(self, doc):
        values = np.array(
            list(self.__cluster_features.keys()) + [self.__idpool.acquire()], dtype=int,
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

    def __iter__(self):
        return iter(self.__cluster_features.items())


class ClusterFeature:
    def __init__(self):
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
        self.__m_z += 1
        for w, freq in doc:
            if w not in self.__n_zw:
                self.__n_zw[w] = 0
            self.__n_zw[w] += freq
            self.__n_z += freq

    def sub(self, doc):
        """
        Subtract or remove the given document from this vector of cluster feature.
        """
        self.__m_z -= 1
        for w, freq in doc:
            self.__n_zw[w] -= freq
            self.__n_z -= freq

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

    def num_words_with_rep(self):
        return sum(self.__n_zw.values())

    def __iter__(self):
        return iter(self.__n_zw.items())

    def __str__(self):
        return f"word_freq: {self.__n_zw}; num_docs: {self.__m_z}; num_totaal_words: {self.__n_z}"


def _old_cluster_prob(doc, cluster, vocab_size, num_total_docs, alpha, beta):
    """
    Probability of a document being part an existing cluster.
    """
    p_cluster = cluster.num_docs() / (num_total_docs - 1 + alpha * num_total_docs)
    numerator = 1
    for w, freq in doc:
        word_freq = cluster.word_freq(w)
        for j in range(1, freq + 1):
            numerator *= word_freq + beta + j - 1
    denominator = 1
    for i in range(1, len(doc) + 1):
        denominator *= cluster.num_words() + vocab_size * beta + i - 1
    p_sim = numerator / denominator
    return p_cluster * p_sim


def _new_cluster_prob(doc, vocab_size, num_total_docs, alpha, beta):
    """
    Probability of a document being part of a new cluster.
    """
    p_cluster = alpha * num_total_docs / (num_total_docs - 1 + alpha * num_total_docs)
    numerator = 1
    for w, freq in doc:
        for j in range(1, freq + 1):
            numerator *= beta + j - 1
    denominator = 1
    for i in range(1, len(doc) + 1):
        denominator *= vocab_size * beta + i - 1
    p_sim = numerator / denominator
    return p_cluster * p_sim
