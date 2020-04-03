import numpy as np


class ClusterFeatureVector:
    def __init__(self, alpha, beta, vocabulary):
        self.__cluster_features = []
        self.__vocabulary = vocabulary
        self.__num_total_docs = 0
        self.__alpha = alpha
        self.__beta = beta

    def sub(self, doc, cluster_id):
        """
        Subtract document `doc` from the cluster feature that has id `cluster_id`.
        """
        self.__cluster_features[cluster_id].sub(doc)
        if self.__cluster_features[cluster_id].empty():
            last_cf = self.__cluster_features.pop()
            if cluster_id != len(self.__cluster_features):
                self.__cluster_features[cluster_id] = last_cf
        self.__num_total_docs -= 1

    def sample_and_add(self, doc):
        """
        Sample the cluster to which document `doc` should belong and add the document to the cluster.
        """
        if len(self.__cluster_features) == 0:
            self.__cluster_features.append(ClusterFeature())
            self.__add(doc, 0)
            return 0

        cluster_id = self.__sample_cluster(doc)
        if cluster_id == len(self.__cluster_features):
            self.__cluster_features.append(ClusterFeature())
        self.__add(doc, cluster_id)
        return cluster_id

    def pick_max_and_add(self, doc):
        """
        Add the document `doc` to the cluster for which it has the highest probability.
        """
        cluster_id = self.__pick_cluster_max_prob(doc)
        if cluster_id == len(self.__cluster_features):
            self.__cluster_features.append(ClusterFeature())
        self.__add(doc, cluster_id)
        return cluster_id

    def __sample_cluster(self, doc):
        cluster_dist = self.__cluster_dist(doc)
        return np.random.choice(len(cluster_dist), p=cluster_dist)

    def __pick_cluster_max_prob(self, doc):
        cluster_dist = self.__cluster_dist(doc)
        return np.argmax(cluster_dist)

    def __add(self, doc, cluster_id):
        self.__cluster_features[cluster_id].add(doc)
        self.__num_total_docs += 1

    def __cluster_dist(self, doc):
        dist = np.zeros(len(self.__cluster_features) + 1)
        for i, cluster in enumerate(self.__cluster_features):
            dist[i] = _old_cluster_prob(
                doc,
                cluster,
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
        return dist * factor

    def __len__(self):
        return len(self.__cluster_features)

    def __iter__(self):
        return iter(self.__cluster_features)


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
        for word_id, freq in doc:
            if word_id not in self.__n_zw:
                self.__n_zw[word_id] = 0
            self.__n_zw[word_id] += freq
            self.__n_z += freq

    def sub(self, doc):
        """
        Subtract or remove the given document from this vector of cluster feature.
        """
        self.__m_z -= 1
        for word_id, freq in doc:
            self.__n_zw[word_id] -= freq
            self.__n_z -= freq

    def empty(self):
        return self.num_docs() == 0

    def num_docs(self):
        return self.__m_z

    def num_words(self):
        return self.__n_z

    def word_freq(self, word_id):
        if word_id not in self.__n_zw:
            return 0

        return self.__n_zw[word_id]

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
    for word_id, freq in doc:
        word_freq = cluster.word_freq(word_id)
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
    for _, freq in doc:
        for j in range(1, freq + 1):
            numerator *= beta + j - 1
    denominator = 1
    for i in range(1, len(doc) + 1):
        denominator *= vocab_size * beta + i - 1
    p_sim = numerator / denominator
    return p_cluster * p_sim
