import numpy as np


class ClusterFeatureVector:
    def __init__(self, alpha, beta, vocabulary):
        # cfs = cluster features
        self.__cfs = []
        self.__vocab = vocabulary
        self.__num_total_docs = 0
        self.__alpha = alpha
        self.__beta = beta

    def sub(self, doc):
        """
        Subtract document `doc` from the cluster feature that has id `cluster_id`.
        """
        cluster_id = doc.topic.id
        self.__cfs[cluster_id].sub(doc)
        if self.__cfs[cluster_id].empty():
            cf = self.__cfs.pop()
            if cluster_id != len(self.__cfs):
                cf.id = cluster_id
                self.__cfs[cluster_id] = cf
        self.__num_total_docs -= 1

    def sample_and_add(self, doc):
        """
        Sample the cluster to which document `doc` should belong and add the document to the cluster.
        """
        cluster_id = self.__sample_cluster(doc)
        self.__add(doc, cluster_id)
        return self.__cfs[cluster_id]

    def pick_max_and_add(self, doc):
        """
        Add the document `doc` to the cluster for which it has the highest probability.
        """
        cluster_id = self.__pick_cluster_max_prob(doc)
        self.__add(doc, cluster_id)
        return self.__cfs[cluster_id]

    def __sample_cluster(self, doc):
        cluster_pmf = self.__cluster_pmf(doc)
        return np.random.choice(len(cluster_pmf), p=cluster_pmf)

    def __pick_cluster_max_prob(self, doc):
        cluster_pmf = self.__cluster_pmf(doc)
        return np.argmax(cluster_pmf).item()

    def __add(self, doc, cluster_id):
        assert cluster_id <= len(self.__cfs)
        if cluster_id == len(self.__cfs):
            self.__cfs.append(ClusterFeature(cluster_id))
        self.__cfs[cluster_id].add(doc)
        self.__num_total_docs += 1

    def __cluster_pmf(self, doc):
        pmf = np.zeros(len(self.__cfs) + 1)
        for i, cf in enumerate(self.__cfs):
            pmf[i] = _old_cluster_prob(
                doc,
                cf,
                len(self.__vocab),
                self.__num_total_docs + 1,
                self.__alpha,
                self.__beta,
            )
        pmf[len(self.__cfs)] = _new_cluster_prob(
            doc,
            len(self.__vocab),
            self.__num_total_docs + 1,
            self.__alpha,
            self.__beta,
        )
        norm_factor = 1.0 / np.sum(pmf)
        return pmf * norm_factor

    def __len__(self):
        return len(self.__cfs)

    def __iter__(self):
        return iter(self.__cfs)


class ClusterFeature:
    def __init__(self, id_):
        self.id = id_
        # frequency of the term 'w' in cluster 'z'
        self.__n_zw = {}
        # number of documents in cluster 'z'
        self.__m_z = 0
        # number of terms in cluster 'z'
        self.__n_z = 0

    def add(self, doc):
        """
        Add the given document to this vector of cluster feature.
        """
        self.__m_z += 1
        for term_id, freq in doc:
            if term_id not in self.__n_zw:
                self.__n_zw[term_id] = 0
            self.__n_zw[term_id] += freq
            self.__n_z += freq

    def sub(self, doc):
        """
        Subtract or remove the given document from this vector of cluster feature.
        """
        self.__m_z -= 1
        for term_id, freq in doc:
            self.__n_zw[term_id] -= freq
            self.__n_z -= freq

    def empty(self):
        return self.num_docs() == 0

    def num_docs(self):
        return self.__m_z

    def num_terms(self):
        return self.__n_z

    def term_freq(self, term_id):
        if term_id not in self.__n_zw:
            return 0

        return self.__n_zw[term_id]

    def num_terms_with_rep(self):
        return sum(self.__n_zw.values())

    def __iter__(self):
        return iter(self.__n_zw.items())

    def __str__(self):
        return f"term_freq: {self.__n_zw}; num_docs: {self.__m_z}; num_total_terms: {self.__n_z}"


def _old_cluster_prob(doc, cluster, vocab_size, num_total_docs, alpha, beta):
    """
    Unormalized robability of document `doc` being part the existing cluster `cluster`.
    """
    # we can remove the denominator because it's a constant; doesn't affect proportionality
    p_cluster = cluster.num_docs() # / (num_total_docs - 1 + alpha * num_total_docs)
    numerator = 1
    for term_id, freq in doc:
        term_freq = cluster.term_freq(term_id)
        for j in range(freq):
            # numerator *= term_freq + beta + j - 1
            numerator *= term_freq + beta + j
    denominator = 1
    for i in range(doc.total_len()):
        # denominator *= cluster.num_terms() + vocab_size * beta + i - 1
        denominator *= cluster.num_terms() + vocab_size * beta + i
    p_sim = numerator / denominator
    return p_cluster * p_sim


def _new_cluster_prob(doc, vocab_size, num_total_docs, alpha, beta):
    """
    Unormalized probability of document `doc` being part of a new cluster.
    """
    # we can remove the denominator because it's a constant; doesn't affect proportionality
    p_cluster = alpha * num_total_docs # / (num_total_docs - 1 + alpha * num_total_docs)
    numerator = 1
    for _, freq in doc:
        for j in range(freq):
            # numerator *= beta + j - 1
            numerator *= beta + j
    denominator = 1
    for i in range(doc.total_len()):
        denominator *= vocab_size * beta + i
    p_sim = numerator / denominator
    return p_cluster * p_sim
