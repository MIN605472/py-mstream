from mstream.clusterfeature import ClusterFeatureVector
from mstream.vocabulary import Vocabulary
from mstream.document import Document
import numpy as np


class Mstream:
    def __init__(self, filename):
        """
        Initialize the model by reading it from the specified file.
        """
        pass

    def __init__(self, num_iter=10, alpha=0.03, beta=0.03):
        self.vocabulary = Vocabulary()
        self.__num_iter = num_iter
        self.__alpha = alpha
        self.__beta = beta
        self.__clusters = ClusterFeatureVector(alpha, beta, self.vocabulary)

    def process(self, batch):
        doc_batch = []
        for doc in batch:
            doc_batch.append(Document(doc["id"], doc["terms"], self.vocabulary))
        # from IPython.core.debugger import set_trace; set_trace()
        self.__first_pass(doc_batch)
        self.__gibbs_sample(doc_batch)
        self.__pick_max_prob(doc_batch)
        return [doc.topic.id for doc in doc_batch]

    def num_topics(self):
        return len(self.__clusters)

    def topic_pmf(self):
        pmf = np.zeros(len(self.__clusters))
        for i, cluster in enumerate(self.__clusters):
            pmf[i] = cluster.num_docs()
        return pmf / pmf.sum()

    def topic_term_pmf(self):
        pmf = np.zeros((len(self.__clusters), len(self.vocabulary)))
        topic_wordid_dist = self.__estimate_posterior()
        for i, wordid_prob in enumerate(topic_wordid_dist):
            for wordid, prob in wordid_prob.items():
                pmf[i, wordid] = prob
        return pmf / pmf.sum(axis=1)[:, None]

    def get_top_terms(self, num_terms_topic=50):
        top_terms_topic = {}
        topic_word_dist = self.__estimate_posterior()
        for topic, word_prob in enumerate(topic_word_dist):
            l = list(word_prob.items())
            l.sort(key=lambda x: x[1])
            total = sum([p for _, p in l])
            top_terms = l[:num_terms_topic]
            top_terms = [(self.vocabulary.id2term(w), p / total) for w, p in top_terms]
            top_terms_topic[topic] = dict(top_terms)
        return top_terms_topic

    def save_model(self):
        pass

    def __first_pass(self, docs):
        for doc in docs:
            doc.topic = self.__clusters.sample_and_add(doc)

    def __gibbs_sample(self, docs):
        for i in range(self.__num_iter - 1):
            for doc in docs:
                self.__clusters.sub(doc)
                doc.topic = self.__clusters.sample_and_add(doc)

    def __pick_max_prob(self, docs):
        for doc in docs:
            self.__clusters.sub(doc)
            doc.topic = self.__clusters.pick_max_and_add(doc)

    def __estimate_posterior(self):
        phi_zw = []
        for i, cluster in enumerate(self.__clusters):
            phi_zw[i] = {}
            s = cluster.num_terms_with_rep()
            for word_id, freq in cluster:
                if freq != 0:
                    phi_zw[i][word_id] = (freq + self.__beta) / (
                        s + len(self.vocabulary) * self.__beta
                    )
        return phi_zw
