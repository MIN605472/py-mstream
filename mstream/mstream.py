from mstream.clusterfeature import ClusterFeatureVector
from mstream.vocabulary import Vocabulary
from mstream.document import Document
import numpy as np


class Mstream:
    # TODO: impl
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
        self.__topics = ClusterFeatureVector(alpha, beta, self.vocabulary)

    def update(self, batch):
        doc_batch = []
        for doc in batch:
            doc_batch.append(Document(doc["id"], doc["terms"], self.vocabulary))
        self.__do_first_pass(doc_batch)
        if self.__num_iter > 1:
            self.__gibbs_sample_topic(doc_batch)
            self.__pick_topic_max_prob(doc_batch)
        return [doc.topic.id for doc in doc_batch]

    def num_topics(self):
        return len(self.__topics)

    def topic_pmf(self):
        pmf = np.zeros(len(self.__topics))
        for i, cluster in enumerate(self.__topics):
            pmf[i] = cluster.num_docs()
        return pmf / pmf.sum()

    def topic_term_pmf(self):
        pmf = np.zeros((len(self.__topics), len(self.vocabulary)))
        topic_termid_dist = self.__topic_term_posterior()
        for i, termid_prob in enumerate(topic_termid_dist):
            for termid, prob in termid_prob.items():
                pmf[i, termid] = prob
        return pmf / pmf.sum(axis=1)[:, None]

    def top_terms(self, num_terms_topic=50):
        top_terms_topic = {}
        topic_term_pmf = self.__topic_term_posterior()
        for topic, term_prob in enumerate(topic_term_pmf):
            l = list(term_prob.items())
            l.sort(key=lambda x: x[1])
            total = sum([p for _, p in l])
            top_terms = l[:num_terms_topic]
            top_terms = [(self.vocabulary.id2term(t), p / total) for t, p in top_terms]
            top_terms_topic[topic] = dict(top_terms)
        return top_terms_topic

    # TODO: impl
    def save_model(self):
        """
        Save the model to a file for later use.
        """
        pass

    def __do_first_pass(self, docs):
        for doc in docs:
            doc.topic = self.__topics.sample_and_add(doc)

    def __gibbs_sample_topic(self, docs):
        for i in range(1, self.__num_iter - 1):
            for doc in docs:
                self.__topics.sub(doc)
                doc.topic = self.__topics.sample_and_add(doc)

    def __pick_topic_max_prob(self, docs):
        for doc in docs:
            self.__topics.sub(doc)
            doc.topic = self.__topics.pick_max_and_add(doc)

    def __topic_term_posterior(self):
        posterior = [None] * len(self.__topics)
        for i, cluster in enumerate(self.__topics):
            posterior[i] = {}
            s = cluster.num_terms_with_rep()
            for term_id, freq in cluster:
                if freq != 0:
                    posterior[i][term_id] = (freq + self.__beta) / (
                        s + len(self.vocabulary) * self.__beta
                    )
        return posterior
