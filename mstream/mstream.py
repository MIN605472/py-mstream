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
            doc_batch.append(Document(doc["id"], doc["words"], self.vocabulary))
        # from IPython.core.debugger import set_trace; set_trace()
        self.__first_pass(doc_batch)
        self.__gibbs_sample(doc_batch)
        self.__pick_max_prob(doc_batch)
        return [doc.cluster_id for doc in doc_batch]

    def topics(self):
        topic_ids = np.zeros(len(self.__clusters), dtype=int)
        for i, (cluster_id, _) in enumerate(self.__clusters):
            topic_ids[i] = cluster_id
        return topic_ids

    def num_topics(self):
        return len(self.__clusters)

    def topic_pmf(self):
        pmf = np.zeros(len(self.__clusters))
        for i, (cluster_id, cluster) in enumerate(self.__clusters):
            pmf[i] = cluster.num_docs()
        return pmf / pmf.sum()

    def topic_term_pmf(self):
        pmf = np.zeros((len(self.__clusters), len(self.vocabulary)))
        topic_wordid_dist = self.__estimate_posterior()
        for i, (topic, wordid_prob) in enumerate(topic_wordid_dist.items()):
            for wordid, prob in wordid_prob.items():
                pmf[i, wordid] = prob
        return pmf / pmf.sum(axis=1)[:, None]

    def get_top_words(self, num_words_topic=50):
        top_words_topic = {}
        topic_word_dist = self.__estimate_posterior()
        for topic, word_prob in topic_word_dist.items():
            l = list(word_prob.items())
            l.sort(key=lambda x: x[1])
            total = sum([p for _, p in l])
            top_words = l[:num_words_topic]
            top_words = [(self.vocabulary.id2word(w), p / total) for w, p in top_words]
            top_words_topic[topic] = dict(top_words)
        return top_words_topic

    def save_model(self):
        pass

    def __first_pass(self, docs):
        for doc in docs:
            doc.cluster_id = self.__clusters.sample_and_add(doc)

    def __gibbs_sample(self, docs):
        for i in range(self.__num_iter - 1):
            for doc in docs:
                self.__clusters.sub(doc, doc.cluster_id)
                doc.cluster_id = self.__clusters.sample_and_add(doc)

    def __pick_max_prob(self, docs):
        for doc in docs:
            self.__clusters.sub(doc, doc.cluster_id)
            doc.cluster_id = self.__clusters.pick_max_and_add(doc)

    def __estimate_posterior(self):
        phi_zw = {}
        for cluster_id, cluster in self.__clusters:
            phi_zw[cluster_id] = {}
            s = cluster.num_words_with_rep()
            for word_id, freq in cluster:
                if freq != 0:
                    phi_zw[cluster_id][word_id] = (freq + self.__beta) / (
                        s + len(self.vocabulary) * self.__beta
                    )
        return phi_zw
