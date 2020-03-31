from mstream.clusterfeature import ClusterFeatureVector
from mstream.vocabulary import Vocabulary
from mstream.document import Document


class Mstream:
    def __init__(self, filename):
        """
        Initialize the model by reading it from the specified file.
        """
        pass

    def __init__(self, num_iter=10, alpha=0.03, beta=0.03, num_words_topic=50):
        self.__vocabulary = Vocabulary()
        self.__num_iter = num_iter
        self.__alpha = alpha
        self.__beta = beta
        self.__num_words_topic = num_words_topic
        self.__clusters = ClusterFeatureVector(alpha, beta, self.__vocabulary)

    def process(self, batch):
        doc_batch = []
        for doc in batch:
            doc_batch.append(Document(doc["id"], doc["word_list"], self.__vocabulary))
        self.__first_pass(doc_batch)
        self.__gibbs_sample(doc_batch)
        self.__pick_max_prob(doc_batch)
        return [doc.cluster_id for doc in doc_batch]

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
        phi_zv = {}
        for cluster in self.clusters:
            pass
