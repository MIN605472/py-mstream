from mstream.clusterfeature import ClusterFeatureVector

class Mstream:
    def __init__(self, filename):
        """
        Initialize the model by reading it from the specified file.
        """
        pass

    def __init__(self, iter_num=100, alpha=0.03, beta=0.03, words_topic_num=50):
        self.__vocabulary = Vocabulary()
        self.__iter_num = iter_num
        self.__alpha = alpha
        self.__beta = beta
        self.__words_topic_num = words_topic_num
        self.__clusters = ClusterFeatureVector()
        self.__free_cluster_ids = [0]

    def process(self, batch):
        doc_batch = []
        for doc in batch:
            doc_batch.append(Document(doc["id"], doc["word_list"]))
        self.__first_pass(doc_batch)
        self.__gibbs_sample(doc_batch)

    def save_model(self):
        pass

    def __first_pass(self, docs):
        for doc in docs:
            doc.cluster_id  = self.__clusters.sample_and_add(doc)

    def __gibbs_sample(self, docs):
        for i in range(self.__iter__num - 1):
            for doc in docs:
                self.__clusters.sub(doc, doc.cluster_id)
                self.__clusters.sample_and_add(doc)

    def __estimate_posterior(self):
        phi_zv = {}
        for cluster in self.clusters:
            pass
