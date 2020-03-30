class Vocabulary:
    def __init__(self):
        self.__vocabulary = set()

    def add(self, word):
        self.__vocabulary.add(word)

    def __contains__(self, word):
        return word in self.__vocabulary

    def __len__(self):
        return len(self.__vocabulary)
