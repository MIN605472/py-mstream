class Vocabulary:
    def __init__(self):
        self.__id2word = []
        self.__word2id = {}

    def add(self, word):
        """
        Add the word `word` to this vocabulary and return its integer id. If the word is already in the dictionary, it just returns its id.

        Parameters:
        - word: the word to be added
        Return:
        The integer id of `word`.
        """
        if word in self.__word2id:
            return self.__word2id[word]

        self.__id2word.append(word)
        id_ = len(self.__id2word) - 1
        self.__word2id[word] = id_
        return id_

    def id2word(self, id_):
        return self.__id2word[id_]

    def word2id(self, word):
        return self.__word2id[word]

    def __contains__(self, word):
        return word in self.__word2id

    def __len__(self):
        return len(self.__id2word)
