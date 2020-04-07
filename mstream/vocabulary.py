class Vocabulary:
    def __init__(self):
        self.__id2term = []
        self.__term2id = {}

    def add(self, term):
        """
        Add the term `term` to this vocabulary and return its integer id. If the term is already in the dictionary, it just returns its id.

        Parameters:
        - term: the term to be added
        Return:
        The integer id of `term`.
        """
        if term in self.__term2id:
            return self.__term2id[term]

        self.__id2term.append(term)
        id_ = len(self.__id2term) - 1
        self.__term2id[term] = id_
        return id_

    def id2term(self, id_):
        return self.__id2term[id_]

    def term2id(self, term):
        return self.__term2id[term]

    def __contains__(self, term):
        return term in self.__term2id

    def __len__(self):
        return len(self.__id2term)
