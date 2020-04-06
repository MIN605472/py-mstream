class Document:
    def __init__(self, id_, words, vocabulary):
        self.id = id_
        self.cluster_id = -1
        self.__bow = {}
        for word in words:
            word_id = vocabulary.add(word)
            if word_id not in self.__bow:
                self.__bow[word_id] = 0
            self.__bow[word_id] += 1

    def total_len(self):
        total = 0
        for _, freq in self.__bow.items():
            total += freq
        return total

    def freq(self, word):
        return self.__bow[word]

    def __len__(self):
        return len(self.__bow)

    def __iter__(self):
        return iter(self.__bow.items())

    def __str__(self):
        return str(self__.bow)
