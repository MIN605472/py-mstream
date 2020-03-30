class Document:
    def __init__(self, id, word_list, vocabulary):
        self.id = id
        self.cluster_id = -1
        self.__bow = {}
        for word in word_list:
            freq = self.__bow.setdefault(word, 0)
            self.__bow[word] += 1
            vocabulary.add(word)

    def total_len(self):
        total = 0
        for _, freq in self.__bow:
            total += freq
        return total

    def freq(self, word):
        return self.__bow[word]

    def __len__(self):
        return len(self.__bow)

    def __iter__(self):
        return iter(self.__bow.items())

    def __next__(self):
        return next(self.__bow)
