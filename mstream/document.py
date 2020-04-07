class Document:
    def __init__(self, id_, terms, vocabulary, topic=None):
        self.id = id_
        # the object to thich topic points is constant and not modifiable
        self.topic = topic
        self.__bow = {}
        for term in terms:
            term_id = vocabulary.add(term)
            if term_id not in self.__bow:
                self.__bow[term_id] = 0
            self.__bow[term_id] += 1

    def total_len(self):
        total = 0
        for _, freq in self.__bow.items():
            total += freq
        return total

    def freq(self, term):
        return self.__bow[term]

    def __len__(self):
        return len(self.__bow)

    def __iter__(self):
        return iter(self.__bow.items())

    def __str__(self):
        return str(self__.bow)
