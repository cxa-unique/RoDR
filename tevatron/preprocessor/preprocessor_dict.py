class TrainPreProcessor:
    def __init__(self, tokenizer, query_max_length=32, text_max_length=256):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.text_max_length = text_max_length

    def __call__(self, example):
        query = self.tokenizer.encode(example['query'],
                                      add_special_tokens=False,
                                      max_length=self.query_max_length,
                                      truncation=True)
        if 'query_variation' in example:
            query_variation = self.tokenizer.encode(example['query_variation'],
                                      add_special_tokens=False,
                                      max_length=self.query_max_length,
                                      truncation=True)
        else:
            query_variation = None

        positives = []
        for pos in example['positive_passages']:
            text = pos['title'] + " " + pos['text'] if 'title' in pos else pos['text']
            positives.append(self.tokenizer.encode(text,
                                                   add_special_tokens=False,
                                                   max_length=self.text_max_length,
                                                   truncation=True))
        negatives = []
        for neg in example['negative_passages']:
            text = neg['title'] + " " + neg['text'] if 'title' in neg else neg['text']
            negatives.append(self.tokenizer.encode(text,
                                                   add_special_tokens=False,
                                                   max_length=self.text_max_length,
                                                   truncation=True))
        if query_variation is None:
            return {'query': query, 'positives': positives, 'negatives': negatives}
        else:
            return {'query': query, 'query_variation': query_variation, 'positives': positives, 'negatives': negatives}


class TestPreProcessor:
    def __init__(self, tokenizer, query_max_length=32):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length

    def __call__(self, example):
        query_id = example['query_id']
        query = self.tokenizer.encode(example['query'],
                                      add_special_tokens=False,
                                      max_length=self.query_max_length,
                                      truncation=True)
        return {'text_id': query_id, 'text': query}


class CorpusPreProcessor:
    def __init__(self, tokenizer, text_max_length=256):
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length

    def __call__(self, example):
        docid = example['docid']
        text = example['title'] + " " + example['text'] if 'title' in example else example['text']
        text = self.tokenizer.encode(text,
                                     add_special_tokens=False,
                                     max_length=self.text_max_length,
                                     truncation=True)
        return {'text_id': docid, 'text': text}
