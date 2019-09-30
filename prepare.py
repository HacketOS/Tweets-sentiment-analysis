import pandas as pd
import numpy as np
import string
import nltk
import re
import gensim


class Prepare_machine:

    def __init__(self, data):
        self.data = data

    def remove_pattern(self, to_replace=None, to_regular=None):
        if to_replace is None and to_regular is None:
            raise Exeption("Empty argumets!")
        if to_replace is not None:
            def del_links(s):
                return s.replace(to_replace, '')
            self.data = self.data.apply(del_links)
        if to_regular is not None:
            def del_with_regular(s):
                return re.sub(to_regular, '', s, flags=re.MULTILINE)
            self.data = self.data.apply(del_with_regular)
        return self.data

    def remove_punctuation(self):
        def del_punctuation(s):
            return ''.join([l for l in s if l not in string.punctuation])
        self.data = self.data.apply(del_punctuation)
        return self.data

    def remove_numbers(self):
        def del_num(s):
            return ''.join([i for i in s if not i.isdigit()])
        self.data = self.data.apply(del_num)
        return self.data

    def tokenize(self, reg):
        self.data = self.data.apply(lambda s: s.lower())
        tokenizer = nltk.tokenize.RegexpTokenizer(reg)
        self.data = self.data.apply(lambda s: tokenizer.tokenize(s))
        return self.data

    def remove_stop_words(self):
        def del_stop_words(s):
            return [w for w in s if w not in nltk.corpus.stopwords.words('english')]
        self.data = self.data.apply(del_stop_words)
        return self.data

    def lemmatizing(self):
        lemmatizer = nltk.stem.WordNetLemmatizer()

        def lemm(s):
            return [lemmatizer.lemmatize(w) for w in s]
        self.data = self.data.apply(lemm)
        return self.data

    def stemming(self):
        stemmer = nltk.stem.porter.PorterStemmer()

        def stemm(s):
            return [stemmer.stem(w) for w in s]
        self.data = self.data.apply(stemm)
        return self.data

    def default_pipeline(self, to_replace=None, to_regular=None,
                         token_reg=None, short_words='lemm'):
        if short_words is not 'lemm' and short_words is not 'stemm':
            raise Exeption(f'Unknown tool: {short_words}')
        print('Removing patterns...')
        self.remove_pattern(to_replace=to_replace, to_regular=to_regular)
        print('Removing punctuation...')
        self.remove_punctuation()
        print('Removing numbers...')
        self.remove_numbers()
        print('Tokenizing...')
        self.tokenize(reg=token_reg)
        print('Removing stop words...')
        self.remove_stop_words()
        if short_words == 'stemm':
            print('Stemming...')
            self.stemming()
        else:
            print('Lemmatizing...')
            self.lemmatizing()
        print('Done!')
        return self.data

    def save_csv(self, name='data.csv'):
        self.data.to_csv(name)
