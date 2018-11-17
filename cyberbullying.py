# Tools to process data
import praw 
import pickle
import pandas
import numpy

# ML tools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

# NLP tools
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import re


class CyberBullyingDetectionEngine:
    # Class that deals with training and deploying cyberbullying detection models

    def __init__(self):
        self.corpus = None
        self.tags = None # either cyberbullying or not cyberbullying
        self.lexicon = None
        self.vectorizer = None
        self.model = None
        self.metrics = None


    # class CustomVectoriser :
    #     # extracts features from texts and create word vectors

    #     def __init__(self, lexicons):
    #         self.lexicons = lexicons

    #     def transform(self, corpus):
    #         word_vectors = list()
    #         for text in corpus:
    #             features = list()
    #             for k ,v in self.lexicons.items():
    #                 features.append(len([w for w in word_tokenize(text) if w in v])) # take a tweet from a text from corpus
    #                 # and break it down in single words and check how words appear in lexicon and make vector from that
    #             word_vectors.append(features)
    #         return numpy.array(word_vectors)



    def _simplify(self, corpus):
         # Removes stop words from list of strings (corpus) and convert to lower case
         # remove non alphanumeric characters (emojis), also does stemming
         # stem in treating the words in different tenses as the same (ie running and ran will change to run)
        stop_words = set(stopwords.words('english'))
        stemmer = SnowballStemmer('english')
        
        def clean(text):
            text = re.sub('[^a-zA-Z0-9]', ' ', text)
            words = [stemmer.stem(w) for w in word_tokenize(text.lower()) if w not in stop_words] 
            return " ".join(words)

        return [clean(text) for text in corpus]

    def _model_metrics(self, features, tags):
        # takes in testing data and returns dictionarry of metrics

        tp = 0 #true positive
        fp = 0 #false positive
        tn = 0 #true negative
        fn = 0 #false negative

        predictions = self.predict(features)
        for r in zip(predictions, tags):
            if r[0] == 1 and r[1] == 1: # means we did it right
                tp+=1
            elif r[0] == 1 and r[1] == 0: # we thought it was something else
                fp+=1
            elif r[0] == 0 and r[1] == 1: # 
                fn+=1
            elif r[0] == 0 and r[1] == 0: # 
                tn+=1
                
            


        precision = tp / (tp + fp) # how good we were
        recall = tp / (tp +fn) # measures how much did the model miss
        return {
            'precision' : precision,
            'recall' : recall,
            'f1' : (2 * precision * recall) / precision + recall # f1 takes harmonic means which penalizes extreme values (as in makes them closer to the extreme)
        }




    def load_corpus(self, path, corpus_col, tag_col):
        # Takes in a path to a picled pandas dataframe and also the name of the corpus column
        # and the name of tag column and extracts a tagged corpus

         data = pandas.read_pickle(path)[[corpus_col, tag_col]].values
         self.corpus = [row[0] for row in data]
         self.tags = [row[1] for row in data]

    # def _get_lexicon(self,path):
    #     # return a set containing every word in the file

    #     words = set()
    #     with open(path) as file:
    #         for line in file:
    #             words.update(line.strip().split(' '))

    #     return words

    # def load_lexicon(self, fname):
    #     # loads a set of words from a text file
    #     if self.lexicon is None:
    #         self.lexicon = {}
    #     self.lexicon[fname] = self._get_lexicon('data/' + fname + '.txt')

    def load_model(self, model_name):
        # loads the ml model, the vectorizer that goes with it and its performance metrics

        self.model = pickle.load(open('./models/' + model_name + 'nl_model.pkl', 'rb'))
        self.vectorizer = pickle.load(open('./models/' + model_name + '_vectorizer.pkl', 'rb'))
        self.metrics = pickle.load(open('./models/' + model_name + '_metrics.pkl', 'rb'))


    def train_using_bow(self):
        # trains a model using a Bag of words (word counts)
        corpus = self._simplify(self.corpus)
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(corpus)

        bag_of_words = self.vectorizer.transform(corpus)
        x_train, x_test, y_train, y_test = train_test_split(bag_of_words, self.tags, test_size= 0.2, stratify = self.tags)

        self.model = MultinomialNB()
        self.model.fit(x_train, y_train)

        self.metrics = self._model_metrics(x_test, y_test)

    def train_using_tfidf(self):
        # trains using tfidf weighted word counts as features
        # takes notes of words frequency in the entire document as well as in each text 
        # in attributes a weight for each word as a result

        corpus = self._simplify(self.corpus)
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(corpus)

        word_vectors = self.vectorizer.transform(corpus)
        x_train, x_test, y_train, y_test = train_test_split(word_vectors, self.tags, test_size=0.2, stratify=self.tags)

        self.model = MultinomialNB()
        self.model.fit(x_train, y_train)

        self.metrics = self._model_metrics(x_test, y_test)

    # def train_using_custom(self):
    #     # trains model using a custom feature extraciton approach

    #     corpus = self._simplify(self.corpus)
    #     self.vectorizer = self.CustomVectoriser(self.lexicon)

    #     word_vectors = self.vectorizer.transform(corpus)

    #     x_train, y_train, x_test, y_test = train_test_split(word_vectors, self.tags, test_size = 0.2, stratify =self.tags)

    #     self.model = SVC() # suport vector classifer
    #     self.model.fit(x_train, y_train)

    #     self.metrics = self._model_metrics(x_test, y_test)

       
    

    def predict(self, corpus):
        # takes in reddit data and returns predictions

        x = self.vectorizer.transform(self._simplify(corpus))
        return self.model.predict(x)

    def evaluate(self):
        return self.metrics

    def save_model(self, model_name):
        # Saves the model for future use
    
        pickle.dump(self.model, open('./models/' + model_name + '_ml_model.pkl', 'wb'))
        pickle.dump(self.vectorizer, open('./models/' + model_name + '_vectorizer.pkl', 'wb'))
        pickle.dump(self.metrics, open('./models/' + model_name + '_metrics.pkl', 'wb'))

       
if __name__ == '__main__':
    reddit = praw.Reddit(
        client_id = '0CImLpjwo5_t9g',  
        client_secret = '_UxJ95DWRlpseQfFjknMpf7Lrhw',
        user_agent = 'script_name by /u/username' 
    )

    new_comments = reddit.subreddit('TwoXChromosomes').comments(limit = 1000)
    queries = [comment.body for comment in new_comments]

    engine = CyberBullyingDetectionEngine()
    engine.load_corpus('./data/final_labelled_data.pkl', 'tweet', 'class') #

    #engine.train_using_bow()
    #engine.save_model('bow')

    engine.train_using_tfidf()
    engine.save_model('tfidf')

    # engine.train_using_custom()
    # engine.save_model('custom')




    print(engine.evaluate())

    print(engine.predict(queries))
