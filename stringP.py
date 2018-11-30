import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import word_tokenize          
import re

# removes common english words (listed in nltk.corpus.stopwords) from corpus
def removeCommon(str):
    
    # creates the set of common words
    sw = set(stopwords.words('english'))

    # splits the string in to words.
    l = str.split()

    res = []

    # if a word is not in set of common words, addes it to the result list
    for x in l:
        if not x in sw:
            res.append(x)
    
    # puts together the result list as a string and returns it
    return ' '.join(res)


# boxes together several functions:
#   removeCommon defined above
#   removing any character other than letters, digits and spaces
#   reduces multiple spaces to one.
#   lower case every letter
def pipeLine(str):

    pipeline = [removeCommon,
            lambda s: re.sub('[^\\w\\s]', '', s),
            lambda s: re.sub( '\\s+', ' ',s).strip(),
            lambda s: s.lower(),
    ]
    
    # applies all functions on input
    for f in pipeline:
        str = f(str)
   
    return str


# removes stem parts of words, i.e., ing, s ...
def stem_tokens(tokens):
    stemmer = PorterStemmer()
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

# for a string returns a list of tokens 
def tokenize(text):

    # runs the pipeLine on the string
    nt = pipeLine(text)

    # tokenizes the string
    tokens = nltk.word_tokenize(nt)

    # removes stem parts of tokens
    stems = stem_tokens(tokens)

    return stems