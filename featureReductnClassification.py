from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import stringP
import pickle
import os

# feature reduction and classification class:
#   1) implements document term matrix
#   2) computes 1000 most common words in our corpus, after removing common words
#   3) creates tf-idf matrix of the training data after spliting training and test data
#   4) uses Chi squared feature selection (4000 features) on tf-idf matrix based on traning data and 
#      transform test data (one can perform a grid search to find an optimized number of features)
#   5) finds best random forest model for chi squared train and test data using grid search cross validation 
#      of 10 folds
#   6) perfoms PCA (0.95 variability) on dtm after spliting training and test data
#   7) finds best logistic regression model for PCA train and test data using grid search cross validation of 10 folds
#   8) creates a pickle file of trained models for future use
class frcClass:

    # test collection size
    tsize = 500

    # pca variability score
    pcaV = 0.95

    # chi squred feature number to select
    fnum = 4000

    # cross validation number
    CV = 10

    # input corpus and labels
    corpus = None
    labels = None

    # count vectorizer
    cvecer = None

    # document term matrix
    dtm = None
    
    # train and test set created by PCA
    x_pca = None
    y_pca = None
    x_tpca = None
    y_tpca = None

    # list of objects to be pickled.
    pkl = []

    # initializes class with data
    # creating document term matrix and count vectorizer
    def __init__(self,corpus,labels,classN):
        self.corpus = corpus
        self.labels = labels
        print('\nLets tokenize and do word count of each document in corpus')
        self.cvecer = CountVectorizer(tokenizer = stringP.tokenize)
        self.dtm = self.cvecer.fit_transform(self.corpus)
        print '\nCreated dtm of shape', self.dtm.shape

        # keep class names to be pickled
        self.pkl.append(classN)

    # finding most n common words
    def findMC(self,num):
        # sums up each collumn of tdm, this gives the number of words
        sum_words = self.dtm.sum(axis=0)
        # creates list of pairs of word and word count for each word in the vocabulary of the count vectorizer
        words_freq = [(word, sum_words[0, idx]) for word, idx in self.cvecer.vocabulary_.items()]
        # sort the list of pairs based on thier word cound with reverse order
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
        # returns n first elements of sorted list
        return words_freq[:num]


    # splits corpus data to train and test sets, creates tf-idf matrix and performs Chi squared feature selection
    def ttSplitReduceChi2(self):

        # splits tdm and labels to train and test set
        print('split data into train and test...')
        x_train,x_test,y_train,y_test = train_test_split(self.corpus,self.labels,test_size = self.tsize)

        # creates tf-idf matrix
        print('Creating tf-idf matrix...')
        vecer = TfidfVectorizer(tokenizer = stringP.tokenize)
        x_vtrain = vecer.fit_transform(x_train)
        x_vtest = vecer.transform(x_test)

        # keep tf-idf tokenizer to be pickled
        self.pkl.append(vecer)

        # performs Chi squared feature selection 
        print('Performing chi squared feature reduction...')
        ch2 = SelectKBest(chi2, k = self.fnum)
        X_train = ch2.fit_transform(x_vtrain, y_train)
        X_test = ch2.transform(x_vtest)
        # keep chi squared feature selection model to be pickled
        self.pkl.append(ch2)
        print('Done!')

        self.x_chi = X_train
        self.y_chi = y_train
        self.x_tchi = X_test
        self.y_tchi = y_test

    def RFclassifierCV(self):
    
        # creates model
        print('To get the best result on RF, we perform 10 fold cross validation on training data.')
        print('We condider min_samples_split and min_samples_leaf as prameters to optimize.')
        rf = RandomForestClassifier()

        # model parameters to be optimized with given values
        parameters = {
            'n_jobs':[5],
            'n_estimators':[1,10,50,100,150],
            'min_samples_split':[1.0,2,3,5,7],
            'min_samples_leaf':[1,0.5,5,7],
        }

        # creates grid search for cross validation
        GS = GridSearchCV(rf, parameters,cv=self.CV,verbose=10)

        # performs cross validation
        GS.fit(self.x_chi,self.y_chi)

        # considers best model
        brf = GS.best_estimator_
        print('Best found model:')
        print(brf)
        # keep random forest model to be pickled
        self.pkl.append(brf)

        # perform predictions on test data
        pred = brf.predict(self.x_tchi)
        # gets the metrics
        score = metrics.accuracy_score(self.y_tchi, pred)
        print("accuracy:   %0.3f" % score)
        print("classification report:")
        print(metrics.classification_report(self.y_tchi, pred))
        print("confusion matrix:")
        print(metrics.confusion_matrix(self.y_tchi, pred))


    # splits dtm data to train and test sets, scales data by mean reduction, performs PCA
    def ttSplitReduce(self):

        # converts dtm (a sparse matrix) to a dense matrix
        darray = self.dtm.toarray()

        # splits tdm and labels to train and test set
        print('split data into train and test...')
        x_train,x_test,y_train,y_test = train_test_split(darray,self.labels,test_size = self.tsize)
        print 'train and test data shape:', x_train.shape,len(y_train),x_test.shape,len(y_test)

        # defines scaler
        scaler = StandardScaler()
        print('fiting scaler...')
        # fit the scaler with train data
        scaler.fit(x_train)
        print('scaling data...')
        # scales train and test data 
        sx_train = scaler.transform(x_train)
        sx_test = scaler.transform(x_test)

        # defines PCA
        pca = PCA(self.pcaV)
        print('fiting PCA...')
        # creates PCA
        pca.fit(sx_train)
        print('reducing data components...')
        # transforms train and test data based on created PCA
        nx_train = pca.transform(sx_train)
        nx_test = pca.transform(sx_test)
        print('Done!')
    
        # saves train and test data for classification
        self.x_pca = nx_train
        self.y_pca = y_train
        self.x_tpca = nx_test
        self.y_tpca = y_test



    # trains a logistic regression model based on PCA reduced data with 10 folds cross validation
    def LRclassifierCV(self):
    
        # creates model
        print('To get the best result on LR, we perform 10 fold cross validation on training data.')
        print('We condider max_iter, C and solver as prameters to optimize.')
        lr = linear_model.LogisticRegression(solver = 'lbfgs')

        # model parameters to be optimized with given values
        parameters = {
            'n_jobs':[5],
            'max_iter':[50,100,150],
            'C':[1,10,100],
            #'solver': ['lbfgs', 'liblinear'],
        }

        # creates grid search for cross validation
        GS = GridSearchCV(lr, parameters,cv = self.CV,verbose=10)

        # performs cross validation
        GS.fit(self.x_pca,self.y_pca)
        print('Cross validation done!')

        # considers best model
        print('Logistic regressin results:')
        blr = GS.best_estimator_
        print('Best found model:')
        print(blr)
    
        # perform predictions on test data
        pred = blr.predict(self.x_tpca)
        # gets the metrics
        score = metrics.accuracy_score(self.y_tpca, pred)
        print("accuracy:   %0.3f" % score)
        print("classification report:")
        print(metrics.classification_report(self.y_tpca, pred))
        print("confusion matrix:")
        print(metrics.confusion_matrix(self.y_tpca, pred))


    # creates pickle file of list of needed objects
    def pickleModels(self):
        with open(os.path.join('pickles','rfModelP.pkl'), 'wb') as f:
            pickle.dump(self.pkl, f)


# performs PCA on term document matrix with given needed variability or component number
def tdmPCA(tdm,pcaV):

    # defines PCA
    pca = PCA(pcaV)
    print('fiting PCA...')
    # creates PCA
    tdmpca = pca.fit_transform(tdm)
    print 'Done!'
        
    return tdmpca