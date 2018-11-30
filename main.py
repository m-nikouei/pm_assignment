import sys, os
import mongoCode
import featureReductnClassification
import nltk


# Helper Functions

# checks NLTK version and if lower than 3.2.3 exits with error 
def nltkCheck():
    v = nltk.__version__
    nl = v.split('.')
    res = True
    if nl[0] <= 3:
        res = False
    elif nl[1] <= 2:
        if len(nl) <= 2:
            res = False
        elif len(nl) == 3:
            if nl[2] < 3:
                res = False
    if not res:
        print 'Error: This program needs NLTK version 3.2.3 or above.'
        sys.exit(2)


# checks provided path to be a folder containing all data files
def checkPath(path):

    # checks if the path of a folder is given
    # exists with error otherwise
    if not os.path.isdir(path):
        print 'Error: The system cannot find the path specified:', path
        sys.exit(2)
    
    # set of expeted files
    exptedSet = {'astronomy_posts.json', 'aviation_posts.json', 'beer_posts.json', 'outdoors_posts.json', 'pets_posts.json'}
    
    # gets the list of files in the folder and creates a set of them
    files = os.listdir(path)    
    sfiles = set(files)

    # checks if all expected files are in the folder
    if not exptedSet == sfiles:
        print('Error: The path does not have all needed files or has more. Please check the path and rerun the program!')
        sys.exit(2)
    
    # existing without error shows everything is in order and program can use the provided path

# Main Function


# main function
def main(argv):
    
    # checks nltk version
    nltkCheck()

    # asks if a second classification with cross validation of logistic regression using PCA generated data of dtm is wanted! 
    # this second classification is added to show how dtm is used (results are close our main classification method)
    # this takes too long to finish and needs 5 GB of free RAM at run time, so running it is optional
    str = 'Do you want to run extra classification  with logistic regression and 10 fold cross validation on PCA generated data? (Y/N)'
    opt = raw_input(str)

    # gets the path from user
    path = ''
    # checks if the path is given as an argument
    if len(argv) == 1:
        path = argv[0]
    # if not asks for it from the user
    elif len(argv) == 0:
        print('To run the program, the path to json files needs to be provided')
        path = raw_input('Please enter the path: ')
    # if there are more than one arguments provided, exists with error
    else:
        print('Error: This program only accepts one argument: the path to json files!')
        sys.exit(2)

    # checks the path
    checkPath(path)

    # create an instanse of our mongo client
    mongoC = mongoCode.mongoClass()
    
    # creating database 'stack' and reading json files in it
    print('\nProblem 1: Loading data set in mongoDB.\n')
    mongoC.createStack(path)

    # reading data form stack databasse to create corpus and labels, together with class names
    print('\nLets create unprocessed corpus from mongo collections:')
    corpus, labels, classnames = mongoC.produceCorpus()
    print('\nFinished reading corpus!')

    # creating an instanse of feature reduction and classification class
    # this is the main analysis class and all analysis tasks are done through it
    frc = featureReductnClassification.frcClass(corpus, labels,classnames)

    # finding 1000 most common words
    print('\nProblem 2: Find 1000 most common words.')
    print 'Here are the 1000 most common words:'
    print(frc.findMC(1000))

    # term document matrix out of document term matrix
    print('\nProblem 3: Create a term document matrix.')
    # converts dtm (a sparse matrix) to a dense matrix
    darray = frc.dtm.toarray()
    # creates tdm
    tdm = darray.transpose()
    print 'Term Document Matrix:', tdm
    print 'Term Document Matrix shape:', tdm.shape

    # performs PCA on tdm
    print('\nProblem 4: Perform dimension reduction (i.e. PCA) on the tdm and retain 95% variability .')
    tdmpca = featureReductnClassification.tdmPCA(tdm,frc.pcaV)
    print 'PCA transformed tdm:', tdmpca
    print 'Its dimensions:', tdmpca.shape 

    # splits data into train and test parts, creates tf-idf matrix and performs Chi squared feature selection
    print('\nProblem 5: build a classifier whose labels are the 5 topic names.')
    print 'We create tf-idf with chi squared selection of our data and using it to create an optimized',
    'random forest classier.'
    frc.ttSplitReduceChi2()

    # runs cross validation for random forest classifier, finds best model and prints its metrics
    print('\nNext lets build an opimal random forest model and test its accuracy.')
    frc.RFclassifierCV()

    # pickles the best model for future use
    print '\nWe create a pickle file of our model for future use. You can use it by running "useModel.py"',
    'script on your text.'
    frc.pickleModels()

    # Base on user choise, runs 10 fold cross validation for logistic regression on data resulted from PAC on dtm
    if opt.lower() == 'y' or opt.lower() == 'yes': 

        # splits data into train and test parts, scales using mean reduction and performs PCA on data
        print 'We are going to use PCA results to perform logistic regression! We keep 500 data points as ',
        'test set and fit pca on the remaning data.'
        frc.ttSplitReduce()

        # creates second classification model and print its metrics
        print('First lets build an opimal logistic regression model and test its accuracy.')
        frc.LRclassifierCV()

    print('\n\nAll problems are solved. Good day!')


if __name__ == "__main__":
   main(sys.argv[1:])
