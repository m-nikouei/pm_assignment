import pickle
import os, sys


# checks the pickle modle against user provided data
def runAgainstModel(pkl,fldr,pp):

    #print pickle filename containing model
    print 'Model is read from:', pp
    
    # reads user data
    text = raw_input('\nPlease enter your text data here to classify:')

    # checks if the data is entered
    if text == '' or text == ' ':
        print('No text data is received. Good day!')
        sys.exit(1)

    
    # runs data against the model and prints results
    print('\nThe result of classification:')
    pred = pkl[3].predict_proba(pkl[2].transform(pkl[1].transform([text])))
    s = '{'
    for i in range(len(pkl[0])):
        s += '"' + pkl[0][i][:-6] + '": ' + str(round(pred[0][i],2)) + ', '
    s = s[:-2] + '}'
    print(s)

    # asks user if another test is needed
    opt = raw_input('\nDo you want to test another stirng? (Y/N):')
    if opt.lower() == 'y' or opt.lower() == 'yes':
        runAgainstModel(pkl,fldr,pp)



# sets pickle filename and calls the main recursive function
def set_arg(argv):

    folder = 'pickles'
    ppath = ''

    # checks if the path is given as an argument
    if len(argv) == 1:
        ppath = argv[0]
    # if more than one argument is provided it is an error
    elif len(argv) > 1:
        print('Error: This program only accepts one argument: the path to pickle file!')
        sys.exit(2)

    # if a file is not provided consider the filename created by main program
    if ppath == '':
        ppath = 'rfModelP.pkl'

    # result of reading pickle objects
    pkl = []
    # reads pickle objects from file
    with open(os.path.join(folder,ppath), 'rb') as f:
        pkl = pickle.load(f)

    # run tests
    runAgainstModel(pkl,folder,ppath)


if __name__ == "__main__":
   set_arg(sys.argv[1:])