# Stack Exchange Posts Analysis 

This python program analyzes stack exchange posts and creates a classification model for 5 categories of posts: astronomy, aviation, beer, outdoors, pets. Input data is not provided in this repo. 

The program has two parts. The main part includes mongoCode.py, stringP.py, featureReductnClassification.py, main.py. This does all the analysis and at the end creates a pickle file of our model. The second part is a small script in file useModel.py to test user inputs against the pickle model.  
 
## Requirements

Python 2.7.xx and latest version of mongoDB is needed to run the program. We also suggest NLTK 3.2.3 and above as version 3.2.2 has a bug which will trigger an IndexError on the program. scikit-learn is also needed. The program is tested on version 0.19.2. 

The program also needs the path to the folder of json data files. This files are not provided in this repo. The user can provide the path as a command line argument at start time. If the argument is not provided, the program asks for it at the beginning of its run. 

The folder containing the code needs to have a subfolder named pickles. The program outputs the pickle file containing the serialized model to this folder.

 
## How It Runs

To run the main program simply run main.py. At the beginning the user is prompted to choose if they want to run a second classification method on PCA generated date. Choosing to do this will significantly increase the time needed to finish the run. Next, user is asked to enter the path to data folder (StackExchange_posts), if the path is not provided as an argument already. After that the program goes through problems one by one.

To run the test script, simply run useModel.py. The program asks for a test string. After entering a valid string, the program shows the classification result based on the pickled model. You can find some test samples in pickltest.txt. This test samples should be copied to the command line by hand. A pre-trained model is provided in file
'mym.pkl'. You can use this model by giving its name as an argument when you run useModel.py.
