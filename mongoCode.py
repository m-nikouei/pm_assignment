import pymongo
import json
import os

class mongoClass:

    # mongoDB client
    client = None

    # mongoDB database name
    dbname = 'stack'

    # initilizes the class with a client to mongoDB
    def __init__(self):
        self.client = pymongo.MongoClient()

    
    # creating mongo database 'stack' and iserting data from json files
    # path: path to data folder
    def createStack(self,path):
        
        # if 'stack' exists first drop it                                  
        if self.dbname in self.client.list_database_names():
            print('stack exists in mongo! We drop it first.')
            self.client.drop_database(self.dbname)

        print '\nWe are ready to create stack and load data in it.'

        # read filenames in data directory
        files = os.listdir(path)

        # create 'stack' database
        db = self.client[self.dbname]

        # for each filename in data folder, read json file
        for fn in files:
            print ('\nreading ' + fn[:-5])
            jfile = open(os.path.join(path, fn), 'r')

            # loading json data from file
            diss = json.loads(jfile.read())

            # count: number of items read
            count = 0

            # for each item read
            for item in diss:

                # remove unwanted fields
                item.pop('tags', None)
                item.pop('userid', None)
                item.pop('id', None)
                for ans in item['answers']:
                    ans.pop('userid', None)
                    ans.pop('id', None)
            
                # add to count
                count += 1
        
            # create the collection with the file name
            col = db[fn[:-5]]

            # insert items to the collection as documents
            col.insert_many(diss)

            print ' -> Added',count,'documents to',fn[:-5],'collection!'
    
        # print the list of collection names in 'stack' as sanity check!
        print '\nstack collections:',db.list_collection_names()


    # reads mongo database 'stack' and produces the corpus list
    def produceCorpus(self):

        # variables for corpus and labels
        corpus = []
        labels = []

        # connecting to stack
        db = self.client[self.dbname]

        # read each collection in stack as follows
        for coln in db.list_collection_names():

            print ('In ' + coln)

            # get the collection
            col = db[coln]

            # read each document in current collection as follows
            for doc in col.find():

                # put body, all answers adn title in a string 
                s = doc['body']
                for ans in doc['answers']:
                    s += ans['body']
                s += doc['title']

                # add the string to corpus list
                corpus.append(s)

                # label for this document is index of its collection in stack
                labels.append(db.list_collection_names().index(coln))

        # return corpus and labels as the data used in our analysis, together with class names to be pickled
        return corpus, labels, db.list_collection_names()