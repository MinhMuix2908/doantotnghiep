import os
import pickle
import numpy as np

class SimilaritySearch:
    def __init__(self, database, filenames, threshold):
        '''
            database: path to pickle contain list of np array or 2D-array 
                      contains vector embedding (normalized)
            filenames: path to pickle, dictionary contains names in database
            option: {"ip", "l2"} metric distance 
            threshold: Threshold for metric distance
        '''
        self.database_path = database
        self.filenames_path = filenames

        if os.path.exists(self.database_path):
            self.database = pickle.load(open(database, "rb"))
            self.filenames = pickle.load(open(filenames, "rb"))
            #Convert list to numpy array
            if type(self.database) == list:
                self.database = np.array(self.database)
        else:
            self.filenames = {}
        self.threshold = threshold
        # _, d = self.database.shape
          
    def add_vector(self, vector, name):
        '''
            Add vector to database and update index object with GPU or CPU
            vector: A face vector embedding, 2-D numpy array, (1, d)
            name: Name for vector
        '''

        
        if not os.path.exists(self.database_path):
            pickle.dump(vector.numpy(), open(self.database_path, "wb"))
            self.filenames[0] = name
            pickle.dump(self.filenames, open(self.filenames_path, "wb"))
            self.database = pickle.load(open(self.database_path, "rb"))
            self.filenames = pickle.load(open(self.filenames_path, "rb"))
        else:
            #Add vector to database
            self.database = np.append(self.database, vector, axis= 0)
            #Add vector to dict
            self.filenames[len(self.filenames)] = name

            #Update database and dict
            pickle.dump(self.database, open(self.database_path, "wb"))
            pickle.dump(self.filenames, open(self.filenames_path, "wb"))

    def search_name(self, query):
        '''
            Search name (no GPU)
            query: A vector embedding. 1-D numpy array, (d,)

            return name
        '''        
        distances = self.database.dot(query.T)    #1-D numpy array, (d, )
        index = np.argmax(distances)
        # print(distances[index])
        max_distance = distances[index]
        if max_distance < self.threshold:
            return "Unknown"
        
        return self.filenames[index]