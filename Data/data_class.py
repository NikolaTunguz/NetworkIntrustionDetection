import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

class DataProcessor():
    def __init__(self, files):
        self.data = None
        self.file_names = files

        self.read_data(self.file_names)

    def read_data(self, file_names):
        #read each csv individually (multiple in dataset)
        files = []
        for name in file_names:
            cur_file = pd.read_csv(name)
            files.append(cur_file)

        #combine all csv into a singular dataframe
        self.data = pd.concat(files, ignore_index = True)

    def preprocess(self):
        #removing null rows
        self.data = self.data.dropna()
        
        #separating features and labels for processing
        features = self.data.drop(columns = ['Label'])
        labels = self.data['Label']

        #standardizing all features from 0 to 1
        standardizer = StandardScaler()
        scaled_features = standardizer.fit_transform(features)

        #encoding the BENIGN and DDOS labels into 0 and 1
        encoder = LabelEncoder()
        encoded_labels = encoder.fit_transform(labels)

        #combining modified data back together
        self.data = pd.DataFrame( scaled_features, columns = features.columns)
        self.data['Label'] = encoded_labels

    def get_data(self, preprocess):
        if preprocess == True:
            self.preprocess()
        return self.data