from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

class MyXGBoost:
    def __init__(self, X, y):  
        #core model variables
        self.model = XGBClassifier( 
            objective = 'multi:softmax',     #criteria 
            n_estimators = 100,              #number of boosting iterations
            max_depth = 5,                   #max tree depth
            learning_rate = 0.2,             #learning rate
            subsample = 0.7,                 #percent sampling of data
            colsample_bytree = 0.8,          #percent sampling of attributes
            )
        
        self.X = X
        self.y = y

        #model varables for train/test split
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.split_data()
    
    #function to split data into training and test
    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.2, random_state = 42)

    #function to train the model
    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    #private function to predict on test dataset
    def predict(self):
        prediction = self.model.predict(self.X_test)
        return prediction

    #function to evaluate prediction performance
    def evaluate(self):
        prediction = self.predict()
        accuracy = accuracy_score(self.y_test, prediction)
        cm = confusion_matrix(self.y_test, prediction)
        report = classification_report(self.y_test, prediction)
        return accuracy, cm, report