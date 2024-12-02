from Data import data_class
from Models import xgboost_class
from Models import neural_network_class
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def main():
    files = [
            'Data/Friday-WorkingHours-Morning.pcap_ISCX.csv',
            'Data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
            'Data/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
            'Data/Monday-WorkingHours.pcap_ISCX.csv',
            'Data/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
            'Data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
            'Data/Tuesday-WorkingHours.pcap_ISCX.csv',
            'Data/Wednesday-workingHours.pcap_ISCX.csv'
            ]
    
    print("Preprocessing Data")
    data_processor = data_class.DataProcessor(files)
    data = data_processor.get_data()

    true_label_names = data_processor.get_original_labels()

    labels = data['Label']
    features = data.drop(columns = ['Label'])

    
    print("\nTraining XGBoost")
    xgboost = xgboost_class.MyXGBoost(features, labels)
    xgboost.train_model()
    accuracy, cm, report = xgboost.evaluate()

    print(accuracy)
    print(cm)
    print(report)

    #xgb_cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = true_label_names)
    #xgb_cm_display.plot()
    #plt.show()


    print("\nTraining Neural Network")
    nn = neural_network_class.MyNN(features, labels)
    nn.train_model()

    accuracy, cm, report = nn.evaluate()
    
    print(accuracy)
    print(cm)
    print(report)

    #nn_cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = true_label_names)
    #nn_cm_display.plot()
    #plt.show()

if __name__ == '__main__':
    main()