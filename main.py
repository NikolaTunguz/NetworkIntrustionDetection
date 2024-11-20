from Data import data_class
from Models import xgboost_class
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
    
    
    data_processor = data_class.DataProcessor(files)
    data = data_processor.get_data()

    labels = data['Label']
    features = data.drop(columns = ['Label'])
    print(np.unique(labels))

    xgboost = xgboost_class.MyXGBoost(features, labels)
    xgboost.train()
    accuracy, cm, report = xgboost.evaluate()


    print(accuracy)
    print(cm)
    print(report)

    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm)
    cm_display.plot()
    plt.show()



if __name__ == '__main__':
    main()