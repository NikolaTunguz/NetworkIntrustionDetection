from Data import data_class

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

    print(data.head())




if __name__ == '__main__':
    main()