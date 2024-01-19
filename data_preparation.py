import wfdb
import os
import numpy as np

def getLabel(value):
    # Your implementation logic here to determine the label based on the given value
    if value in ('N', 'L', 'R'):
        return 0
    elif value in ('A', 'a', 'J', 'S'):
        return 1
    elif value in ('V', 'E'):
        return 2
    elif value in 'F':
        return 3
    else:
        return 4


def load_and_process_data(data_path, window_size=300):
    heartbeats = []
    labels = []
    
    record_list = [f.replace('.hea', '') for f in os.listdir(data_path) if f.endswith('.hea')]

    for record_name in record_list:
        record_path = os.path.join(data_path, record_name)
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')
        
        for i in range(len(annotation.symbol)):
            symbol = annotation.symbol[i]
            label = getLabel(symbol)

            if symbol in ['N', 'L', 'R', 'A', 'a', 'J', 'S', 'V', 'E', 'F', '/']:
                center = annotation.sample[i]
                window_start = max(0, center - window_size // 2)
                window_end = min(len(record.p_signal), center + window_size // 2)
                heartbeat = tuple(record.p_signal[window_start:window_end, 0])
                
                if len(heartbeat) == window_size:
                    heartbeats.append(heartbeat)
                    labels.append(label)
                
    data = np.array(heartbeats)
    data = data.reshape((data.shape[0], data.shape[1], 1))
    labels = np.array(labels)

    return data, labels
