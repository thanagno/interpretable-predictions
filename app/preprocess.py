import pandas as pd

def preprocess_data(data_path str) - pd.DataFrame
    Reads and preprocesses data.
    # Read the dataset
    data = pd.read_csv(data_path)
    
    # Extract date components
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data['Hour'] = data['Timestamp'].dt.hour
    data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
    data['Month'] = data['Timestamp'].dt.month
    
    # Clean text data
    from nltk.corpus import stopwords
    import re
    sw = stopwords.words('english')
    data['Processed_Ad_Topic_Line'] = data['Ad Topic Line'].apply(
        lambda x ' '.join([re.sub('[^a-z]', '', w.lower()) for w in x.split() if w not in sw])
    )
    
    # Drop unnecessary columns
    data.drop(columns=['Timestamp', 'Ad Topic Line'], inplace=True)
    
    return data
