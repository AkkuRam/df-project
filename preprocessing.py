import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

'''
Steps:
- Denoising? (filter's to remove noise, i.e. low/high-pass, butterworth, etc)
- Scaling
- Resampling
- Feature Extraction?
- Dimensionality Reduction (Discord chat - choose one from the textbook)
- Signal Smoothing (Savitzky Golay)
'''


'''
Preprocessing pipeline for data
'''
class SignalPipeline:
    def __init__(self, config):
        self.config = config
    
    '''
    Encoding: Categorical variables
    Standardization: Numerical variables
    '''
    def scaling(self, train_data, test_data):
        num_scaler = StandardScaler()
        cat_scaler = OrdinalEncoder()
        cat_cols = ['color', 'transparency']
        num_cols = train_data.columns[5:]

        if self.config.get("scaling", False):
            train_data[num_cols] = pd.DataFrame(
                num_scaler.fit_transform(train_data[num_cols]),
                columns=num_cols,
                index=train_data.index
            )
            test_data[num_cols] = pd.DataFrame(
                num_scaler.transform(test_data[num_cols]),
                columns=num_cols,
                index=test_data.index
            )
            train_data[cat_cols] = pd.DataFrame(
                cat_scaler.fit_transform(train_data[cat_cols]),
                columns=cat_cols,
                index=train_data.index
            )
            test_data[cat_cols] = pd.DataFrame(
                cat_scaler.transform(test_data[cat_cols]),
                columns=cat_cols,
                index=test_data.index
            )
        
            return train_data, test_data
        return train_data, test_data
    
    def resampling(self, train_data, test_data):
        pass
    
    def dimensionality_reduction(self, train_data, test_data):
        pass
    
    def preprocess(self, train_data, test_data):
        x_train, x_test = self.scaling(train_data, test_data)
        return x_train, x_test


'''
Here we read the data and split it into training, test set
'''
df = pd.read_csv('data.csv')
train_data, test_data = train_test_split(df, test_size=0.2, shuffle = True, random_state=42)

'''
Below we have a config, which consist of the preprocessing steps applied
'''
config = {
    "scaling": True,
}

pipeline = SignalPipeline(config)
x_train, x_test = pipeline.preprocess(train_data, test_data)
print(x_train)