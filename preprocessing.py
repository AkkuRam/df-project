import pandas as pd
import matplotlib.pyplot as plt

'''
Here we read the 
'''
df = pd.read_csv('data.csv')
df = df[df['class'] != 8]
print(df)