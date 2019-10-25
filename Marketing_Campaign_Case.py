import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# reading in csv file
marketing_df = pd.read_csv('marketing_campaign_takehome.csv')
print(marketing_df.head())

# looking at data information
print(marketing_df.info())

# checking for missing data
print(marketing_df.isnull().sum())

# creating univariate plots to look at data
for col in list(marketing_df):
    if marketing_df[col].dtypes != 'float64':
        sns.countplot(marketing_df[col])
        plt.title(col)
        plt.show()
    else:
        plt.hist(marketing_df[col])
        plt.title(col)
        plt.show()