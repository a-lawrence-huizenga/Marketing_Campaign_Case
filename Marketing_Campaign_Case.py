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

# creating tables to look at relationship between variables and conversion
day_table = pd.crosstab(marketing_df['converted'], marketing_df['day_of_week'], normalize='columns')
print(day_table)

browser_table = pd.crosstab(marketing_df['converted'], marketing_df['browser'], normalize='columns')
print(browser_table)

campaign_table = pd.crosstab(marketing_df['converted'], marketing_df['campaign'], normalize='columns')
print(campaign_table)

traffic_table = pd.crosstab(marketing_df['converted'], marketing_df['traffic_source'], normalize='columns')
print(traffic_table)

previous_table = pd.crosstab(marketing_df['converted'], marketing_df['previous_visitor'], normalize='columns')
print(previous_table)

day_campaign_table = pd.crosstab(marketing_df['converted'], [marketing_df['day_of_week'], marketing_df['campaign']], normalize='columns')
print(day_campaign_table)

browser_campaign_table = pd.crosstab(marketing_df['converted'], [marketing_df['browser'], marketing_df['campaign']],
                                     normalize='columns')
print(browser_campaign_table['Chrome'])
print(browser_campaign_table['Firefox'])
print(browser_campaign_table['InternetExplorer'])
print(browser_campaign_table['Safari'])

traffic_campaign_table = pd.crosstab(marketing_df['converted'], [marketing_df['traffic_source'], marketing_df['campaign']],
                                    normalize='columns')
print(traffic_campaign_table)

previous_campaign_table = pd.crosstab(marketing_df['converted'], [marketing_df['previous_visitor'], marketing_df['campaign']],
                                      normalize='columns')
print(previous_campaign_table)

browser_traffic_campaign_table = pd.crosstab(marketing_df['converted'], [marketing_df['traffic_source'],
                                             marketing_df['campaign'], marketing_df['browser']], normalize='columns')
print(browser_traffic_campaign_table)

# creating pairplot for float data
sns.pairplot(marketing_df[['visiting_time', 'total_amount_due', 'previous_payment_amount']])
plt.show()

# creating boxplots to look at relationship between float data and conversion
marketing_df.boxplot('visiting_time', 'converted')
plt.show()

marketing_df.boxplot('total_amount_due', 'converted')
plt.show()

marketing_df.boxplot('previous_payment_amount', 'converted')
plt.show()

# creating pivot tables to look at relationships between float data, campaign, and conversion
visit_time_table = pd.pivot_table(marketing_df, values='visiting_time', index='campaign', columns='converted', margins=True)
print(visit_time_table)

total_table = pd.pivot_table(marketing_df, values='total_amount_due', index='campaign', columns='converted', margins=True)
print(total_table)

previous_table = pd.pivot_table(marketing_df, values='previous_payment_amount', index='campaign', columns='converted', margins=True)
print(previous_table)

# creating pivot tables to look at whether browsers differ since browser may interact with campaign conversion
visit_time_table = pd.pivot_table(marketing_df, values='visiting_time', index='browser', columns='converted', margins=True)
print(visit_time_table)

total_table = pd.pivot_table(marketing_df, values='total_amount_due', index='browser', columns='converted', margins=True)
print(total_table)

previous_table = pd.pivot_table(marketing_df, values='previous_payment_amount', index='browser', columns='converted', margins=True)
print(previous_table)
