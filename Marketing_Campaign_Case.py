import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split

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
        sns.boxplot(x=marketing_df[col])
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
print(traffic_campaign_table['mobile'])

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

# calculating correlations
print(marketing_df.corr()['visiting_time'])

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

# Fitting generalized linear model without traffic source
log_mod = smf.glm(formula='converted ~ browser + visiting_time + total_amount_due + previous_payment_amount + '
                          'previous_visitor + day_of_week + campaign', data=marketing_df,
               family=sm.families.Binomial()).fit()
print(log_mod.summary())
print(log_mod.null_deviance - log_mod.deviance)
#
# Iterating to find best model
log_mod2 = smf.glm(formula='converted ~ browser + visiting_time + total_amount_due + previous_payment_amount + '
                           'previous_visitor + day_of_week + campaign + campaign:browser', data=marketing_df,
               family=sm.families.Binomial()).fit()
print(log_mod2.summary())
print(log_mod.deviance - log_mod2.deviance)

log_mod3 = smf.glm(formula='converted ~ browser + visiting_time + total_amount_due + previous_payment_amount + '
                           'previous_visitor + day_of_week + campaign + campaign:browser + campaign:visiting_time',
                   data=marketing_df, family=sm.families.Binomial()).fit()
print(log_mod3.summary())
print(log_mod2.deviance - log_mod3.deviance)

log_mod4 = smf.glm(formula='converted ~ browser + visiting_time + total_amount_due + previous_payment_amount + '
                           'previous_visitor + day_of_week + campaign + campaign:browser + campaign:visiting_time + '
                           'campaign:previous_payment_amount',
                   data=marketing_df, family=sm.families.Binomial()).fit()
print(log_mod4.summary())
print(log_mod3.deviance - log_mod4.deviance)

log_mod5 = smf.glm(formula='converted ~ browser + visiting_time + total_amount_due + previous_payment_amount + '
                           'previous_visitor + day_of_week + campaign + campaign:browser + campaign:visiting_time + '
                           'campaign:previous_payment_amount + campaign:total_amount_due',
                   data=marketing_df, family=sm.families.Binomial()).fit()
print(log_mod5.summary())
print(log_mod4.deviance - log_mod5.deviance)

log_mod6 = smf.glm(formula='converted ~ browser + visiting_time + total_amount_due + previous_payment_amount + '
                           'previous_visitor + day_of_week + campaign + campaign:browser + campaign:visiting_time + '
                           'campaign:previous_payment_amount + campaign:total_amount_due + campaign:previous_visitor',
                   data=marketing_df, family=sm.families.Binomial()).fit()
print(log_mod6.summary())
print(log_mod5.deviance - log_mod6.deviance)

log_mod7 = smf.glm(formula='converted ~ browser + visiting_time + total_amount_due + previous_payment_amount + '
                           'day_of_week + campaign + campaign:browser + campaign:visiting_time + '
                           'campaign:previous_payment_amount + campaign:total_amount_due',
                   data=marketing_df, family=sm.families.Binomial()).fit()
print(log_mod7.summary())
print(log_mod6.deviance - log_mod7.deviance)

log_mod8 = smf.glm(formula='converted ~ browser + visiting_time + total_amount_due + previous_payment_amount + '
                           'campaign + campaign:browser + campaign:visiting_time + '
                           'campaign:previous_payment_amount + campaign:total_amount_due',
                   data=marketing_df, family=sm.families.Binomial()).fit()
print(log_mod8.summary())
print(log_mod7.deviance - log_mod8.deviance)

# Centering and dropping main effect of browser to ease interpretation of coef
log_mod_final = smf.glm(formula='converted ~ center(visiting_time) + center(total_amount_due) + center(previous_payment_amount) + '
                           'campaign + campaign:browser + campaign:center(visiting_time) + '
                           'campaign:center(previous_payment_amount) + campaign:center(total_amount_due)',
                   data=marketing_df, family=sm.families.Binomial()).fit()
print(log_mod_final.summary())


# Fitting generalized liner model with traffic source
log_all_traffic = smf.glm(formula='converted ~ browser + visiting_time + total_amount_due + previous_payment_amount + '
                          'previous_visitor + day_of_week + campaign + traffic_source', data=marketing_df,
                  family=sm.families.Binomial()).fit()
print(log_all_traffic.summary())
print(log_all_traffic.null_deviance - log_all_traffic.deviance)

log_all_traffic2 = smf.glm(formula='converted ~ browser + visiting_time + total_amount_due + previous_payment_amount + '
                          'previous_visitor + day_of_week + campaign + traffic_source + traffic_source:campaign', data=marketing_df,
                  family=sm.families.Binomial()).fit()
print(log_all_traffic2 .summary())
print(log_all_traffic.deviance - log_all_traffic2.deviance)

log_all_traffic3 = smf.glm(formula='converted ~ browser + visiting_time + total_amount_due + previous_payment_amount + '
                           'previous_visitor + day_of_week + campaign + traffic_source + traffic_source:campaign + '
                           'campaign:browser', data=marketing_df,
                  family=sm.families.Binomial()).fit()
print(log_all_traffic3 .summary())
print(log_all_traffic2.deviance - log_all_traffic3.deviance)

log_all_traffic4 = smf.glm(formula='converted ~ browser + visiting_time + total_amount_due + previous_payment_amount + '
                           'previous_visitor + day_of_week + campaign + traffic_source + traffic_source:campaign + '
                           'campaign:previous_payment_amount', data=marketing_df,
                  family=sm.families.Binomial()).fit()
print(log_all_traffic4 .summary())
print(log_all_traffic2.deviance - log_all_traffic4.deviance)

log_all_traffic5 = smf.glm(formula='converted ~ browser + visiting_time + total_amount_due + previous_payment_amount + '
                           'previous_visitor + day_of_week + campaign + traffic_source + traffic_source:campaign + '
                           'campaign:total_amount_due', data=marketing_df,
                  family=sm.families.Binomial()).fit()
print(log_all_traffic5 .summary())
print(log_all_traffic2.deviance - log_all_traffic5.deviance)

log_all_traffic6 = smf.glm(formula='converted ~ browser + visiting_time + total_amount_due + previous_payment_amount + '
                           'previous_visitor + day_of_week + campaign + traffic_source + traffic_source:campaign + '
                           'campaign:visiting_time', data=marketing_df,
                  family=sm.families.Binomial()).fit()
print(log_all_traffic6 .summary())
print(log_all_traffic2.deviance - log_all_traffic6.deviance)

log_all_traffic7 = smf.glm(formula='converted ~ browser + total_amount_due + previous_payment_amount + '
                           'previous_visitor + day_of_week + campaign + traffic_source + traffic_source:campaign',
                           data=marketing_df,
                  family=sm.families.Binomial()).fit()
print(log_all_traffic7 .summary())
print(log_all_traffic2.deviance - log_all_traffic7.deviance)

log_all_traffic8 = smf.glm(formula='converted ~ browser + total_amount_due + previous_payment_amount + '
                           'day_of_week + campaign + traffic_source + traffic_source:campaign',
                           data=marketing_df,
                  family=sm.families.Binomial()).fit()
print(log_all_traffic8 .summary())
print(log_all_traffic7.deviance - log_all_traffic8.deviance)

log_all_traffic9 = smf.glm(formula='converted ~ browser + total_amount_due + previous_payment_amount + '
                           'campaign + traffic_source + traffic_source:campaign',
                           data=marketing_df,
                  family=sm.families.Binomial()).fit()
print(log_all_traffic9 .summary())
print(log_all_traffic8.deviance - log_all_traffic9.deviance)

log_all_traffic10 = smf.glm(formula='converted ~ browser + previous_payment_amount + '
                           'campaign + traffic_source + traffic_source:campaign',
                           data=marketing_df,
                  family=sm.families.Binomial()).fit()
print(log_all_traffic10 .summary())
print(log_all_traffic9.deviance - log_all_traffic10.deviance)

log_all_traffic11 = smf.glm(formula='converted ~ browser + '
                           'campaign + traffic_source + traffic_source:campaign',
                           data=marketing_df,
                  family=sm.families.Binomial()).fit()
print(log_all_traffic11 .summary())
# print(log_all_traffic10.deviance - log_all_traffic11.deviance)

log_all_traffic12 = smf.glm(formula='converted ~ '
                           'campaign + traffic_source + traffic_source:campaign',
                           data=marketing_df,
                  family=sm.families.Binomial()).fit()
print(log_all_traffic12 .summary())
print(log_all_traffic11.deviance - log_all_traffic12.deviance)

log_all_traffic_final = smf.glm(formula='converted ~ browser + '
                           'campaign + campaign:C(traffic_source, Treatment("web"))',
                           data=marketing_df,
                  family=sm.families.Binomial()).fit()
print(log_all_traffic_final .summary())

# # creating model to predict traffic source

# # creating dummy variables
# nom_vars = ['browser', 'day_of_week', 'campaign', 'traffic_source']
# for var in nom_vars:
#     nom_dum = pd.get_dummies(marketing_df[var], prefix=var)
#     data = marketing_df.join(nom_dum)
#     marketing_df = data
#
# # Selecting predictors based on model fit to full data
# pred = ['visiting_time', 'total_amount_due', 'previous_payment_amount',
#         'browser_Chrome', 'browser_InternetExplorer', 'browser_Safari', 'browser_Firefox']
#
# log_model_all = sm.MNLogit(marketing_df['traffic_source'], marketing_df[pred])
# log_result_all = log_model_all.fit()
# print(log_result_all.summary())
#
# # Separating data into train and test
# train_x, test_x, train_y, test_y = train_test_split(marketing_df[pred],
#     marketing_df['traffic_source'], train_size=0.7)
#
# # fitting multinomial regression model to train data using predictors selected previously
# mn_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=5000).fit(train_x, train_y)
#
# # looking at model accuracy
# print(metrics.accuracy_score(train_y, mn_lr.predict(train_x)))
# print(metrics.accuracy_score(test_y, mn_lr.predict(test_x)))
#
# # creating new variable using predicted values
# test = mn_lr.predict(marketing_df[pred])
# print(np.count_nonzero(test == 'mobile'))
# print(np.count_nonzero(test == 'web'))
# print(np.count_nonzero(test == 'in_store'))

# checking whether new variable adds anything to model
# marketing_df['pred_traffic'] = test
#
# log_all8 = smf.glm(formula='converted ~ browser + center(visiting_time) + center(total_amount_due) + center(previous_payment_amount) + '
#                            'campaign + campaign:browser + campaign:center(visiting_time) + pred_traffic + '
#                            'campaign:center(previous_payment_amount) + campaign:center(total_amount_due)',
#                    data=marketing_df, family=sm.families.Binomial()).fit()
# print(log_all8.summary())
#
# checking whether new variable can replace true value
# log_all_trafficpred_final = smf.glm(formula='converted ~ browser + '
#                            'campaign + pred_traffic + pred_traffic:campaign',
#                            data=marketing_df,
#                   family=sm.families.Binomial()).fit()
# print(log_all_trafficpred_final .summary())