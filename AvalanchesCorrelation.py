import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import libs.Utils as utils
import libs.Definitions as definition
import statsmodels.api as sm
import PrepareData as data

# Read dataset
df = data.get_clean_dataset()


# Transform categorical data
correlation_df = utils.create_dummy_df(df, ['snow_type', 'trigger_type', 'season'], False)
print(correlation_df.columns)

# Drop those columns which are irrelevant for the question
#correlation_df.drop(columns=['no', 'year', 'day', 'month', 'perimeter_length_m', 'length_m', 'width_m', 'area_m2', 'min_elevation_m', 'x'], inplace=True)
correlation_df = correlation_df[['max_elevation_m', 'aval_size_class', 'weight_AAI', 'max.danger.corr',
                'snow_type_dry', 'snow_type_mixed', 'snow_type_unknown',
                'snow_type_wet', 'trigger_type_EXPLOSIVE', 'trigger_type_HUMAN',
                'trigger_type_NATURAL', 'trigger_type_UNKNOWN', 'season_autumn',
                'season_spring', 'season_summer', 'season_winter',]]

corr = correlation_df.corr(method='pearson')
plt.subplots(figsize=(16, 6))
mask = np.triu(np.ones_like(corr, dtype=np.bool))
heatmap = sns.heatmap(corr, vmin=-1, vmax=1, annot=True, mask=mask)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);
plt.show()

# Select this columns with a correlation over 0.1
# https://towardsdatascience.com/feature-selection-correlation-and-p-value-da8921bfb3cf
# columns = np.full((corr.shape[0],), True, dtype=bool)
# for i in range(corr.shape[0]):
#     for j in range(i+1, corr.shape[0]):
#         if corr.iloc[i,j] >= 0.9:
#             if columns[j]:
#                 columns[j] = False
# selected_columns = correlation_df.columns[columns]
# correlation_df = correlation_df[selected_columns]
# corr = correlation_df.corr()
# plt.subplots(figsize=(20,15))
# sns.heatmap(corr)
# plt.show()


corr = correlation_df.corr()[['max.danger.corr']].sort_values(by='max.danger.corr', ascending=False)
plt.figure(figsize=(8, 12))
heatmap = sns.heatmap(corr, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Features Correlating with Sales Price', fontdict={'fontsize':18}, pad=16)
plt.show()


corr1 = correlation_df.corr()[['snow_type_dry']].sort_values(by='snow_type_dry', ascending=False)
corr2 = correlation_df.corr()[['snow_type_wet']].sort_values(by='snow_type_wet', ascending=False)
df5 = pd.concat([corr1, corr2], axis=1)
df5.reset_index(inplace=True)
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Horizontally stacked subplots')
heatmap1 = sns.heatmap(df5.snow_type_wet, annot=True, cmap='BrBG', ax=ax1)
heatmap2 = sns.heatmap(df5.snow_type_dry, annot=True, cmap='BrBG', ax=ax2)
fig.show()
plt.show()
# df_not_cat = df.copy()
# label_encoder = LabelEncoder()
# df_not_cat['snow_type'] = label_encoder.fit_transform(df_not_cat['snow_type']).astype('float64')
# df_not_cat['trigger_type'] = label_encoder.fit_transform(df_not_cat['trigger_type']).astype('float64')
# corr = df_not_cat.corr()
# plt.subplots(figsize=(20,15))
# sns.heatmap(corr)
# plt.show()

# Has the avalanche change with respect to time?
# Which are the factors that influence the most an avalanche trigger?
# Avalanche activity vs. danger level

