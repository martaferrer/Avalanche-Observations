import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import libs.Utils as utils
import PrepareData as data

# Read dataset
df = data.get_clean_dataset()


# Transform categorical data
correlation_df = utils.create_dummy_df(df, ['snow_type', 'trigger_type', 'season'], False)
print(correlation_df.columns)

correlation_2 = utils.remap_cat_var(df, ['snow_type', 'trigger_type', 'season'])
print(correlation_df.columns)

# Drop those columns which are irrelevant for the question

selected_col = ['max_elevation_m', 'aval_size_class', 'weight_AAI', 'max.danger.corr',
                 'snow_type_DRY', 'snow_type_MIXED', 'snow_type_WET',
                 'trigger_type_EXPLOSIVE', 'trigger_type_HUMAN', 'trigger_type_NATURAL',
                 'season_autumn', 'season_spring', 'season_summer', 'season_winter']
correlation_df = correlation_df[selected_col]

selected_col = ['season', 'snow_type', 'trigger_type', 'aspect_degrees', 'aval_size_class',
                'max_elevation_m', 'max.danger.corr']
correlation_2 = correlation_2[selected_col]

# Full correlation map
utils.create_correlation_map(df=correlation_2, title='Avalanche features correlation map')
#utils.create_correlation_map(df=df, title='Avalanche features correlation map')
#utils.create_correlation_map(df=correlation_df, title='Avalanche features correlation map')

# From the correlation map we can assume:
# - Avalanche size is not related to avalanche danger level - This is surprising
# - 0.19 - Snow type and season - That is quire obvious
# - 0.12 - Trigger type and season
# - 0.41 - Trigger type and snow
# - 0.17 - Max danger
# Single column correlation map
utils.create_single_col_corr_map(df=correlation_df,
                                 col='max.danger.corr',
                                 title='Avalanche Danger Level correlation map')


corr1 = correlation_df.corr()[['snow_type_DRY']].sort_values(by='snow_type_DRY', ascending=False)
corr2 = correlation_df.corr()[['snow_type_WET']].sort_values(by='snow_type_WET', ascending=False)

# Share index
df_all = pd.concat([corr1, corr2], axis=1)

# Create single dataframes
df1 = pd.DataFrame(df_all.iloc[:,0])
df2 = pd.DataFrame(df_all.iloc[:,1])

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Horizontally stacked subplots')
heatmap1 = sns.heatmap(df1, annot=True, cmap='BrBG', ax=ax1)
heatmap2 = sns.heatmap(df2, annot=True, cmap='BrBG', ax=ax2)
plt.show()




# plt.subplots(figsize=(20,15))
# sns.heatmap(corr)
# plt.show()

# Has the avalanche change with respect to time?
# Which are the factors that influence the most an avalanche trigger?
# Avalanche activity vs. danger level

