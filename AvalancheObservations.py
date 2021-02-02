
import numpy as np
import matplotlib.pyplot as plt
import libs.Utils as utils
import PrepareData as data

# Read dataset

# df = pd.read_csv('./Dataset/data_set_1_avalanche_observations_wi9899_to_wi1819_davos.csv', delimiter=';')
# print('Original dataset shape: ', df.shape)
# print(df.columns)
df = data.get_clean_dataset()

# # Select the rows with missing values
# missing_val_tot = 0
# print('Columns with missing val: ', set(df.columns[df.isnull().any()]))
# for missing_val_col in set(df.columns[df.isnull().any()]):
#     # Check how many missing values has every column
#     missing_val = df[df[missing_val_col].isna() == True].shape[0]
#     #print('{} col has {} missing values'.format(missing_val_col, missing_val))
#     missing_val_tot += missing_val
#
# print('We are missing {}% of the avalanche danger level values'.format(missing_val_tot/(df.shape[0])*100))
#
# #Drop missing values, it is not worth it to predict them
# df.dropna(axis=0, inplace=True)
# df.reset_index(inplace=True, drop=True)
#
# ###### Avalanche date ########
# print('The dataset has been populated from {} to {}'.format(df['x'].iloc[0], df['x'].iloc[df.shape[0]-1]))
# print('It contains {} avalanches within {} days'.format(df.shape[0], len(df['x'].unique())))
#
# # Rework date col to divide it into three columns: day, month and year
# date = pd.to_datetime(df['x'])
# date_df = pd.concat([pd.DataFrame(date.array.year),
#                      pd.DataFrame(date.array.month),
#                      pd.DataFrame(date.array.day)],
#                     axis=1)
# date_df.columns = ['year', 'month', 'day']
# df = pd.concat([date_df, df], axis=1)
#
# # Drop the original column
# df.drop(columns=['x'], inplace=True)

# Plot number of avalanches per year
utils.create_bar_plot(df, 'year', 'Year', 'Num avalanches', True)
# From this plot we can see there hasn't been a clear pattern of increasing/decreasing the avalanche activity within time
# In addition, we can see that the data collected the fist seven years is much smaller than the years after.
# Therefore, comparision between years should be done through the entire dataset.

# Group by season categorical value created from the month of release
# df.insert(3, "season", None, True)
# df.loc[[x in definition.winter for x in df['month']], 'season'] = 'winter'
# df.loc[[x in definition.spring for x in df['month']], 'season'] = 'spring'
# df.loc[[x in definition.summer for x in df['month']], 'season'] = 'summer'
# df.loc[[x in definition.autumn for x in df['month']], 'season'] = 'autumn'

# Remove the avalanches triggered in purpose using explosives, those rows can bias our results
print('We have {} avalanches intentionally triggered'.format(len(df[df['trigger_type'] == 'EXPLOSIVE'])))
df2 = df[df['trigger_type'] != 'EXPLOSIVE']

# Plot the avalanche danger level with respect to the four year seasons
utils.create_stacked_bar_plot(df2, 'max.danger.corr', 'season')
#utils.create_stacked_histogram(df2, 'season', 'aval_size_class')

###### Type of snow ########

#utils.create_stacked_bar_plot(df2, 'max.danger.corr', 'snow_type')
#utils.create_bar_plot(df, 'snow_type', 'Snow type', 'Num avalanches', False)

###### Type of trigger ########

#utils.create_stacked_bar_plot(df, 'max.danger.corr', 'trigger_type')
#utils.create_stacked_bar_plot(df, 'trigger_type', 'max.danger.corr')
#utils.create_bar_plot(df, 'snow_type', 'Snow type', 'Num avalanches', False)
df_av_danger_2_4 = df[df['max.danger.corr'] != 1]
df_av_danger_2_4 = df_av_danger_2_4[df_av_danger_2_4['max.danger.corr'] != 5]
df_av_danger_2_4_winter = df_av_danger_2_4[df_av_danger_2_4['season'] == 'winter']
utils.create_bi_bar_plot(df_av_danger_2_4_winter, 'snow_type', 'trigger_type', 'max.danger.corr')
utils.create_bi_bar_plot(df_av_danger_2_4, 'snow_type', 'trigger_type', 'max.danger.corr')

# Aspect degree
df_north, df_northeast, df_east, df_southeast, df_south, df_southwest, df_west, df_northwest = data.get_df_aspect()


weight = [len(df_north),len(df_northeast),len(df_east),len(df_southeast),\
               len(df_south),len(df_southwest),len(df_west),len(df_northwest)]
weight_cmap = np.true_divide(weight, len(df_northeast))
labels = ['N\n'+str(len(df_north)), 'NE\n'+str(len(df_northeast)), 'E\n'+str(len(df_east)), 'SE\n'+str(len(df_southeast)),
         'S\n'+str(len(df_south)), 'SW\n'+str(len(df_southwest)), 'W\n'+str(len(df_west)), 'NW\n'+str(len(df_northwest))]
weights = [1/8,1/8,1/8,1/8,1/8,1/8,1/8,1/8]

utils.create_pie(sizes=weights, labels=labels, list=weight_cmap, startangle=90+22)
utils.create_two_pie(sizes=weights, labels=labels, list=weight_cmap, startangle=90+22)



# Avalanche danger lever
# The danger levels (1-Low, 2-Moderate, 3-Considerable, 4-High, 5-Very High)
danger_level = df['max.danger.corr']
print(df['max.danger.corr'].dtype)
print(danger_level.unique())
utils.create_stacked_histogram(df,'year','max.danger.corr')
utils.create_stacked_histogram(df,'year','max.danger.corr')

labels = []
sizes = []
for avanche_level in sorted(list(danger_level.unique())):
    sizes.append(len(danger_level[danger_level == avanche_level]))
    labels.append('Danger level {}'.format(avanche_level))

utils.create_pie(sizes, labels)



# Avalanche size
df['aval_size_class'] = df['aval_size_class'].astype(int)
avalanche_size = df['aval_size_class']
print('Avalanche size unique:', avalanche_size.unique())

print(df.shape)
small_avalanches = df.shape[0]
df.dropna(subset=['length_m','width_m'], axis=0, inplace=True)
print(df.shape)
small_avalanches = small_avalanches - df.shape[0]
print('Small avalanches in dataset: ', small_avalanches)
#print(df['aval_size_class'].dtype)

avalanche_length = df['length_m']
avalanche_width = df['width_m']

# Keep only the "good" points
# "~" operates as a logical not operator on boolean numpy arrays
df2 = df[utils.is_outlier(points=df['length_m'])]
df_filtered = df2[utils.is_outlier(points=df2['width_m'])]
print('Filter outliers to plot from {} to {} rows'.format(df.shape[0], df_filtered.shape[0]))

avalanche_length_filtered = df_filtered['length_m']
avalanche_width_filtered = df_filtered['width_m']

# Plot the results
fig, (ax1, ax2) = plt.subplots(nrows=2)

ax1.set_title('Original')
ax2.set_title('Without Outliers')

for aval_size in sorted(list(avalanche_size.unique())):
    print('{} avalanches of size {}'.format(len(df.aval_size_class[df['aval_size_class'] == aval_size]), aval_size))
    ax1.scatter(avalanche_width[df['aval_size_class'] == aval_size],
                avalanche_length[df['aval_size_class'] == aval_size], marker='.')
    ax2.scatter(avalanche_width_filtered[df_filtered['aval_size_class'] == aval_size],
                avalanche_length_filtered[df_filtered['aval_size_class'] == aval_size], marker='.')

ax2.legend(('Av. size 1', 'Av. size 2', 'Av. size 3', 'Av. size 4', 'Av. size 5'),
           loc='upper right')
plt.show()

