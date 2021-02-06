
import numpy as np
import matplotlib.pyplot as plt
import libs.Utils as utils
import PrepareData as data

# Loead dataset
df = data.get_clean_dataset()

#############
def plot_avalanche_activity_per_year():
    # Plot number of avalanches per year
    utils.create_bar_plot(df, 'year', 'Year', 'Num avalanches', True)

    # From this plot we can see there hasn't been a clear pattern of increasing/decreasing the avalanche activity within time
    # In addition, we can see that the data collected the fist seven years is much smaller than the years after.
    # Therefore, comparision between years should be done through the entire dataset.

#############
def plot_avalanche_activity_index():

    df_winter = df[df['season'] == 'winter']
    print('Number of avalanches {} in {} days -> {} avalanches per day.'.format(df_winter.shape[0],
            len(df_winter['x'].unique()), np.around(df_winter.shape[0] / len(df_winter['x'].unique()), 2)))
    data1=[]
    data2=[]
    for i in sorted(df_winter['max.danger.corr'].unique()):
        # Take a subset of specific avalanche danger level
        df_winter_av_freq = df_winter[df_winter['max.danger.corr'] == i]
        df_av_freq        = df[df['max.danger.corr'] == i]
        print('Level {}: Number of avalanches {} in {} days -> {} avalanches per day.'.format( i, df_winter_av_freq.shape[0],
            len(df_winter_av_freq['x'].unique()), np.around(df_winter_av_freq.shape[0]/len(df_winter_av_freq['x'].unique()), 2)))

        # Gather the data to create the box plot
        data2.append(df_av_freq['x'].value_counts().values)
        data1.append(df_winter_av_freq['x'].value_counts().values)

    # Box plot
    utils.create_box_plot(data2)
    utils.create_box_plot(data1)

    # Plot the avalanche danger level with respect to the four year seasons
    utils.create_stacked_bar_plot(df, 'max.danger.corr', 'season')
    #utils.create_stacked_histogram(df2, 'season', 'aval_size_class')

#############
def plot_avalancge_danger_vs_snow_and_trigger_type():

    # Remove the avalanches triggered in purpose using explosives, those rows can bias our results
    # Rethinking about this topic, I will leave this column since maybe if there were none explosions,
    # the avalanches would have released naturally.
    #print('We have {} avalanches intentionally triggered'.format(len(df[df['trigger_type'] == 'EXPLOSIVE'])))
    #df2 = df[df['trigger_type'] != 'EXPLOSIVE']

    #########
    # Why in winter there are more avalanches when the danger level is 3 than when it is 4?
    # - Is it because the type of snow?
    # - Because people feel more confident to go to the mountain then the level is 3?

    # Plot avalanche danger vs snow type

    # Simple bar plot
    # utils.create_bar_plot(df, 'snow_type', 'Snow type', 'Num avalanches', False)
    #utils.create_bar_plot(df, 'snow_type', 'Snow type', 'Num avalanches', False)

    # Stacked bar plot
    utils.create_stacked_bar_plot(df, 'max.danger.corr', 'snow_type')
    utils.create_stacked_bar_plot(df, 'max.danger.corr', 'trigger_type')


    # Bi directional bar plot
    #Remove danger level 1 and 5
    df_av_danger_2_4 = df[df['max.danger.corr'] != 1]
    df_av_danger_2_4 = df_av_danger_2_4[df_av_danger_2_4['max.danger.corr'] != 5]
    utils.create_bi_bar_plot(df_av_danger_2_4, 'snow_type', 'trigger_type', 'max.danger.corr')
    # Take only the avalanches which were released during winter period
    df_av_danger_2_4_winter = df_av_danger_2_4[df_av_danger_2_4['season'] == 'winter']
    utils.create_bi_bar_plot(df_av_danger_2_4_winter, 'snow_type', 'trigger_type', 'max.danger.corr')

    utils.create_stacked_bar_subplot(df, 'max.danger.corr', ['snow_type', 'trigger_type', 'aval_size_class', 'season'])

#############
def plot_avalanche_activity_vs_aspect():
    # - Is the orientation of the avalanche important?
    # Plot the avalanche activity vs the aspect degree
    df_north, df_northeast, df_east, df_southeast, df_south, df_southwest, df_west, df_northwest = data.get_df_aspect()

    labels = ['N\n'+str(len(df_north)), 'NE\n'+str(len(df_northeast)), 'E\n'+str(len(df_east)), 'SE\n'+str(len(df_southeast)),
             'S\n'+str(len(df_south)), 'SW\n'+str(len(df_southwest)), 'W\n'+str(len(df_west)), 'NW\n'+str(len(df_northwest))]
    coordinates = ['North', 'North-East', 'East', 'South-East', 'South', 'South-West', 'West', 'North-West']

    # For better observation each aspect will have the same space in the pie
    pie_weights = [1/8,1/8,1/8,1/8,1/8,1/8,1/8,1/8]
    # On the other hand is the color of each trunch who determines the weight (number of avalanches)
    weight = [df_north.shape[0], df_northeast.shape[0], df_east.shape[0], df_southeast.shape[0], df_south.shape[0], df_southwest.shape[0],
              df_west.shape[0], df_northwest.shape[0]]
    weight_cmap = np.true_divide(weight, len(df_northeast))

    # Take the percentage of each orientation
    total = sum(weight)
    percentage = np.around(np.true_divide(weight, total), 2)
    percentage_ = np.multiply(percentage, 100)
    percentage_int = [int(i) for i in percentage_]

    percentage_str = [str(len(df_north))+'\n'+str(percentage_int[0])+'%',
              str(len(df_northeast))+'\n'+str(percentage_int[1])+'%',
              str(len(df_east))+'\n'+str(percentage_int[2])+'%',
              str(len(df_southeast))+'\n'+str(percentage_int[3])+'%',
              str(len(df_south))+'\n'+str(percentage_int[4])+'%',
              str(len(df_southwest))+'\n'+str(percentage_int[5])+'%',
              str(len(df_west))+'\n'+str(percentage_int[6])+'%',
              str(len(df_northwest))+'\n'+str(percentage_int[7])+'%']

    #utils.create_pie(sizes=pie_weights, labels=labels, colorweight=weight_cmap, startangle=90+22)
    utils.create_two_pie(sizes1=pie_weights,
                         sizes2=pie_weights,
                         labels1=coordinates,
                         labels2=percentage_str,
                         colorweight=weight_cmap,
                         startangle=90+22)

#############
def plot_avalanche_danger_level():
    # Avalanche danger lever
    # The danger levels (1-Low, 2-Moderate, 3-Considerable, 4-High, 5-Very High)
    danger_level = df['max.danger.corr']
    print(df['max.danger.corr'].dtype)
    print(danger_level.unique())
    utils.create_stacked_bar_plot(df,'year','max.danger.corr')
    utils.create_stacked_bar_plot(df,'max.danger.corr', 'year')

    labels = []
    sizes = []
    for avanche_level in sorted(list(danger_level.unique())):
        sizes.append(len(danger_level[danger_level == avanche_level]))
        labels.append('Danger level {}'.format(avanche_level))

    utils.create_pie(sizes, labels)

#############
def plot_avalanche_size():
    # Avalanche size
    df['aval_size_class'] = df['aval_size_class'].astype(int)
    avalanche_size = df['aval_size_class']
    print('Avalanche size unique:', avalanche_size.unique())

    # print(df.shape)
    # small_avalanches = df.shape[0]
    # df.dropna(subset=['length_m','width_m'], axis=0, inplace=True)
    # print(df.shape)
    # small_avalanches = small_avalanches - df.shape[0]
    # print('Small avalanches in dataset: ', small_avalanches)

    # For better understanding remove the outliers
    df_not_outliers = df[utils.is_outlier(points=df['length_m'])]
    df_filtered = df_not_outliers[utils.is_outlier(points=df_not_outliers['width_m'])]
    print('Filter outliers to plot from {} to {} rows'.format(df.shape[0], df_filtered.shape[0]))

    #  Plot the results
    utils.create_scatter_plot(df_filtered, 'length_m', 'width_m', 'aval_size_class')

    #  Plot the results
    utils.create_scatter_plot(df_filtered, 'length_m', 'width_m', 'orientation')



#############
def plot_bar_plot_cat_features(df4):

    ylabel = ['2-Moderate', '3-Considerable', '4-High']
    labels = sorted(df4['max.danger.corr'].unique())


    label2 = ['MIXED', 'UNKNOWN', 'WET', 'DRY'] #df['snow_type'].unique()

    features_set = []
    for i in label2:
        features_subset = []
        for j in labels:
            condition1 = df4[df4['snow_type'] == i]
            features_subset.append(condition1[condition1['max.danger.corr'] == j].shape[0])
        features_set.append(features_subset)

    pal = ['#0F084B', '#3D60A7', '#81B1D5', '#A0D2E7']
    utils.create_sublabels_bar_plot(features_set, labels, label2, ylabel, pal, 'Snow type')

    label2 = ['EXPLOSIVE', 'HUMAN','NATURAL', 'UNKNOWN']#df4['trigger_type'].unique()

    features_set = []
    for i in label2:
        features_subset = []
        for j in labels:
            condition1 = df4[df4['trigger_type'] == i]
            features_subset.append(condition1[condition1['max.danger.corr'] == j].shape[0])
        features_set.append(features_subset)

    #pal = ['#0D381E', '#164F2C', '#2A834D', '#7ED493']
    pal = ['#1E5631', '#A4DE02', '#4C9A2A', '#76BA1B']
    utils.create_sublabels_bar_plot(features_set, labels, label2, ylabel, pal, 'Trigger type')


#plot_avalanche_activity_per_year()
#plot_avalanche_activity_index()
#plot_avalancge_danger_vs_snow_and_trigger_type()
#plot_avalanche_activity_vs_aspect()
#plot_avalanche_danger_level()
plot_avalanche_size()


#utils.plot_bar_plot_cat_features(df)

df_copy = df.copy()
df_copy['max.danger.corr'][df_copy['max.danger.corr'] == 1] = 2
df_copy['max.danger.corr'][df_copy['max.danger.corr'] == 5] = 4

plot_bar_plot_cat_features(df_copy)

df_level2 = df_copy[df_copy['max.danger.corr'] == 2]
df_level3 = df_copy[df_copy['max.danger.corr'] == 3]
df_level4 = df_copy[df_copy['max.danger.corr'] == 4]

