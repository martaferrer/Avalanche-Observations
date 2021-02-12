import numpy as np
import libs.Utils as utils
import PrepareData as data

# Load dataset
df = data.get_clean_dataset()


def plot_avalanche_activity_per_year():
    '''
    :def: This function creates a bar plot displaying the avalanche activity (number of avalancher) per year of study
    (21 years in total).

    :return: void
    '''

    # Plot number of avalanches per year
    utils.create_bar_plot(df, 'year', 'Year', 'Num avalanches', True)

    # From this plot we can see there hasn't been a clear pattern of increasing/decreasing the avalanche activity within time
    # In addition, we can see that the data collected the fist seven years is much smaller than the years after.
    # Therefore, comparision between years should be done through the entire data set.

def plot_avalanche_feature_correlation():
    '''
    :def: This function creates the correlation map between a pre-selected columns including categorical variables.

    :return: void
    '''

    #############################
    # CORRELATION MAP 1
    # Transform categorical data
    correlation_df = utils.create_dummy_df(df, ['snow_type', 'trigger_type', 'season'], False)
    print(correlation_df.columns)

    # Select those columns we are interested in displaying
    selected_col = ['max_elevation_m', 'aval_size_class', 'max.danger.corr',
                    'snow_type_DRY', 'snow_type_MIXED', 'snow_type_WET',
                    'trigger_type_EXPLOSIVE', 'trigger_type_HUMAN', 'trigger_type_NATURAL',
                    'season_autumn', 'season_spring', 'season_summer', 'season_winter']
    correlation_df = correlation_df[selected_col]

    # Plot correlation map
    utils.create_correlation_map(df=correlation_df, title='Avalanche features correlation map')

    # Single column correlation map
    utils.create_single_col_corr_map(df=correlation_df,
                                     col='max.danger.corr',
                                     title='Avalanche Danger Level correlation map')
    #############################

    #############################
    # CORRELATION MAP 2
    # Transform categorical data with weights
    # TODO: check how to create categorical values with weights, that looks wrong to me
    #correlation_df2 = utils.remap_cat_var(df, ['snow_type', 'trigger_type', 'season'])
    #print(correlation_df2.columns)

    # Select those columns we are interested in displaying
    #selected_col = ['season', 'snow_type', 'trigger_type', 'aspect_degrees', 'aval_size_class',
    #                'max_elevation_m', 'max.danger.corr']
    #correlation_df2 = correlation_df2[selected_col]

    # Plot correlation map
    #utils.create_correlation_map(df=correlation_df2, title='Avalanche features correlation map')
    #############################

# def plot_avalanche_danger_level():
#     # Avalanche danger lever
#     # The danger levels (1-Low, 2-Moderate, 3-Considerable, 4-High, 5-Very High)
#     danger_level = df['max.danger.corr']
#     print(df['max.danger.corr'].dtype)
#     print(danger_level.unique())
#     utils.create_stacked_bar_plot(df,'year','max.danger.corr')
#     utils.create_stacked_bar_plot(df,'max.danger.corr', 'year')
#
#     labels = []
#     sizes = []
#     for avanche_level in sorted(list(danger_level.unique())):
#         sizes.append(len(danger_level[danger_level == avanche_level]))
#         labels.append('Danger level {}'.format(avanche_level))
#
#     # Plot pie
#     utils.create_pie(sizes, labels)

def plot_avalanche_activity_index():
    '''
    :def: This function creates box plot showing the Avalanche Activity Index (AAI) per day when AAI > 0
    (only data available from dataset)

    :return: void
    '''

    # Get data to plot
    data=[]
    for i in sorted(df['max.danger.corr'].unique()):
        # Take a subset of specific avalanche danger level
        df_av_freq = df[df['max.danger.corr'] == i]

        # Gather the data to create the box plot
        data.append(df_av_freq['x'].value_counts().values)

    # Box plot
    utils.create_box_plot(df=data, xlabel='Avalanche Danger Level', ylabel='Avalanche Activity Index (AAI)')


    # Plot the avalanche danger level with respect to the four year seasons
    utils.create_stacked_bar_plot(df, 'max.danger.corr', 'season')

# def plot_avalancge_danger_vs_snow_and_trigger_type():
#
#     # Remove the avalanches triggered in purpose using explosives, those rows can bias our results
#     # Rethinking about this topic, I will leave this column since maybe if there were none explosions,
#     # the avalanches would have released naturally.
#     #print('We have {} avalanches intentionally triggered'.format(len(df[df['trigger_type'] == 'EXPLOSIVE'])))
#     #df2 = df[df['trigger_type'] != 'EXPLOSIVE']
#
#     #########
#     # Why in winter there are more avalanches when the danger level is 3 than when it is 4?
#     # - Is it because the type of snow?
#     # - Because people feel more confident to go to the mountain then the level is 3?
#
#     # Plot avalanche danger vs snow type
#
#     # Simple bar plot
#     # utils.create_bar_plot(df, 'snow_type', 'Snow type', 'Num avalanches', False)
#     #utils.create_bar_plot(df, 'snow_type', 'Snow type', 'Num avalanches', False)
#
#     # Stacked bar plot
#     utils.create_stacked_bar_plot(df, 'max.danger.corr', 'snow_type')
#     utils.create_stacked_bar_plot(df, 'max.danger.corr', 'trigger_type')
#
#
#     # Bi directional bar plot
#     #Remove danger level 1 and 5
#     df_av_danger_2_4 = df[df['max.danger.corr'] != 1]
#     df_av_danger_2_4 = df_av_danger_2_4[df_av_danger_2_4['max.danger.corr'] != 5]
#     utils.create_bi_bar_plot(df_av_danger_2_4, 'snow_type', 'trigger_type', 'max.danger.corr')
#     # Take only the avalanches which were released during winter period
#     df_av_danger_2_4_winter = df_av_danger_2_4[df_av_danger_2_4['season'] == 'winter']
#     utils.create_bi_bar_plot(df_av_danger_2_4_winter, 'snow_type', 'trigger_type', 'max.danger.corr')
#
#     utils.create_stacked_bar_subplot(df, 'max.danger.corr', ['snow_type', 'trigger_type', 'aval_size_class', 'season'])

def plot_bar_plot_cat_features():
    '''
    :def: This function creates bar plot of the snow type and trigger type with respect to the danger level.
    Both plots follow the same structure.

    :return: void
    '''

    # Remove danger level 1 and 5 from the dataset (better readability of the plots afterwards)
    df_small_set = df[df['max.danger.corr'] != 1]
    df_small_set = df_small_set[df_small_set['max.danger.corr'] != 5]

    # Variables from x axis (value and string)
    xlabel_str = ['2-Moderate', '3-Considerable', '4-High']
    xlabels_val = sorted(df_small_set['max.danger.corr'].unique())


    ##### SNOW TYPE
    # Variables reported in the y axis string - ['MIXED', 'UNKNOWN', 'WET', 'DRY'] #
    ylabel_str = df['snow_type'].unique()

    # Create the arrays to get the bar plot
    features_set = []
    for i in ylabel_str:
        features_subset = []
        for j in xlabels_val:
            condition1 = df_small_set[df_small_set['snow_type'] == i]
            features_subset.append(condition1[condition1['max.danger.corr'] == j].shape[0])
        features_set.append(features_subset)

    # Nice blue palette
    pal = ['#0F084B', '#3D60A7', '#81B1D5', '#A0D2E7']

    # Display plot
    utils.create_sublabels_bar_plot(features_set, xlabels_val, ylabel_str, xlabel_str, pal, 'Snow type')


    ##### TRIGGER TYPE
    # Variables reported in the y axis string - ['EXPLOSIVE', 'HUMAN','NATURAL', 'UNKNOWN']
    ylabel_str = df['trigger_type'].unique()

    # Create the arrays to get the bar plot
    features_set = []
    for i in ylabel_str:
        features_subset = []
        for j in xlabels_val:
            condition1 = df_small_set[df_small_set['trigger_type'] == i]
            features_subset.append(condition1[condition1['max.danger.corr'] == j].shape[0])
        features_set.append(features_subset)

    # Nice green palette
    pal = ['#1E5631', '#A4DE02', '#4C9A2A', '#76BA1B']

    # Display plot
    utils.create_sublabels_bar_plot(features_set, xlabels_val, ylabel_str, xlabel_str, pal, 'Trigger type')

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

def plot_avalanche_size():
    '''
    :def: This function creates scatter plot showing the avalanches with respect to their length and width.
    '''

    # Avalanche size
    df['aval_size_class'] = df['aval_size_class'].astype(int)
    avalanche_size = df['aval_size_class']
    print('Avalanche size unique:', avalanche_size.unique())

    # For better understanding remove the outliers
    df_not_outliers = df[utils.is_outlier(points=df['length_m'])]
    df_filtered = df_not_outliers[utils.is_outlier(points=df_not_outliers['width_m'])]
    print('Filtered number of entries {} out of {}'.format(df.shape[0], df_filtered.shape[0]))

    #  Plot the results
    utils.create_scatter_plot(df=df_filtered,
                              colx='length_m', xlabel = 'Avalanche length (m)',
                              coly='width_m', ylabel = 'Avalanche width (m)',
                              color_class='aval_size_class')





# 1
plot_avalanche_activity_per_year()

# 2
plot_avalanche_feature_correlation()

#3 - not used [deprecated check #4]
#plot_avalanche_danger_level()

#4
plot_avalanche_activity_index()

#5 - not used [deprecated check #7]
#plot_avalancge_danger_vs_snow_and_trigger_type()

#6
plot_bar_plot_cat_features()

#7 - not used [TODO: for the time being it is not used in the project]
#plot_avalanche_activity_vs_aspect()

#8
plot_avalanche_size()


