import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import cm

def is_outlier(points, thresh=10):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score < thresh


def create_stacked_histogram(df, x_var, groupby_var):
    # Prepare data
    df_agg = df.loc[:, [x_var, groupby_var]].groupby(groupby_var)
    vals = [df[x_var].values.tolist() for i, df in df_agg]

    y_max = 0
    for i in range(len(vals)):
        counts = len(vals[i])
        if counts > y_max: y_max = counts

    # Draw
    plt.figure(figsize=(10, 9), dpi=80)
    colors = [plt.cm.Spectral(i / float(len(vals) - 1)) for i in range(len(vals))]
    n, bins, patches = plt.hist(vals,
                                df[x_var].unique().__len__(),
                                stacked=True,
                                density=False,
                                color=colors[:len(vals)],
                                align='mid',
                                edgecolor='black',
                                rwidth=20)

    binss = (bins[len(bins)-1]+0.75)/df[x_var].unique().__len__()
    bins2 = []
    for i in range(df[x_var].unique().__len__()):
        bins2.append(binss*i)

    # Decoration
    plt.legend({group: col for group, col in zip(np.unique(df[groupby_var]).tolist(), colors[:len(vals)])})
    plt.title(f"Stacked Histogram of ${x_var}$ colored by ${groupby_var}$", fontsize=12)
    plt.xlabel(x_var)
    plt.ylabel("Avalanche activity")
    plt.ylim(0, y_max)
    #plt.xticks(ticks=bins)
    #plt.xticks(ticks=bins2, labels=list(df[x_var].unique()), rotation=0, horizontalalignment='center')
    plt.show()

def create_pie(sizes, labels, list=[], startangle=338):

    if(len(list) == 0):
        mypie, _ = plt.pie(sizes, labels=labels, labeldistance=1.2, startangle=startangle, shadow=False,
                           counterclock=False)
    else:
        cmap = plt.cm.Reds
        colors = cmap(np.linspace(0., 1., len(labels)))
        colors = cmap(list)

        mypie, _ = plt.pie(sizes, labels=labels, labeldistance=1.2, startangle=startangle, shadow=False, colors=colors, counterclock=False)


    plt.setp(mypie, edgecolor='white')
    plt.axis('equal')
    #plt.legend()
    plt.show()

def create_dummy_df(df, cat_cols, dummy_na):
    '''
    INPUT:
    df - pandas dataframe with categorical variables you want to dummy
    cat_cols - list of strings that are associated with names of the categorical columns
    dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not

    OUTPUT:
    df - a new dataframe that has the following characteristics:
            1. contains all columns that were not specified as categorical
            2. removes all the original columns in cat_cols
            3. dummy columns for each of the categorical columns in cat_cols
            4. if dummy_na is True - it also contains dummy columns for the NaN values
            5. Use a prefix of the column name with an underscore (_) for separating
    '''

    # df_non_cat = df.drop(axis=1, columns=cat_cols, inplace=False)
    # dummy_col = pd.get_dummies(df[cat_cols], dummy_na=dummy_na, prefix_sep="_")
    # new_df = pd.concat([df_non_cat, dummy_col], axis=1)

    for col in  cat_cols:
        try:
            # for each cat add dummy var, drop original column
            dummy_col = pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=False, dummy_na=dummy_na)
            df_non_cat = df.drop(col, axis=1)
            df = pd.concat([df_non_cat, dummy_col], axis=1)

        except:
            continue

    return df




def create_two_pie(sizes, labels, list, startangle=338):
    cmap = plt.cm.Reds
    colors = cmap(np.linspace(0., 1., len(labels)))
    colors = cmap(list)

    # First Ring (outside)
    fig, ax = plt.subplots()
    mypie, _ = ax.pie(sizes, labels=labels, radius=1.3, labeldistance=1.2, startangle=startangle, shadow=False, colors=colors, counterclock=False)
    plt.setp(mypie, width=0.3, edgecolor='white')
    plt.axis('equal')

    # Second Ring (Inside)
    mypie2, _ = ax.pie(sizes, radius=1.3 - 0.3,
                       labels=['52&', 'r','52&', 'r','52&', 'r','52&','i'],  startangle=startangle, labeldistance=0.7, colors=colors, counterclock=False)#
    #[a(0.5), a(0.4),
     #                                                                    a(0.3), b(0.5), b(0.4), c(0.6), c(0.5), c(0.4),
     #                                                                    c(0.3), c(0.2)])
    plt.setp(mypie2,  edgecolor='white')
    plt.margins(0, 0)
    #plt.legend()
    plt.show()


#######################################################################################################################
#######################################################################################################################
# BAR PLOT
#######################################################################################################################
#######################################################################################################################
def create_stacked_bar_plot(df, stacked_col_name, bar_col_name):
    # Prepare data
    stack_labels = sorted(df[stacked_col_name].unique(), reverse=False)
    stack_vals = [df[bar_col_name][df[stacked_col_name] == i] for i in stack_labels]
    bar_labels = df[bar_col_name].unique()
    N = len(bar_labels)

    # TODO: create a color palet depending on the stack array length
    pal = ['#FFC305', '#FF5733', '#C70039', '#900C3F', '#581845']

    # Get the required arrays to correctly plot the data
    # inspiration comes from: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/bar_stacked.html
    bottom = None
    for i in range(len(stack_labels)):
        temp = pd.DataFrame(stack_vals[i])
        stack = [len(temp[temp[bar_col_name] == j]) for j in bar_labels]
        plt.bar(np.arange(N), stack, width=0.5, bottom=bottom, color=pal[i])
        if(bottom == None):
            bottom = stack # First stack in each column
        else:
            bottom = [a + b for a, b in zip(stack, bottom)] # Set were the next stack must start
        print(stack, bottom)

    plt.ylabel('Avalanche activity')
    plt.title('Avalanche Observations')
    plt.xticks(np.arange(N), bar_labels)
    ax = plt.gca() # set color background
    ax.set_facecolor('xkcd:grey')
    ax.grid(zorder=1) #set grid in background
    plt.legend(stack_labels, title='Danger level')

    plt.show()

def create_bar_plot(df, column, x_label, y_label, sort_x):
    feature_counts = []
    for unique_feature in sorted(df[column].unique()):
        feature_counts.append(len(df[df[column] == unique_feature]))
    if(sort_x == True):
        plt.bar(x=sorted(df[column].unique()), height=feature_counts)
    else:
        plt.bar(x=df[column].unique(), height=feature_counts)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def create_bi_bar_plot(df, stacked_col_name1, stacked_col_name2, bar_col_name):
    # Prepare data
    stack_labels1 = sorted(df[stacked_col_name1].unique(), reverse=False)
    stack_vals1 = [df[bar_col_name][df[stacked_col_name1] == i] for i in stack_labels1]
    stack_labels2 = sorted(df[stacked_col_name2].unique(), reverse=False)
    stack_vals2 = [df[bar_col_name][df[stacked_col_name2] == i] for i in stack_labels2]

    bar_labels = sorted(df[bar_col_name].unique())
    N = len(bar_labels)

    # TODO: create a color palet depending on the stack array length (https://colourco.de/)
    pal1 = ['#0F084B', '#3D60A7', '#81B1D5', '#A0D2E7']
    pal2 = ['#0D381E', '#164F2C', '#2A834D', '#7ED493'] #349E5E


    fig, axs = plt.subplots(1, 2, sharex=False, sharey=True, gridspec_kw={'wspace': 0})
    # Get the required arrays to correctly plot the data
    # inspiration comes from: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/bar_stacked.html
    bottom1 = None
    bottom2 = None
    for i in range(len(stack_labels1)):
        temp1 = pd.DataFrame(stack_vals1[i])
        temp2 = pd.DataFrame(stack_vals2[i])
        stack1 = [len(temp1[temp1[bar_col_name] == j]) for j in bar_labels]
        stack2 = [len(temp2[temp2[bar_col_name] == j]) for j in bar_labels]
        stack1 = [a*-1 for a in stack1]
        axs[0].barh(y=np.arange(N), width=stack1, left=bottom1, color=pal1[i])
        axs[1].barh(y=np.arange(N), width=stack2, left=bottom2, color=pal2[i])
        #axs[0].barh(np.arange(N), height=stack1, bottom=bottom1, color=pal[i])
        #axs[1].barh(np.arange(N), height=stack2, bottom=bottom2, color=pal[i])

        if (bottom1 == None):
            bottom1 = stack1  # First stack in each column
            bottom2 = stack2
        else:
            bottom1 = [a + b for a, b in zip(stack1, bottom1)]
            bottom2 = [a + b for a, b in zip(stack2, bottom2)] # Set were the next stack must start


    #plt.ylabel('Avalanche activity')
    #plt.title('Avalanche Observations in relation with the trigger and the snow type')
    fig.set_size_inches(10, 4)

    #plt.yticks(np.arange(N), bar_labels)
    #ax = plt.gca()  # set color background
    axs[0].set_facecolor('xkcd:grey')
    axs[1].set_facecolor('xkcd:grey')
    #axs[0].grid()  # set grid in background
    #axs[1].grid(zorder=2)  # set grid in background
    axs[0].legend(stack_labels1, title='Snow type', loc='center left')
    axs[1].legend(stack_labels2, title='Trigger', loc='center right')
    #Remove all axis -> not helping understanding
    axs[0].get_xaxis().set_visible(False)
    axs[1].get_xaxis().set_visible(False)
    axs[0].get_yaxis().set_visible(False)
    axs[1].get_yaxis().set_visible(False)

    # Get x axis where fits legend
    axs[0].set_xlim([-bottom2[2]-4000, 0])
    axs[1].set_xlim([0, bottom2[2]+4000])
    #axs[0].get_yaxis().tick_right()
    plt.show()