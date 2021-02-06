import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

    for col in  cat_cols:
        try:
            # for each cat add dummy var, drop original column
            dummy_col = pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=False, dummy_na=dummy_na)
            df_non_cat = df.drop(col, axis=1)
            df = pd.concat([df_non_cat, dummy_col], axis=1)

        except:
            continue

    return df

def remap_cat_var(df, cat_cols):
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
    df_copy = df.copy()
    for col in  cat_cols:
        unique_val = df_copy[col].unique()
        for i in range(len(unique_val)):
            df_copy[col][df_copy[col] == unique_val[i]] = i
        df_copy[col] = df_copy[col].astype('int64')

    return df_copy

# def create_stacked_histogram(df, x_var, groupby_var):
#     # Prepare data
#     df_agg = df.loc[:, [x_var, groupby_var]].groupby(groupby_var)
#     vals = [df[x_var].values.tolist() for i, df in df_agg]
#
#     y_max = 0
#     for i in range(len(vals)):
#         counts = len(vals[i])
#         if counts > y_max: y_max = counts
#
#     # Draw
#     plt.figure(figsize=(10, 9), dpi=80)
#     colors = [plt.cm.Spectral(i / float(len(vals) - 1)) for i in range(len(vals))]
#     n, bins, patches = plt.hist(vals,
#                                 df[x_var].unique().__len__(),
#                                 stacked=True,
#                                 density=False,
#                                 color=colors[:len(vals)],
#                                 align='mid',
#                                 edgecolor='black',
#                                 rwidth=20)
#
#     binss = (bins[len(bins)-1]+0.75)/df[x_var].unique().__len__()
#     bins2 = []
#     for i in range(df[x_var].unique().__len__()):
#         bins2.append(binss*i)
#
#     # Decoration
#     plt.legend({group: col for group, col in zip(np.unique(df[groupby_var]).tolist(), colors[:len(vals)])})
#     plt.title(f"Stacked Histogram of ${x_var}$ colored by ${groupby_var}$", fontsize=12)
#     plt.xlabel(x_var)
#     plt.ylabel("Avalanche activity")
#     plt.ylim(0, y_max)
#     #plt.xticks(ticks=bins)
#     #plt.xticks(ticks=bins2, labels=list(df[x_var].unique()), rotation=0, horizontalalignment='center')
#     plt.show()

def create_scatter_plot(df, colx, coly, stackedcol):
    color = ['#FFC305', '#FF5733', '#C70039', '#900C3F']
    x = df[colx]
    y = df[coly]

    fig, ax = plt.subplots()
    i=0
    for val in sorted(list(df[stackedcol].unique())):
        ax.scatter(df[colx][df[stackedcol] == val],
                    df[coly][df[stackedcol] == val])#,
                    #marker='.')#,
                    #c=color[i])

        i+=1

    ax.scatter(df[colx][df['max.danger.corr'] == 4],
               df[coly][df['max.danger.corr'] == 4], marker = '.', color='black', s=0.9)

    leg = plt.legend(loc='upper right', title="\n\u2022 Avalanche with \n risk 4-High")
    #h = [plt.plot([], [], color="gray", marker="o", ms=i, ls="")[0] for i in range(5, 13)]
    ax.add_artist(leg)

    ax.legend([ 'Size '+ str(i) for i in sorted(list(df[stackedcol].unique()))],
                    loc='lower right')

    #ax.text(10, 10, 'Danger level 4-High', fontsize=15)

    plt.xlabel('Avalanche length')
    plt.ylabel('Avalanche width')
    ax = plt.gca()  # set color background
    ax.set_facecolor('xkcd:grey')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def create_pie(sizes, labels, colorweight=None, startangle=0, labeldistance=1.2, radius=1):

    if colorweight is not None:
        cmap = plt.cm.Blues
        colors = cmap(colorweight)
    else:
        colors = None

    mypie, _ = plt.pie(x=sizes,
                        labels=labels,
                        labeldistance=labeldistance,
                        radius=radius,
                        startangle=startangle,
                        shadow=False,
                        colors=colors,
                        counterclock=False)


    plt.setp(mypie, edgecolor='white')
    plt.axis('equal')
    #plt.legend()
    plt.show()

def create_two_pie(sizes1, sizes2, labels1, labels2, colorweight=None, startangle=None, labeldistance=1.2):

    if colorweight is not None:
        cmap = plt.cm.Blues
        colors = cmap(colorweight)
    else:
        colors = None

    # First Ring (outside)
    fig, ax = plt.subplots()
    mypie, _ = ax.pie(sizes1,
                      labels=labels1,
                      radius=1,
                      labeldistance=1.1,
                      startangle=startangle,
                      shadow=False,
                      colors=colors,
                      counterclock=False)
    # Create the donut shape
    plt.setp(mypie, width=0.3, edgecolor='white')
    plt.axis('equal')

    # Second Ring (Inside)
    mypie2, _ = ax.pie(sizes2,
                       radius=1, #- 0.3,
                       labels=labels2,
                       startangle=startangle,
                       labeldistance=0.6,
                       colors=colors,
                       counterclock=False)
    # Create the donut shape
    plt.setp(mypie2,  edgecolor='white')

    #Set no margin between the inside and outsite pie
    plt.margins(0, 0)
    plt.show()

def create_tri_pie(sizes1, sizes2, size3, labels1, labels2, labels3, colorweight=None, startangle=0, labeldistance=1.2):

    if colorweight is not None:
        cmap = plt.cm.Reds
        colors = cmap(colorweight)
    else:
        colors = None

    # First Ring (outside)
    fig, ax = plt.subplots()
    mypie, _ = ax.pie(sizes1,
                      labels=labels1,
                      radius=1.3,
                      labeldistance=1.2,
                      startangle=startangle,
                      shadow=False,
                      colors=colors,
                      counterclock=False)
    # Create the donut shape
    plt.setp(mypie, width=0.3, edgecolor='white')
    plt.axis('equal')

    # Second Ring (Inside)
    mypie2, _ = ax.pie(sizes2,
                       radius=1.3 - 0.3,
                       labels=labels2,
                       startangle=startangle,
                       labeldistance=0.7,
                       colors=colors,
                       counterclock=False)
    # Create the donut shape
    plt.setp(mypie2,  edgecolor='white')


    # Third Ring (Inside)
    mypie2, _ = ax.pie(size3,
                       radius=1 - 0.3,
                       labels=labels3,
                       startangle=startangle,
                       labeldistance=0.3,
                       colors=colors,
                       counterclock=False)
    # Create the donut shape
    plt.setp(mypie2,  edgecolor='white')

    #Set no margin between the inside and outsite pie
    plt.margins(0, 0)
    plt.show()


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
        plt.bar(np.arange(N), stack, width=0.5, bottom=bottom, color=None)
        if(bottom == None):
            bottom = stack # First stack in each column
        else:
            bottom = [a + b for a, b in zip(stack, bottom)] # Set were the next stack must start

    plt.ylabel('Avalanche activity')
    plt.title('Avalanche Observations')
    plt.xticks(np.arange(N), bar_labels)
    ax = plt.gca() # set color background
    ax.set_facecolor('xkcd:grey')
    ax.grid(zorder=1) #set grid in background
    plt.legend(stack_labels, title='Danger level')

    plt.show()

def create_stacked_bar_subplot(df, stacked_col_name, bar_col_name):

    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Sharing x per column, y per row')
    for k in range(len(bar_col_name)):
        if (k == 0): m = 0; k = 0
        if (k == 1): m = 0; k = 1
        if (k == 2): m = 1; k = 0
        if (k == 3): m = 1; k = 1
        # Prepare data
        stack_labels = sorted(df[stacked_col_name].unique(), reverse=False)
        stack_vals = [df[bar_col_name[k]][df[stacked_col_name] == i] for i in stack_labels]
        bar_labels = df[bar_col_name[k]].unique()
        N = len(bar_labels)

        # TODO: create a color palet depending on the stack array length
        pal = ['#FFC305', '#FF5733', '#C70039', '#900C3F', '#581845']

        # Get the required arrays to correctly plot the data
        # inspiration comes from: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/bar_stacked.html
        bottom = None
        for i in range(len(stack_labels)):
            temp = pd.DataFrame(stack_vals[i])
            stack = [len(temp[temp[bar_col_name[k]] == j]) for j in bar_labels]
            axs[m,k].bar(np.arange(N), stack, width=0.5, bottom=bottom, color=None)
            if(bottom == None):
                bottom = stack # First stack in each column
            else:
                bottom = [a + b for a, b in zip(stack, bottom)] # Set were the next stack must start

        #plt.ylabel('Avalanche activity')
        #plt.title('Avalanche Observations')
        #axs[m,k].xticks(np.arange(N), bar_labels)
        #axs[m,k].set(xlabel=bar_labels)

        ax = plt.gca() # set color background
        axs[m,k].set_facecolor('xkcd:grey')
        axs[m,k].grid(zorder=1) #set grid in background
        #axs[m,k].legend(stack_labels, title='Danger level')

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

def create_sublabels_bar_plot(features_set, labels, label2, ylabel, color, title):

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()

    rects1 = ax.bar(x - width - width / 2, features_set[0], width, label=label2[0], color=color[0])
    rects2 = ax.bar(x - width / 2,         features_set[1], width, label=label2[1], color=color[1])
    rects3 = ax.bar(x + width / 2,         features_set[2], width, label=label2[2], color=color[2])
    rects4 = ax.bar(x + width + width / 2, features_set[3], width, label=label2[3], color=color[3])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of avalanches')
    ax.set_xlabel('Avalanche Danger Level')
    #ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(ylabel)
    ax.legend(title=title)

    # def autolabel(rects):
    #     """Attach a text label above each bar in *rects*, displaying its height."""
    #     for rect in rects:
    #         height = rect.get_height()
    #         ax.annotate('{}'.format(height),
    #                     xy=(rect.get_x() + rect.get_width() / 2, height),
    #                     xytext=(0, 3),  # 3 points vertical offset
    #                     textcoords="offset points",
    #                     ha='center', va='bottom')
    #
    # autolabel(rects1)
    # autolabel(rects2)
    # autolabel(rects3)
    # autolabel(rects4)

    ax = plt.gca()  # set color background
    ax.set_facecolor('xkcd:grey')
    ax.grid(zorder=1)
    ax.set_axisbelow(True)

    fig.tight_layout()
    plt.show()



def create_correlation_map(df, title):
    corr = df.corr(method='pearson')
    plt.subplots(figsize=(16, 6))
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    heatmap = sns.heatmap(data=corr, vmin=-1, vmax=1, annot=True, mask=mask)
    heatmap.set_title(title, fontdict={'fontsize': 12}, pad=12);
    plt.show()

def create_single_col_corr_map(df, col, title):
    corr = df.corr()[[col]].sort_values(by=col, ascending=False)
    heatmap = sns.heatmap(data=corr, vmin=-1, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title(title)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.show()


def create_box_plot(data, notch=False):
    fig1, ax1 = plt.subplots()
    #ax1.set_title('Basic Plot')
    bp = ax1.boxplot(data, notch=notch)
    plt.setp(bp['boxes'], color='black')
    plt.xticks([1, 2, 3, 4, 5], ['1-Low', '2-Moderate', '3-Considerable', '4-High', '5-Very high'])
    ax1.set_xlabel('Avalanche Danger Level')
    ax1.set_ylabel('Avalanche Activity Index (AAI)')
    ax1.grid(zorder=1)
    # set grid in background
    ax1.set_axisbelow(True)
    plt.show()