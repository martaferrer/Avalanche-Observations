import pandas as pd
from pandas.api.types import CategoricalDtype
import libs.Definitions as definition




def get_clean_dataset():
    # Read dataset
    df = pd.read_csv('./Dataset/data_set_1_avalanche_observations_wi9899_to_wi1819_davos.csv', delimiter=';')
    print('Original dataset shape: ', df.shape)
    print(df.columns)

    # Select the rows with missing values
    missing_val_tot = 0
    print('Columns with missing val: ', set(df.columns[df.isnull().any()]))
    for missing_val_col in set(df.columns[df.isnull().any()]):
        # Check how many missing values has every column
        missing_val = df[df[missing_val_col].isna() == True].shape[0]
        #print('{} col has {} missing values'.format(missing_val_col, missing_val))
        missing_val_tot += missing_val

    print('We are missing {}% of the avalanche danger level values'.format(missing_val_tot/(df.shape[0])*100))

    #Drop missing values, it is not worth it to predict them
    df.dropna(axis=0, inplace=True)
    df.reset_index(inplace=True, drop=True)

    ###### Avalanche date ########
    print('The dataset has been populated from {} to {}'.format(df['x'].iloc[0], df['x'].iloc[df.shape[0]-1]))
    print('It contains {} avalanches within {} days'.format(df.shape[0], len(df['x'].unique())))

    # Rework date col to divide it into three columns: day, month and year
    date = pd.to_datetime(df['x'])
    date_df = pd.concat([pd.DataFrame(date.array.year),
                        pd.DataFrame(date.array.month),
                        pd.DataFrame(date.array.day)],
                        axis=1)
    date_df.columns = ['year', 'month', 'day']
    df = pd.concat([date_df, df], axis=1)

    # Drop the original column
    #df.drop(columns=['x'], inplace=True)

    # Group by season categorical value created from the month of release
    df.insert(3, "season", None, True)
    df.loc[[x in definition.winter for x in df['month']], 'season'] = 'winter'
    df.loc[[x in definition.spring for x in df['month']], 'season'] = 'spring'
    df.loc[[x in definition.summer for x in df['month']], 'season'] = 'summer'
    df.loc[[x in definition.autumn for x in df['month']], 'season'] = 'autumn'

    # Group aspect in degrees in orientation
    # TODO: add column to dataset (so far, out od scope)
    # df_north1    = df[(df['aspect_degrees']>=definition.north1[0]) & (df['aspect_degrees']<=definition.north1[1])]
    # df_north2    = df[(df['aspect_degrees']>=definition.north2[0]) & (df['aspect_degrees']<=definition.north2[1])]
    # df_north     = pd.concat([df_north1, df_north2], axis=0)
    # df_south     = df[(df['aspect_degrees']>=definition.south[0]) & (df['aspect_degrees']<=definition.south[1])]
    # df_southwest = df[(df['aspect_degrees']>=definition.southwest[0]) & (df['aspect_degrees']<=definition.southwest[1])]
    # df_southeast = df[(df['aspect_degrees']>=definition.southeast[0]) & (df['aspect_degrees']<=definition.southeast[1])]
    # df_west      = df[(df['aspect_degrees']>=definition.west[0]) & (df['aspect_degrees']<=definition.west[1])]
    # df_east      = df[(df['aspect_degrees']>=definition.east[0]) & (df['aspect_degrees']<=definition.east[1])]
    # df_northeast = df[(df['aspect_degrees']>=definition.northeast[0]) & (df['aspect_degrees']<=definition.northeast[1])]
    # df_northwest = df[(df['aspect_degrees']>=definition.northwest[0]) & (df['aspect_degrees']<=definition.northwest[1])]
    #

    ###### Type of snow ########
    # Set type of snow column as categorical
    snow_type = list(df['snow_type'].unique())
    df.snow_type = df['snow_type'].astype(CategoricalDtype(categories=snow_type, ordered=False))
    print(df['snow_type'].unique())
    df.snow_type = df['snow_type'].str.upper()

    ###### Type of trigger ########
    # Set avalanche trigger column as categorical
    trigger = list(df['trigger_type'].unique())
    df.trigger_type = df['trigger_type'].astype(CategoricalDtype(categories=trigger, ordered=False))
    print(df['trigger_type'].unique())
    df.trigger_type = df['trigger_type'].str.upper()


    return df

def get_df_aspect():
    df = get_clean_dataset()

    ###### Type of trigger ########
    # Aspect degree
    df_north1    = df[(df['aspect_degrees']>=definition.north1[0]) & (df['aspect_degrees']<=definition.north1[1])]
    df_north2    = df[(df['aspect_degrees']>=definition.north2[0]) & (df['aspect_degrees']<=definition.north2[1])]
    df_north     = pd.concat([df_north1, df_north2], axis=0)
    df_south     = df[(df['aspect_degrees']>=definition.south[0]) & (df['aspect_degrees']<=definition.south[1])]
    df_southwest = df[(df['aspect_degrees']>=definition.southwest[0]) & (df['aspect_degrees']<=definition.southwest[1])]
    df_southeast = df[(df['aspect_degrees']>=definition.southeast[0]) & (df['aspect_degrees']<=definition.southeast[1])]
    df_west      = df[(df['aspect_degrees']>=definition.west[0]) & (df['aspect_degrees']<=definition.west[1])]
    df_east      = df[(df['aspect_degrees']>=definition.east[0]) & (df['aspect_degrees']<=definition.east[1])]
    df_northeast = df[(df['aspect_degrees']>=definition.northeast[0]) & (df['aspect_degrees']<=definition.northeast[1])]
    df_northwest = df[(df['aspect_degrees']>=definition.northwest[0]) & (df['aspect_degrees']<=definition.northwest[1])]


    return df_north, df_northeast, df_east, df_southeast, df_south, df_southwest, df_west, df_northwest


