# SHARED UTILITIES FOR UCB ML/AI PRACTICAL II
#
# Fazeel Mufti


import pandas as pd
import numpy as np
import time


# Calculate Mod-Z score
# Converted to a funciton so I can call it from get_cleansed_data() to refresh the data repeatedly
#
# Source: https://medium.com/towards-data-science/3-simple-statistical-methods-for-outlier-detection-db762e86cd9d

def remove_outliers_modz(df_with_outliers, indent_output=False):
    """
    Removes outliers based on the Modified Z-score covered in the article

    :param df_with_outliers: Raw data to be cleaned
    :param indent_output: Set to true if you're embedding this inisde your print statements
    :return: Returns DF with outliers removed
    """
        
    # Apply the above function to the entire column to get a modified
    # z score for every data point.
    # temp = vehicles_raw#[0:100000]
    temp = df_with_outliers
    modz = (np.abs(temp['price'] - temp['price'].median())).median()
    med = temp['price'].median()
    const = 0.6745
    print('... ', end='') if indent_output else None
    print('ModZ: {}, med: {}, const: {}'.format(modz, med, const))
    
    start_time = time.time()
    temp['mod_zscore'] = temp['price'].transform(lambda x: (const * (x - med) / modz))
    end_time = time.time()
    print('... ', end='') if indent_output else None
    print('Time: {}'.format(end_time - start_time))
    
    return temp[(temp['mod_zscore'] >= -3.5) & (temp['mod_zscore'] <= 3.5)]


def value_counts_2df(df1, df2, df1_label, df2_label, column, dropna=True):
    """
    Returns the value_counts of two DataFrames that have a common column

    :param df1: First DF
    :param df2: Second DF
    :param df1_label: Label for the counts of DF1
    :param df2_label: Label for the counts of DF2
    :param column: Column to count the values
    :param dropna: Default is true, over-ride to get NaN counts also
    """
        
    # Count the occurrences of each key in both DataFrames separately
    count_df1 = df1[column].value_counts(dropna=dropna).reset_index()
    count_df1.columns = [column, df1_label]
    
    count_df2 = df2[column].value_counts(dropna=dropna).reset_index()
    count_df2.columns = [column, df2_label]
    
    # Merge the counts into a single DataFrame
    result = pd.merge(count_df1, count_df2, on=column, how='outer').fillna(0)
    
    # Convert counts to integers
    result[df1_label] = result[df1_label].astype(int)
    result[df2_label] = result[df2_label].astype(int)

    return result

def format_floats(value):
    """
    :param value: Value to format as float
    :returns: Returns the formatted float
    """
    return f'{value:,.5f}'

def df_style_floats(df):
    """
    :param df: In-place replacement of the float columns to {:,.4f} format
    :return: Returns the same DF back
    """
    styled_df = df
    float_cols = df.select_dtypes(include='float').columns
    styled_df[float_cols] = styled_df[float_cols].map(format_floats)
    return styled_df

def get_cleansed_data(infile='data/vehicles.csv'):
    """
    :param infile: CSV input file, defaults to 'data/vehicles.csv'
    :return: Returns raw_data_frame and cleansed_data_frame after applying the data cleansing findings of this assignment
    """

    raw = pd.DataFrame()

    # Read input file
    # infile = 'data/vehicles.csv'
    print('Reading {} ... '.format(infile), end='')
    raw = pd.read_csv(infile)
    print('Done: {}'.format(raw.shape))

    # make a copy of raw data and start to cleanse it
    cleansed = raw.copy()

    # Handle price: remove outliers, mod-zscore column will be dropped later
    print('\nCleansing price column ... ')
    before = cleansed.shape
    print('... Removing price outliers using ModZ method ... ')
    cleansed = remove_outliers_modz(raw, indent_output=True)
    print('... Removed {:,d} outliers'.format(before[0] - cleansed.shape[0]))
    before = cleansed.shape
    print('... Removing cars with price = 0 ... ', end='')
    cleansed = cleansed.query('price > 0')
    print(' Removed {:,d} rows'.format(before[0] - cleansed.shape[0]))
    after = cleansed.shape
    print('Done: {} -> {}'.format(before, after))

    # Handle dropna() columns
    dropna_cols = ['year', 'manufacturer', 'fuel', 'title_status', 'odometer', 'transmission']

    before_t = cleansed.shape
    print('\nDropNA from columns: ')
    null_cnt = 0
    for col in dropna_cols:
        before = cleansed.shape[0]
        null_cnt += cleansed[col].isnull().sum()
        print('... {}: {:,d} rows ({:,.2f}% of total): {:,d} -> {:,d}'
              .format(col, null_cnt, (1 - (before - null_cnt) / before) * 100, before, before - null_cnt))

    cleansed = cleansed.dropna(subset=dropna_cols)
    print('Done: {} -> {}'.format(before_t, cleansed.shape))

    # handle columns to be dropped
    drop_cols = ['mod_zscore', 'id', 'model', 'VIN']
    before = cleansed.shape
    print('\nDropping columns: ' + str(drop_cols))
    for col in drop_cols:
        print('... {}'.format(col))
    cleansed = cleansed.drop(columns=drop_cols)
    after = cleansed.shape
    print('Done: {} -> {}'.format(before, after))

    # Handle data type changes
    print('\nData Transformations:')
    fl_to_int_cols = ['year', 'odometer']
    for col in fl_to_int_cols:
        print('... {} float -> int: '.format(col), end='')
        cleansed[col] = cleansed[col].astype(int)
        print('Done')

    # Make columnt cagtegorical 
    print('\nCategory Transformations:')
    obj_to_cat_cols = ['condition', 'manufacturer', 'cylinders', 'fuel', 
                       'title_status', 'state', 'transmission', 'drive', 'size', 'type', 'paint_color']
    for col in obj_to_cat_cols:
        print('... Converting column "{}" -> Category: '.format(col), end='')
        cleansed[col] = cleansed[col].astype('category')
        print('Done')
    
    print('\nReturned Raw({:,d}x{}) and Cleansed({:,d}x{}) data'.format(raw.shape[0], raw.shape[1], 
                                                                        cleansed.shape[0], cleansed.shape[1]))
    print('Dataset reduced by {:,d} rows (preserved {:,.2f}% of total)'.format(raw.shape[0] - cleansed.shape[0],
                                                                              (cleansed.shape[0] / raw.shape[0]) * 100))
    
    return raw, cleansed

# r, c = get_cleansed_data()
# print('raw[{}], cleansed[{}]\n\n'.format(r.shape, c.shape))
# c.info()
