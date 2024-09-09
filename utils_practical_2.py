# SHARED UTILITIES FOR UCB ML/AI PRACTICAL II
#
# Fazeel Mufti


import pandas as pd
import numpy as np
import time

import matplotlib.pyplot as plt

# Export dataFrame's as images
import dataframe_image as dfi

# Global rules

global logging

# Page width to use for wide figures - make it consistent across the project
page_width=10

# Clean up axes labels
axes_fonts = {'fontweight': 'bold'}
title_fonts = {'fontweight': 'bold', 'fontsize': 14}


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
    return f'{value:,.6f}'

def df_style_floats(df):
    """
    :param df: In-place replacement of the float columns to {:,.4f} format
    :return: Returns the same DF back
    """
    styled_df = df
    float_cols = df.select_dtypes(include='float').columns
    styled_df[float_cols] = styled_df[float_cols].map(format_floats)
    return styled_df


# Evaluate models
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def get_model_metrics_as_results(name, clf, scaler, X_test, y_test, override_rmse=0):
    """
    Build standardized results row given the predictions and y_test values

    :param name: Model name for labeling the row in the table
    :param clf: Fitted classifier to get metrics
    :param scaler: Fitted scaler used for transformation
    :param X_test: Test data used to fit the classifier
    :param y_test: Evaluation data to be used for the metrics
    :param override_rmse: Default 0 will internally calculate RMSE as np.sqrt(MSE), pass in a 
        value for cross-validated estimator result if needed
    :return: Returns single row of results summary table containing:
    
        [model_name, MAE, MSE, RMSE, R2_Score, y-intercept]
    """
    logging.debug(f'Working on {name}')

    # Get predictions
    y_preds = clf.predict(X_test)

    # get metrics
    mae = mean_absolute_error(y_preds, y_test)
    mse = mean_squared_error(y_preds, y_test)
    if override_rmse == 0:
        rmse = np.sqrt(mse)
    else:
        rmse=override_rmse
    r2 = r2_score(y_test, y_preds)
    score = clf.score(X_test, y_test)
    if (name == 'DummyRegressor'):
        y_intercept = 0
    else:
        y_intercept = np.abs(clf.intercept_)
    sname = scaler.__class__.__name__
    
    logging.debug(f'... {name}: Scaler: {sname} MAE: {mae:,.4f}, MSE: {mse:,.4f}, RMSE: {rmse:,.4f}, Override RMSE: {override_rmse:,.4f}, R2: {r2:,.4f}, Score: {score:,.4f}, y-int: {y_intercept:,.4f}')

    return [name, sname, mae, mse, rmse, score, y_intercept]


from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def fit_plot_segments(segment_dict, results_seg, model=None, scaler=None, random_state=42, filename=False):
    """
    Predict and plot prices for selected segment

    :param segment_dict: Dictionary containg segment data and plot information
    :param results_seg: Array to append to for tabulating results
    :param model: Model to use for training
    :param scaler: Scaler to transform data
    :param random_state: Random state for modeling
    :param filename: Optional filename to save the plot
    :returns: Returns the results table that was passed in
    """

    # global logging
    # logging.getLogger().setLevel(logging.DEBUG)

    if model is None:
        logging.debug('No model passed in - nothing to do!')
        
    plt.figure(figsize=(page_width,6))
    # plt.figure(figsize=(8,8))
    
    results_seg = []
    for segment in segment_dict['seg_data']:
        logging.debug(f"Segment: {segment}, #Samples: {len(segment_dict['seg_data'][segment])}")
    
        if len(segment_dict['seg_data'][segment]) <= 0:
            return results_seg
        
        # Predict on the segment
        X = segment_dict['seg_data'][segment].drop('price', axis='columns')
        y = segment_dict['seg_data'][segment]['price']
        logging.debug(f'X: {X.shape} y: {y.shape}')
    
        # transform the data: OneHotEncoding
        X = pd.get_dummies(X, drop_first=True)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
        logging.debug(f'X_train: {X_train.shape} X_test: {X_test.shape} y_train: {y_train.shape} y_test: {y_test.shape}')
    
        # Scale the data
        if scaler is not None:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.fit_transform(X_test)
            logging.debug(f'X_scaled: {X_train_scaled.shape} X_test_scaled: {X_test_scaled.shape}')
    
        model.fit(X_train_scaled, y_train)
        
        # Plot predictions
        y_preds = model.predict(X_test_scaled)
        score = model.score(X_test_scaled, y_test)
        r2 = r2_score(y_test, y_preds)
        y_intercept = model.intercept_
        logging.debug(f'Score: {score}, r2: {r2}, y_int: {y_intercept}')
        y_preds = y_preds + y_intercept
    
        # results_seg[segment_name, #samples, R2, y-intercept, mean-price]
        results_seg.append([segment, len(segment_dict['seg_data'][segment]), r2*100, y_intercept, y_preds.mean()])
    
        perfect_pt_1 = [min(y_test), max(y_test)]
        # perfect_pt_2 = [min(y_test), max(y_test)]
        perfect_pt_2 = [min(y_test), max(y_test)] + y_intercept
        plt.scatter(x=y_test, y=y_preds, label=f'{segment} / ${y_intercept:,.2f}', alpha=0.5)
        plt.plot(perfect_pt_1, perfect_pt_2, linestyle='--', color='red')
    
    plt.xlabel(segment_dict['graph_x_label'], fontdict=axes_fonts)
    plt.ylabel(segment_dict['graph_y_label'], fontdict=axes_fonts)
    plt.title(segment_dict['graph_title'], fontdict=title_fonts)
    plt.legend().set_title('Price Segment: / Base Price')
    plt.tight_layout()

    if (filename):
        plt.savefig(filename)
        
    plt.show()

    # logging.getLogger().setLevel(logging.INFO)
    
    return results_seg


def generate_segments_table(segment_name, results_seg, png_filename=False):
    """
    Generate results table for the selected data segment

    :param segment_name: Segment name to use in Model Description
    :param results_seg: Results array that will be appropriately converted to a DataFrame
    :param png_filename: Optional filename to save the results DF
    :returns: Returns the styled DF
    """
    results_seg_df = pd.DataFrame(results_seg,
                                  columns=[segment_name, '# Cars', 'R2 Score', 'Base Price', 'Avg Price'])
    results_seg_df.set_index(results_seg_df.columns[0], inplace=True)
    
    # Export results for README
    results_seg_df_styled = df_style_floats(results_seg_df)
    
    if (png_filename):
        dfi.export(results_seg_df_styled, png_filename)

    return results_seg_df_styled



def get_cleansed_data(cleanse_data=False, infile='data/vehicles.csv'):
    """
    :param cleanse_data: If True, will cleanse the data based on the decisions made during the DataInvestigation
    :param infile: CSV input file, defaults to 'data/vehicles.csv'
    :return: Returns raw_data_frame and cleansed_data_frame (cleanse_data=True) after applying the data cleansing findings of this assignment
    """

    raw = pd.DataFrame()

    # Read input file
    # infile = 'data/vehicles.csv'
    print('Reading {} ... '.format(infile), end='')
    raw = pd.read_csv(infile)
    print('Done: {}'.format(raw.shape))

    if not cleanse_data:
        return raw
    
    # make a copy of raw data and start to cleanse it
    cleansed = raw.copy()

    # Handle price: remove outliers, mod-zscore column will be dropped later
    print('\nCleansing price column ... ')
    before = cleansed.shape
    print('... Removing price outliers using ModZ method ... ')
    cleansed = remove_outliers_modz(raw, indent_output=True)
    print('... Removed {:,d} outliers'.format(before[0] - cleansed.shape[0]))
    before = cleansed.shape
    print('... Removing cars with price <= 0 ... ', end='')
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
