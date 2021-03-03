import altair as alt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn import tree
from sklearn.compose import make_column_selector
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

# sns.regplot(y_test,predictions) estimations plotting


file_ramen = '/Users/joaquinramiro/ximo/regression-analysis/resources/data/ramen-ratings.csv'
count = 'Count'
country = 'Country'
brand = 'Brand'
variety = 'Variety'
style = 'Style'
rev = 'Review #'
stars = 'Stars'

models_dict = {}
models_dict['Ridge'] = Ridge()
models_dict['RandomForest'] = RandomForestRegressor(max_depth=12, random_state=0, max_features=6)
models_dict['tree.DecisionTreeRegressor()'] = tree.DecisionTreeRegressor()


def fix_cat(df, col_name):
    '''
    Transform label columns into category fields, adds the encoded value for that label.
    :param df:  df to process.
    :param col_name: column to transform to category.
    :return: df with 1 additional column with the encoded value.
    '''

    df_copy = df.copy()
    df_col = df[[col_name]]

    df_copy[col_name] = df_col[col_name].astype('category')  # Assigning numerical values and storing in another column
    df_copy[col_name + '_Cat'] = df_copy[col_name].cat.codes
    return df_copy


def group_col(df, column_name):
    df_count = df.groupby(column_name).count().reset_index()
    df_mean = df.groupby(column_name)[stars].mean().reset_index()
    df_mean[count] = df_count[stars]
    return df_mean.sort_values(by=[stars, count], ascending=False)


def agg_average(df, col_name):
    """
    Method to group df by a column name cleaning low frequency values
     and get average stars rate and count of each value.
    :param df: df to clean and group.
    :param col_name: column to group by.
    :return: dataframe grouped.
    """
    df_grouped = group_col(df, col_name)
    cut_off = df_grouped[count].mean() / 2.0
    df_grouped_clean = df_grouped[df_grouped[count] > cut_off]
    return df_grouped_clean


def plot_df_column(df, col_name):
    '''
    For a given df , groups by column and plots the average stars
    :param df:  df to use
    :param col_name: column to groupby
    :return:
    '''
    df_col = agg_average(df, col_name)
    chart_col = alt.Chart(df_col).mark_bar().encode(
        x=col_name + ':N',
        y='Stars:Q')
    mean = alt.Chart(df_col).mark_rule(color='red').encode(
        y='mean(Stars):Q'
    )
    st.altair_chart(chart_col + mean)


def get_model_train():
    '''
    Render list of models and returned the selected models.
    :return: model
    '''
    list_options = list(models_dict.keys())
    option = st.selectbox(
        'Model',
        list_options)
    st.write('You selected:', option)
    return models_dict[option]


def get_plot_for_column():
    '''
    Render list of features to get value distribution information.
    :return: feature
    '''
    list_options = [brand, country, style]
    option = st.selectbox(
        'Feature',
        list_options)
    st.write('You selected:', option)
    return option


def print_column_info(df):
    '''
    Renders graph for feature.
    :param df:
    :return:
    '''
    list_options = [brand, country, style]
    option = st.selectbox(
        'Explore feature',
        list_options)
    st.write('You selected:', option)
    df_group = df.groupby(option).count().reset_index()
    df_count = df_group[stars]
    st.dataframe(df_count.describe())


def run_cv_model(model, X, y):
    '''
    Runs CrossValidation for a model and dataset
    :param model: model to run
    :param X: features
    :param y: target
    :return: void
    '''
    one_hot_encoder = make_column_transformer(
        (OneHotEncoder(sparse=False, handle_unknown='ignore'),
         make_column_selector(dtype_include='category')),
        remainder='passthrough')
    # pipeline = Pipeline(one_hot_encoder, model)
    pipeline = make_pipeline(one_hot_encoder, model)
    cv_results = cross_validate(pipeline, X, y, cv=4, scoring='neg_root_mean_squared_error', verbose=1,
                                n_jobs=6)
    # print(sorted(cv_results.keys()))
    print("Model: " + str(model))
    print("test_score")
    print(cv_results['test_score'])
    print("average: ", np.average(cv_results['test_score']))
    return cv_results['test_score']


def full_training(model, X, y):
    '''
    Trains a model over full split. Prints error adn plots estimations for test.
    :param model: model to train
    :param X: features
    :param y: labels
    :return: void
    '''
    one_hot_encoder = make_column_transformer(
        (OneHotEncoder(sparse=False, handle_unknown='ignore'),
         make_column_selector(dtype_include='category')),
        remainder='passthrough')
    pipeline = make_pipeline(one_hot_encoder, model)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model_predict = pipeline.fit(X_train, y_train)
    predictions = model_predict.predict(X_test)

    # The coefficients
    # print('Coefficients: \n', model_predict[1].coef_)
    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(y_test, predictions))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
          % r2_score(y_test, predictions))
    sns.regplot(y_test, predictions)
    st.pyplot()


def prepare_df(df):
    df_selected = df[[rev, country, style, brand, stars]]

    df_cat = fix_cat(df_selected, 'Brand')
    df_cat = fix_cat(df_cat, 'Style')
    df_cat = fix_cat(df_cat, 'Country')

    df_coded = df_cat[[rev, stars, brand, country]]

    df_coded_sample = df_coded.sample(frac=1)
    return df_coded_sample


if __name__ == '__main__':

    # Read df and print summary
    df = pd.read_csv(file_ramen, dtype={'Stars': float})
    st.dataframe(df)

    # Get information of feature, distribution of values
    print_column_info(df)
    option = get_plot_for_column()
    plot_df_column(df, option)

    # Map strings to categories Sample the df mixing all the rows
    df_coded_sample = prepare_df(df)
    # ATTENTION YOU SHOULD KEEP A TEST SUBSET FOR FINAL EVALUATION THIS IS JUST FOR DEMO PURPOSES
    # Split features (Brand, Country, since Style is not very relevant) and labels (Stars)
    y = df_coded_sample[[stars]]
    X = df_coded_sample[['Brand', 'Country']]

    # Models to run cv and show in the dropdown list
    models = [RandomForestRegressor(max_depth=12, random_state=0, max_features=6), Ridge(),
              tree.DecisionTreeRegressor()]

    results = []

    # Run CV for each model and print results
    for m in models:
        res_model = run_cv_model(m, X, y)
        st.text(str(m) + " Cross validation results")
        st.dataframe(res_model)

    model_to_train = get_model_train()

    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Run full training for selected model
    full_training(model_to_train, X, y)
