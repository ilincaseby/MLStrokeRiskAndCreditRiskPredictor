import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

avc_cont_num_attrs = ['mean_blood_sugar_level', 'body_mass_indicator',\
                      'analysis_results', 'biological_age_index']
salary_prediction_num_attrs = ['fnl', 'gain', 'loss', 'prod']
avc_discret = ['cardiovascular_issues', 'job_category', 'sex', 'tobacco_usage',\
               'high_blood_pressure', 'married', 'living_area', 'years_old',\
                'chaotic_sleep', 'cerebrovascular_accident']
salary_prediction_discret = ['hpw', 'relation', 'country', 'job', 'edu_int', 'years',\
                             'work_type', 'partner', 'edu', 'gender', 'race', 'gtype']
salary_prediction_categ = ['relation', 'country', 'job', 'work_type', 'partner', 'edu', 'gender', 'race', 'gtype']
avc_categ = ['cardiovascular_issues', 'job_category', 'sex', 'tobacco_usage',\
                           'high_blood_pressure', 'married', 'living_area', 'cerebrovascular_accident']


def discret_extraction(data_frame: pd.DataFrame, discret_attrs: list, dataset: str):
    data_extracted = {}
    for attr in discret_attrs:
        attr_series = data_frame[attr].dropna()
        attr_filled = attr_series.count()
        attr_unique_count = attr_series.nunique()
        attr_dict = {'No empty values': attr_filled, 'Number of unique values': attr_unique_count}
        data_extracted[attr] = attr_dict
    data_frame_on_extracted = pd.DataFrame(data_extracted)
    data_frame_on_extracted.to_csv(dataset + '_on_discret_or_cat_values.csv', index=False)
    data_frame_on_extracted.hist(width=0.5)
    plt.savefig(dataset + '_histogram.png')
    plt.close()

def numeric_extraction(data_frame: pd.DataFrame, cont_num_attrs: list, dataset: str):
    data_extracted = {}
    for attr in cont_num_attrs:
        attr_series = data_frame[attr].dropna()
        attr_filled = attr_series.count()
        attr_mean = attr_series.mean()
        attr_std = attr_series.std()
        attr_min = attr_series.min()
        attr_quantile_25 = attr_series.quantile(0.25)
        attr_quantile_50 = attr_series.quantile(0.5)
        attr_quantile_75 = attr_series.quantile(0.75)
        attr_max = attr_series.max()
        attr_dict = {'No empty values': attr_filled, 'Mean value': attr_mean, \
                     'Standard deviation': attr_std, 'Min value': attr_min, \
                        'Quantile 25%': attr_quantile_25, 'Quantile 50%': attr_quantile_50, \
                            'Quantile 75%': attr_quantile_75, 'Max value': attr_max}
        data_extracted[attr] = attr_dict
        plt.plot(attr_series)
        plt.savefig(dataset + '_plot_on_' + attr + '.png')
        plt.close()
    data_frame_on_extracted = pd.DataFrame(data_extracted)
    data_frame_on_extracted.to_csv(dataset + '_on_continuous_numeric_values.csv', index=False)

def countplot(data_frame: pd.DataFrame, dataset: str):
    data_extracted_col = []
    data_extracted_count = []
    for col in data_frame.columns:
        attr_series = data_frame[col].dropna()
        attr_filled = attr_series.count()
        data_extracted_col.append(col)
        data_extracted_count.append(attr_filled)
    df = pd.DataFrame({'Attributes': data_extracted_col, 'Frequency': data_extracted_count})
    ax = df.plot.bar(x='Attributes', y='Frequency', rot=0)
    fig = ax.get_figure()
    fig.savefig(dataset + '_bar_plot.png')

def correlation_analysis_cont(data_frame: pd.DataFrame, list_attr: list, dataset: str):
    new_data_frame = data_frame[list_attr]
    corr_matrix = new_data_frame.corr(method='pearson').values
    fig = plt.figure(figsize=(len(list_attr), len(list_attr)))
    ax = fig.add_subplot(111)
    # Normalise data with vmin and vmax
    cax = ax.matshow(corr_matrix, vmin=-1, vmax=1)
    # Add color bar
    fig.colorbar(cax)
    # Define ticks
    ticks = np.arange(0, len(list_attr), 1)
    # Set ticks on x and y
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    # Set names
    ax.set_xticklabels(list_attr)
    ax.set_yticklabels(list_attr)
    fig.savefig(dataset + '_corr_matrix.png')

def correlation_analysis_categ(data_frame: pd.DataFrame, list_attr: list, dataset: str):
    pass


def avc_predict():
    data_frame = pd.read_csv('tema2_AVC/AVC_train.csv')
    # numeric_extraction(data_frame, avc_cont_num_attrs, 'AVC')
    # discret_extraction(data_frame, avc_discret, 'AVC')
    # countplot(data_frame, 'AVC')
    correlation_analysis_cont(data_frame, avc_cont_num_attrs, 'AVC')

def salary_predict():
    data_frame = pd.read_csv('tema2_SalaryPrediction/SalaryPrediction_train.csv')
    # numeric_extraction(data_frame, salary_prediction_num_attrs, 'SalaryPrediction')
    # discret_extraction(data_frame, salary_prediction_discret, 'SalaryPrediction')
    # countplot(data_frame, 'SalaryPrediction')
    correlation_analysis_cont(data_frame, salary_prediction_num_attrs, 'SalaryPrediction')

def main():
    avc_predict()
    salary_predict()


if __name__ == "__main__":
    main()