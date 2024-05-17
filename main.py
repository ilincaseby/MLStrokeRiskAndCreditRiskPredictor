import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier

avc_cont_num_attrs = ['mean_blood_sugar_level', 'body_mass_indicator',\
                      'analysis_results', 'biological_age_index']
salary_prediction_num_attrs = ['fnl', 'gain', 'loss', 'prod']
avc_discret = ['cardiovascular_issues', 'job_category', 'sex', 'tobacco_usage',\
               'high_blood_pressure', 'married', 'living_area', 'years_old',\
                'chaotic_sleep', 'cerebrovascular_accident']
salary_prediction_discret = ['hpw', 'relation', 'country', 'job', 'edu_int', 'years',\
                             'work_type', 'partner', 'edu', 'gender', 'race', 'gtype']
salary_prediction_categ = ['relation', 'country', 'job', 'work_type', 'partner', 'edu', 'gender',\
                           'race', 'gtype']
avc_categ = ['cardiovascular_issues', 'job_category', 'sex', 'tobacco_usage',\
                           'high_blood_pressure', 'married', 'living_area',\
                            'cerebrovascular_accident']
avc_continuous_work = ['body_mass_indicator', 'analysis_results', 'biological_age_index']
avc_discret_and_categoric_work = ['cardiovascular_issues', 'sex', 'married', 'living_area',\
                                  'years_old', 'chaotic_sleep', 'cerebrovascular_accident']
salary_prediction_continuous_work = ['fnl', 'gain', 'loss']
salary_prediction_discret_and_categoric_work = ['hpw', 'relation', 'country', 'job', 'edu_int',\
                                                 'years', 'work_type', 'edu']


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
    fig = plt.figure(figsize=(2*len(list_attr), 2*len(list_attr)))
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
    ax.set_xticklabels(list_attr, rotation=45)
    ax.set_yticklabels(list_attr, rotation=45)
    fig.savefig(dataset + '_corr_matrix.png')

def correlation_analysis_categ(data_frame: pd.DataFrame, list_attr: list, dataset: str):
    correlation_matrix = pd.DataFrame(index=list_attr, columns=list_attr)
    
    for attr1 in list_attr:
        for attr2 in list_attr:
            cross_tab = pd.crosstab(index=data_frame[attr1], columns=data_frame[attr2])
            chi_res = chi2_contingency(cross_tab)
            if chi_res[1] < 0.05:
                correlation_matrix.loc[attr1, attr2] = 1.0
            else:
                correlation_matrix.loc[attr1, attr2] = 0.0
    correlation_matrix = correlation_matrix.astype(float)
    fig = plt.figure(figsize=(2*len(list_attr), 2*len(list_attr)))
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlation_matrix, cmap='coolwarm')
    fig.colorbar(cax)
    ticks = np.arange(0, len(list_attr), 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(list_attr, rotation=45)
    ax.set_yticklabels(list_attr, rotation=45)
    fig.savefig(dataset + '_corr_matrix_on_categ.png')

def imputer_on_cont_discret_categ(data_frame: pd.DataFrame, list_attr_cont: list,\
                                  list_attr_disc_and_categ: list):
    cont_imp = SimpleImputer(strategy='mean')
    disc_and_categ_imp = SimpleImputer(strategy='most_frequent')
    data_frame[list_attr_cont] = cont_imp.fit_transform(data_frame[list_attr_cont])
    data_frame[list_attr_disc_and_categ] = disc_and_categ_imp.fit_transform(data_frame[list_attr_disc_and_categ])
    data_frame.to_csv('test.csv', index=False)
    return data_frame

def no_extreme_values(data_frame: pd.DataFrame, list_attr_cont: list, list_attr_disc: list):
    for attr in list_attr_cont:
        attr_series = data_frame[attr]
        attr_quantile_90 = attr_series.quantile(0.90)
        attr_quantile_10 = attr_series.quantile(0.10)
        outliers = (data_frame[attr] < attr_quantile_10) | (data_frame[attr] > attr_quantile_90)
        data_frame.loc[outliers, attr] = np.nan
    # for attr in list_attr_disc:
    #     attr_series = data_frame[attr]
    #     attr_quantile_90 = attr_series.quantile(0.90)
    #     attr_quantile_10 = attr_series.quantile(0.10)
    #     outliers = (data_frame[attr] < attr_quantile_10) | (data_frame[attr] > attr_quantile_90)
    #     data_frame.loc[outliers, attr] = np.nan
    return data_frame

def standard_scaler(data_frame: pd.DataFrame, list_attr: list):
    st_scaler = StandardScaler()
    to_scale_data_frame = data_frame[list_attr]
    to_scale_data_frame = st_scaler.fit_transform(to_scale_data_frame)
    data_frame[list_attr] = to_scale_data_frame
    return data_frame

def one_hot_encoder(data_frame: pd.DataFrame, list_attr: list):
    o_h_encoder = OneHotEncoder()
    aux = pd.get_dummies(data_frame[list_attr])
    data_frame = data_frame.drop(columns=list_attr)
    data_frame = pd.concat([data_frame, aux], axis=1)
    return data_frame

def label_encoder(data_frame: pd.DataFrame, list_only_target: list):
    l_encoder = LabelEncoder()
    aux = l_encoder.fit_transform(data_frame[list_only_target[0]])
    data_frame[list_only_target[0]] = aux
    return data_frame

def sk_logistic_regression(data_frame: pd.DataFrame, data_frame_test: pd.DataFrame,\
                           target: str, prediction_category: str):
    x_train = data_frame.drop(target, axis=1)
    y_train = data_frame[target]
    x_test = data_frame_test.drop(target, axis=1)
    y_test = data_frame_test[target]
    x_train, x_test = x_train.align(x_test, join='inner', axis=1)
    logistic_model = LogisticRegression(penalty='l2', C=0.5)
    logistic_model.fit(x_train, y_train)
    predicted = logistic_model.predict(x_test)
    target_test = y_test.ravel()
    prec = precision_score(target_test, predicted)
    recall = recall_score(target_test, predicted)
    acc = accuracy_score(target_test, predicted)
    f1 = f1_score(target_test, predicted)
    print(f'For {prediction_category} using Logistic Regression, values for metrics are: ')
    print(f'Accuracy = {acc}, Precision = {prec}, Recall = {recall}, F1 = {f1}')

def mlp_predictions(data_frame: pd.DataFrame, data_frame_test: pd.DataFrame,\
                    target: str, prediction_category: str):
    x_train = data_frame.drop(target, axis=1)
    y_train = data_frame[target]
    x_test = data_frame_test.drop(target, axis=1)
    y_test = data_frame_test[target]
    x_train, x_test = x_train.align(x_test, join='inner', axis=1)
    mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', 
                               learning_rate='constant', learning_rate_init=0.001, 
                               max_iter=500, batch_size=32, early_stopping=True, 
                               alpha=0.0001)
    mlp_model.fit(x_train, y_train)
    predicted = mlp_model.predict(x_test)
    target_test = y_test.ravel()
    prec = precision_score(target_test, predicted)
    recall = recall_score(target_test, predicted)
    acc = accuracy_score(target_test, predicted)
    f1 = f1_score(target_test, predicted)
    print(f'For {prediction_category} using MLP(Multi Layered Perceptron), values for metrics are: ')
    print(f'Accuracy = {acc}, Precision = {prec}, Recall = {recall}, F1 = {f1}')

def logistic(x):
    e_x = np.exp(-x)
    numitor = 1 + e_x
    return 1 / numitor

def nll(Y, T):
    N = T.shape[0]
    first_op = T * np.log(Y)
    second_op = (1 - T) * np.log(1 - Y)
    avg = np.mean(first_op + second_op)
    return -avg

def accuracy(Y, T):
    N = Y.shape[0]
    sum = 0
    no = 0
    for i in range(len(Y)):
        if T[i] == 1 and Y[i] >= 0.5:
            sum += 1
        if T[i] == 0 and Y[i] < 0.5:
            sum += 1
        no += 1
    return sum /no

def predict_logistic(X, w):
    a = np.dot(X, w)
    return logistic(a)

def train_and_eval_logistic(X_train, T_train, N, D, lr=.01, epochs_no=300):
    # Initializare ponderi
    w = np.random.randn(D)

    for _ in range(epochs_no):
        Y_train = predict_logistic(X_train, w)
        
        w -= lr * np.dot(X_train.T, Y_train - T_train) / N

    return w

def logistic_prediction(data_frame: pd.DataFrame, data_frame_test: pd.DataFrame, target: str, topic: str):
    x_train = data_frame.drop(target, axis=1)
    y_train = data_frame[target]
    x_test = data_frame_test.drop(target, axis=1)
    y_test = data_frame_test[target]
    x_train, x_test = x_train.align(x_test, join='inner', axis=1)
    # X_train_np = x_train.values
    X_train_np = np.array([[int(val) if isinstance(val, bool) else val for val in row] for row in x_train.values])
    X_train_np = np.concatenate([X_train_np, np.ones((X_train_np.shape[0], 1))], axis=1)
    Y_train_np = y_train.values
    X_test_np = np.array([[int(val) if isinstance(val, bool) else val for val in row] for row in x_test.values])
    X_test_np = np.concatenate([X_test_np, np.ones((X_test_np.shape[0], 1))], axis=1)
    w = train_and_eval_logistic(X_train_np, Y_train_np, X_train_np.shape[0], X_train_np.shape[1])
    Y = predict_logistic(X_test_np, w)
    for i in range(len(Y)):
        if Y[i] >= 0.5:
            Y[i] = 1
        if Y[i] < 0.5:
            Y[i] = 0
    Y_test_np = y_test.values
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    for i in range(len(Y)):
        if Y[i] == Y_test_np[i]:
            if Y[i] == 0:
                true_neg += 1
            if Y[i] == 1:
                true_pos += 1
        if Y[i] != Y_test_np[i]:
            if Y[i] == 0:
                false_neg += 1
            if Y[i] == 1:
                false_pos += 1
    print(f'Using own implementation of Logistic Regression for {topic}, results have the following metrics')
    denom_accuracy = true_pos + true_neg + false_neg + false_pos
    accuracy_rl = (true_pos + true_neg) / denom_accuracy if denom_accuracy != 0 else 0
    denom_precision = true_pos + false_pos
    precision_rl = true_pos / denom_precision if denom_precision != 0 else 0
    denom_recall = true_pos + false_neg
    recall_rl = true_pos / denom_recall if denom_recall != 0 else 0
    denom_f1 = precision_rl + recall_rl
    f1_score_rl = 2 * precision_rl * recall_rl / denom_f1 if denom_f1 != 0 else 0
    print(f'Accuracy: {accuracy_rl}, Precision: {precision_rl}, Recall: {recall_rl}, F1_score: {f1_score_rl}')


def avc_predict():
    data_frame = pd.read_csv('tema2_AVC/AVC_train.csv')
    data_frame_test = pd.read_csv('tema2_AVC/AVC_test.csv')
    # numeric_extraction(data_frame, avc_cont_num_attrs, 'AVC')
    # discret_extraction(data_frame, avc_discret, 'AVC')
    # countplot(data_frame, 'AVC')
    # correlation_analysis_cont(data_frame, avc_cont_num_attrs, 'AVC')
    # correlation_analysis_categ(data_frame, avc_categ, 'AVC')
    data_frame = no_extreme_values(data_frame, avc_cont_num_attrs, [elem for elem in avc_discret if\
                                                       elem not in avc_categ])
    data_frame = imputer_on_cont_discret_categ(data_frame, avc_cont_num_attrs, avc_discret)
    data_frame = standard_scaler(data_frame, avc_cont_num_attrs)
    data_frame_test = no_extreme_values(data_frame_test, avc_cont_num_attrs, [elem for elem in avc_discret if\
                                                       elem not in avc_categ])
    data_frame_test = imputer_on_cont_discret_categ(data_frame_test, avc_cont_num_attrs, avc_discret)
    data_frame = standard_scaler(data_frame, avc_cont_num_attrs)
    data_frame_test = standard_scaler(data_frame_test, avc_cont_num_attrs)
    data_frame = data_frame[avc_continuous_work + avc_discret_and_categoric_work]
    data_frame_test = data_frame_test[avc_continuous_work + avc_discret_and_categoric_work]
    data_frame = one_hot_encoder(data_frame, [elem for elem in avc_categ if elem in\
                                 avc_discret_and_categoric_work and elem\
                                    != 'cerebrovascular_accident'])
    data_frame = label_encoder(data_frame, ['cerebrovascular_accident'])
    data_frame_test = one_hot_encoder(data_frame_test, [elem for elem in avc_categ if elem in\
                                 avc_discret_and_categoric_work and elem\
                                    != 'cerebrovascular_accident'])
    data_frame_test = label_encoder(data_frame_test, ['cerebrovascular_accident'])
    sk_logistic_regression(data_frame, data_frame_test, 'cerebrovascular_accident', 'AVC')
    mlp_predictions(data_frame, data_frame_test, 'cerebrovascular_accident', 'AVC')
    logistic_prediction(data_frame, data_frame_test, 'cerebrovascular_accident', 'AVC')

def salary_predict():
    data_frame = pd.read_csv('tema2_SalaryPrediction/SalaryPrediction_train.csv')
    data_frame_test = pd.read_csv('tema2_SalaryPrediction/SalaryPrediction_test.csv')
    # numeric_extraction(data_frame, salary_prediction_num_attrs, 'SalaryPrediction')
    # discret_extraction(data_frame, salary_prediction_discret, 'SalaryPrediction')
    # countplot(data_frame, 'SalaryPrediction')
    # correlation_analysis_cont(data_frame, salary_prediction_num_attrs, 'SalaryPrediction')
    # correlation_analysis_categ(data_frame, salary_prediction_categ, 'SalaryPrediction')
    data_frame = no_extreme_values(data_frame, salary_prediction_num_attrs, [elem for elem in\
                                                                salary_prediction_discret if elem\
                                                                not in salary_prediction_categ])
    data_frame = imputer_on_cont_discret_categ(data_frame, salary_prediction_num_attrs,\
                                  salary_prediction_discret)
    data_frame = standard_scaler(data_frame, salary_prediction_num_attrs)
    data_frame = data_frame[salary_prediction_continuous_work +\
                            salary_prediction_discret_and_categoric_work + ['money']]
    data_frame = one_hot_encoder(data_frame, [elem for elem in salary_prediction_categ if elem in\
                                 salary_prediction_discret_and_categoric_work])
    data_frame = label_encoder(data_frame, ['money'])
    data_frame_test = no_extreme_values(data_frame_test, salary_prediction_num_attrs, [elem for elem in\
                                                                salary_prediction_discret if elem\
                                                                not in salary_prediction_categ])
    data_frame_test = imputer_on_cont_discret_categ(data_frame_test, salary_prediction_num_attrs,\
                                  salary_prediction_discret)
    data_frame_test = standard_scaler(data_frame_test, salary_prediction_num_attrs)
    data_frame_test = data_frame_test[salary_prediction_continuous_work +\
                            salary_prediction_discret_and_categoric_work + ['money']]
    data_frame_test = one_hot_encoder(data_frame_test, [elem for elem in salary_prediction_categ if elem in\
                                 salary_prediction_discret_and_categoric_work])
    data_frame_test = label_encoder(data_frame_test, ['money'])
    sk_logistic_regression(data_frame, data_frame_test, 'money', 'Salary Prediction')
    mlp_predictions(data_frame, data_frame_test, 'money', 'Salary Prediction')
    logistic_prediction(data_frame, data_frame_test, 'money', 'Salary')

def main():
    avc_predict()
    salary_predict()


if __name__ == "__main__":
    main()