from sklearn.metrics import log_loss
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import math

D_Category = {
    0: 'infectious and parasitic diseases',
    1: 'neoplasms',
    2: 'endocrine, nutritional and metabolic diseases, and immunity disorders',
    3: 'diseases of the blood and blood-forming organs',
    4: 'mental disorders',
    5: 'diseases of the nervous system and sense organs',
    6: 'diseases of the circulatory system',
    7: 'diseases of the respiratory system',
    8: 'diseases of the digestive system',
    9: 'diseases of the genitourinary system',
    10: 'complications of pregnancy, childbirth, and the puerperium',
    11: 'diseases of the skin and subcutaneous tissue',
    12: 'diseases of the musculoskeletal system and connective tissue',
    13: 'congenital anomalies',
    14: 'certain conditions originating in the perinatal period',
    15: 'symptoms, signs, and ill-defined conditions',
    16: 'injury and poisoning',
    17: 'external causes of injury and supplemental classification'
}

def calculate_age(df, cur_time, dob):
    df[cur_time] = pd.to_datetime(df[cur_time])
    df[dob] = pd.to_datetime(df[dob])
    result = df[cur_time].dt.year - df[dob].dt.year
    result.loc[result>=300] = 90
    return result

def assign_category(icd9_code):
    try:
        code = int(icd9_code)
        if code >= 10000:
            code = code // 100
        elif code >= 1000:
            code = code // 10
        else:
            pass
        if code <= 139:
            return 0
        if code <= 239:
            return 1
        if code <= 279:
            return 2
        if code <= 289:
            return 3
        if code <= 319:
            return 4
        if code <= 389:
            return 5
        if code <= 459:
            return 6
        if code <= 519:
            return 7
        if code <= 579:
            return 8
        if code <= 629:
            return 9
        if code <= 679:
            return 10
        if code <= 709:
            return 11
        if code <= 739:
            return 12
        if code <= 759:
            return 13
        if code <= 779:
            return 14
        if code <= 799:
            return 15
        return 16
    except:
        return 17

def fill_missing_mean(df, col_name, is_countinuous):
    num_missing = df.count()['HADM_ID'] - df.count()[col_name]
    if num_missing == 0:
        return df
    try: 
        if is_countinuous:
            avg = df[col_name].mean()
            print(avg)
            df[col_name] = df[col_name].fillna(avg)

        else:
            print(col_name)
            values = df[df[col_name].notna()][col_name].unique()
            rates = df[col_name].value_counts(normalize=True)
            for value in values:
                df[col_name] = df[col_name].fillna(value, limit=math.ceil(rates[value]*num_missing))
    except:     
        print('unable to process item ', col_name)
    return df

def fill_missing_quantile(df, col_name, is_countinuous):
    num_missing = df.count()['HADM_ID'] - df.count()[col_name]
    if num_missing == 0:
        return df
    try: 
        if is_countinuous:
            vmin = df[col_name].min()
            q25 = df[col_name].quantile(0.25)
            df[col_name] = df[col_name].fillna(random.uniform(vmin, q25), limit=math.ceil(0.25*num_missing))
            med = df[col_name].quantile(0.5)
            df[col_name] = df[col_name].fillna(random.uniform(q25, med), limit=math.ceil(0.25*num_missing))
            q75 = df[col_name].quantile(0.75)
            df[col_name] = df[col_name].fillna(random.uniform(med, q75), limit=math.ceil(0.25*num_missing))
            vmax = df[col_name].max()
            df[col_name] = df[col_name].fillna(random.uniform(q75, vmax), limit=math.ceil(0.25*num_missing))

        else:
            values = df[df[col_name].notna()][col_name].unique()
            rates = df[col_name].value_counts(normalize=True)
            for value in values:
                df[col_name] = df[col_name].fillna(value, limit=math.ceil(rates[value]*num_missing))
    except:     
        print('unable to process item ', col_name)
    return df

def convert(x):
    try:
        return float(x)
    except:
        return None

def clean_numeric(col):
    col = col.apply(lambda x: convert(x))
    return col

def get_corr(df, target):
    corr = df['valuenum'].corr(df[target], method='pearson')
    print("Pearson Correlation of item and "+target+": ", corr)
    return corr

def item_eda(df, plot_type):
    f, a = plt.subplots(figsize=(10, 7))
    if plot_type=='box':
        a = sns.boxplot(x='valuenum', y='LOS', data=df, hue='HOSPITAL_EXPIRE_FLAG')
    else:
        a = sns.scatterplot(x='valuenum', y='LOS', data=df, hue='HOSPITAL_EXPIRE_FLAG')

def explain_logistic_regression(lr, feature_names):
#     plt.rcParams['figure.figsize'] = [16, 9]
    plt.figure(figsize=(50, 20))
    x = lr.coef_[0]
    plt.xticks(rotation=90)
    plt.xlabel('feature_names')
    plt.ylabel('coefficient')
    plt.grid()
    plt.bar(feature_names, x)
    plt.savefig('lr.png')
    plt.show()

def permutation_importance(model, feature_names, X, y):
    # calculate importance score for each feature by permutation approach
    y_pred = model.predict_proba(X)
    base_error = log_loss(y, y_pred)
    importance = []
    for i in range(len(feature_names)):
        x = X.copy()
        x[:, i] = np.random.permutation(x[:, i])
        y_pred = model.predict_proba(x)
        perm_error = log_loss(y, y_pred)
        importance.append(abs(base_error - perm_error))

    temp = importance.copy()
    temp.sort()
    top_10 = temp[-10:]
    result_importance = []
    result_feature = []
    for e in top_10:
        idx = np.where(importance==e)[0][0]
        result_importance.append(e)
        result_feature.append(feature_names[idx])
    plt.figure(figsize=(16, 9))
    plt.bar(result_feature, result_importance)
    plt.xlabel('feature_names')
    plt.ylabel('feature importance')
    plt.xticks(rotation=90)
    plt.show()
