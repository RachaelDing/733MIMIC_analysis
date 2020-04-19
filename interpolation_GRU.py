import pandas as pd
import numpy as np
import copy
import sys
from datetime import datetime
import argparse
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score as auprc
from sklearn.metrics import roc_auc_score as auc_score
import keras
from keras.utils import multi_gpu_model
from keras.layers import Input, Dense, GRU, Lambda, Permute, LSTM
from keras.models import Model
from interpolation_layer import single_channel_interp, cross_channel_interp
import warnings
from sklearn.preprocessing import KBinsDiscretizer
warnings.filterwarnings("ignore")
import random
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import  MaxAbsScaler, MinMaxScaler
from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--target", choices = ["m","3","7"], required=True, type=str, 
    help="Target : m for mortality, 3 for length of stay separated by 3 days,  7 for length of stay separated by 7 days.")
args = parser.parse_args()
target = args.target


admission = pd.read_csv('./raw_data/ADMISSIONS.csv.gz')
patient = pd.read_csv('./raw_data/PATIENTS.csv.gz')
admission = admission.merge(patient, on='SUBJECT_ID')
admission["age"] = calculate_age(admission, "ADMITTIME","DOB")
admission['DISCHTIME']=  pd.to_datetime(admission['DISCHTIME'])
admission['ADMITTIME'] =  pd.to_datetime(admission['ADMITTIME'])
admission['LOS'] = (admission['DISCHTIME'] - admission['ADMITTIME']).dt.total_seconds()/86400
admission= admission[["HADM_ID", "ADMISSION_TYPE", "LOS","HOSPITAL_EXPIRE_FLAG"]]
admission = admission[admission['LOS'] > 1]

if target == "3" or target == "7":
    cat1 = admission[admission['LOS'] < 3]
    cat2 = admission[admission['LOS'].between(3, 7)]
    cat3 = admission[admission['LOS'] > 7]
    print(cat1.count())
    print(cat2.count())
    print(cat3.count()) 
    cat1_ids = cat1['HADM_ID'].tolist()
    cat2_ids = cat2['HADM_ID'].tolist()
    cat3_ids = cat3['HADM_ID'].tolist()



# heart rate
item_hr = pd.read_csv('gru_features/HR.csv')
item_hr['CHARTTIME'] = pd.to_datetime(item_hr['CHARTTIME']).dt.tz_localize(None)
item_hr = item_hr[item_hr['VALUE'].notna()]

# dbp
item_dbp = pd.read_csv('gru_features/dbp.csv')
item_dbp['CHARTTIME'] = pd.to_datetime(item_dbp['CHARTTIME']).dt.tz_localize(None)
item_dbp = item_dbp[item_dbp['VALUE'].notna()]

# glucose
item_glucose = pd.read_csv('gru_features/HR.csv')
item_glucose['CHARTTIME'] = pd.to_datetime(item_glucose['CHARTTIME']).dt.tz_localize(None)
item_glucose = item_glucose[item_glucose['VALUE'].notna()]

# ph
item_ph = pd.read_csv('gru_features/ph.csv')
item_ph['CHARTTIME'] = pd.to_datetime(item_ph['CHARTTIME']).dt.tz_localize(None)
item_ph = item_ph[item_ph['VALUE'].notna()]

# rr
item_rr = pd.read_csv('gru_features/RR.csv')
item_rr['CHARTTIME'] = pd.to_datetime(item_rr['CHARTTIME']).dt.tz_localize(None)
item_rr = item_rr[item_rr['VALUE'].notna()]

# sbp
item_sbp = pd.read_csv('gru_features/sbp.csv')
item_sbp['CHARTTIME'] = pd.to_datetime(item_sbp['CHARTTIME']).dt.tz_localize(None)
item_sbp = item_sbp[item_sbp['VALUE'].notna()]

# spo2
item_spo2 = pd.read_csv('gru_features/SpO2.csv')
item_spo2['CHARTTIME'] = pd.to_datetime(item_spo2['CHARTTIME']).dt.tz_localize(None)
item_spo2 = item_spo2[item_spo2['VALUE'].notna()]

# temp
item_temp = pd.read_csv('gru_features/temp.csv')
item_temp['CHARTTIME'] = pd.to_datetime(item_temp['CHARTTIME']).dt.tz_localize(None)
item_temp = item_temp[item_temp['VALUE'].notna()]

# tgcs
item_tgcs = pd.read_csv('gru_features/tgcs.csv')
item_tgcs['CHARTTIME'] = pd.to_datetime(item_tgcs['CHARTTIME']).dt.tz_localize(None)
item_tgcs = item_tgcs[item_tgcs['VALUE'].notna()]

# uo
item_uo = pd.read_csv('gru_features/uo.csv')
item_uo['CHARTTIME'] = pd.to_datetime(item_uo['CHARTTIME']).dt.tz_localize(None)
item_uo = item_uo[item_uo['VALUE'].notna()]

# scale each item
scaler =  MinMaxScaler()
item_dbp[['VALUE']] = scaler.fit_transform(item_dbp[['VALUE']])
item_glucose[['VALUE']] = scaler.fit_transform(item_glucose[['VALUE']])
item_hr[['VALUE']] = scaler.fit_transform(item_hr[['VALUE']])
item_ph[['VALUE']] = scaler.fit_transform(item_ph[['VALUE']])
item_rr[['VALUE']] = scaler.fit_transform(item_rr[['VALUE']])
item_sbp[['VALUE']] = scaler.fit_transform(item_sbp[['VALUE']])
item_spo2[['VALUE']] = scaler.fit_transform(item_spo2[['VALUE']])
item_temp[['VALUE']] = scaler.fit_transform(item_temp[['VALUE']])
item_tgcs[['VALUE']] = scaler.fit_transform(item_tgcs[['VALUE']])
item_uo[['VALUE']] = scaler.fit_transform(item_uo[['VALUE']])


# parameters used
vitals_dict = {}
np.random.seed(10)
num_features = 10
max_length = 2000 
gpu_num = 0
ref_points = 192
hid = 50
hours_looks_ahead =24
num_fold = 5
stds = []

def select_ids(df, ids):
    return df[df['HADM_ID'].isin(ids)]


def time_val_toLst(df, hadmin_id):
    temp = df.loc[df['HADM_ID'] == hadmin_id].sort_values(by=['CHARTTIME'])
    if not temp.empty: 
        times = temp.groupby('HADM_ID')['CHARTTIME'].apply(list).tolist()[0]
        vals = temp.groupby('HADM_ID')['VALUE'].apply(list).tolist()[0]
        return [times, vals]
    else:
        return []


def flatten(df, los=hours_looks_ahead):
    a = np.full((len(df), num_features, max_length), -100) # initlizae all as missing
    timestamps = []
    for i in range(len(df)):
        l = []
    
        # find all the unique observed timestamps
        for j in range(num_features):
            ts_ij = df[i][j]
            if ts_ij != []:
                for ts in df[i][j][0]:
                    if ts not in l:
                        l.append(ts)
        l.sort()
        T = copy.deepcopy(l)
        TS = []
        for t in T:
            if (t - T[0]).total_seconds() / 3600 <= los:
                TS.append(t)
        timestamps.append(TS)
        for j in range(num_features):
            s_ij = df[i][j]
            if s_ij != []:
                ts_ij = s_ij[0] 
                c_max = len(ts_ij)
                c = 0
                for k in range(len(TS)):
                    if c < c_max:
                        ts_ijc = ts_ij[c] # cur ts
                        diff = abs(TS[k] - ts_ijc).seconds / 60 # difference between ts's
                        if TS[k] == ts_ijc or diff < 5:
                            try:
                                a[i,j,k] = s_ij[1][c] 
                            except:
                                a[i,j,k] = -100
                            c += 1
    return a, timestamps


def input_format(x, T):
    real_len = 200
    
    for i in range(len(T)):
        if len(T[i]) > real_len:
            T[i] = T[i][:real_len]

    x = a[:, :, :real_len]
    M = np.zeros_like(x)
    delta = np.zeros_like(x)

    for t in T:
        for i in range(1, len(t)):
            t[i] = (t[i] - t[0]).total_seconds() / 3600 # hours difference
        if len(t) != 0:
            t[0] = 0
    
    # count outliers and negative values as missing values
    # M = 0 indicates missing value
    # M = 1 indicates observed value
    # now since we have mask variable, we don't need -100
    M[x > 500] = 0
    x[x > 500] = 0.0
    M[x < 0] = 0
    x[x < -30] = 0.0
    M[x > 0] = 1

    for i in range(num_features):
        for j in range(x.shape[0]):
            for k in range(len(T[j])):
                delta[j, i, k] = T[j][k]
    return x, M, delta


def missing_mean(M, x):
    counts = np.sum(np.sum(M, axis=2), axis=0)
    counts = [c + np.finfo(float).tiny for c in counts]
    mean_values = np.sum(np.sum(x*M, axis=2), axis=0)/counts
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.sum(M[i, j]) == 0:
                M[i, j, 0] = 1
                x[i, j, 0] = mean_values[j]
    return 


def hold_out(mask, perc=0.2):
    """To implement the autoencoder component of the loss, we introduce a set
    of masking variables mr (and mr1) for each data point. If drop_mask = 0,
    then we removecthe data point as an input to the interpolation network,
    and includecthe predicted value at this time point when assessing
    the autoencoder loss. In practice, we randomly select 20% of the
    observed data points to hold out from
    every input time series."""
    drop_mask = np.ones_like(mask)
    drop_mask *= mask
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            count = np.sum(mask[i, j], dtype='int')
            if int(0.20*count) > 1:
                index = 0
                r = np.ones((count, 1))
                b = np.random.choice(count, int(0.20*count), replace=False)
                r[b] = 0
                for k in range(mask.shape[2]):
                    if mask[i, j, k] > 0:
                        drop_mask[i, j, k] = r[index]
                        index += 1
    return drop_mask


stds = []
stds.append(item_dbp['VALUE'].std())
stds.append(item_glucose['VALUE'].std())
stds.append(item_hr['VALUE'].std())
stds.append(item_ph['VALUE'].std())
stds.append(item_rr['VALUE'].std())
stds.append(item_sbp['VALUE'].std())
stds.append(item_spo2['VALUE'].std())
stds.append(item_temp['VALUE'].std())
stds.append(item_tgcs['VALUE'].std())
stds.append(item_uo['VALUE'].std())

def y_trans(val):
    if val < int(target):
        return 1.0
    else:
        return 0.0

# [17.3, 22, 22.45, 2.32, 23.41，5.7, 3.33, 14.1]
def customloss(ytrue, ypred):
    """ Autoencoder loss
    """
    num_fs = num_features
    # standard deviation of each feature mentioned in paper for MIMIC_III data
    wc = np.array(stds)
    print(wc.shape, ytrue.shape)
    wc.shape = (1, num_fs)
    y = ytrue[:, :num_fs, :]
    m2 = ytrue[:, 3*num_fs:4*num_fs, :]
    m2 = 1 - m2
    m1 = ytrue[:, num_fs:2*num_fs, :]
    m = m1*m2
    ypred = ypred[:, :num_fs, :]
    x = (y - ypred)*(y - ypred)
    x = x*m
    count = tf.reduce_sum(m, axis=2)
    count = tf.where(count > 0, count, tf.ones_like(count))
    x = tf.reduce_sum(x, axis=2)/count
    x = x/(wc**2)  # dividing by standard deviation
    x = tf.reduce_sum(x, axis=1)/num_fs
    return tf.reduce_mean(x)


def interp_net():
    num_fs = num_features
    #if gpu_num > 1:
    #dev = "/cpu:0"
    #else:
    dev = "/gpu:0"
    with tf.device(dev):
        main_input = Input(shape=(4*num_fs, timestamp), name='input')
        sci = single_channel_interp(ref_points, hours_looks_ahead)
        cci = cross_channel_interp()
        interp = cci(sci(main_input))
        reconst = cci(sci(main_input, reconstruction=True),
                      reconstruction=True)
        aux_output = Lambda(lambda x: x, name='aux_output')(reconst)
        z = Permute((2, 1))(interp)
        z = GRU(hid, activation='tanh', recurrent_dropout=0.2, dropout=0.2)(z)
        main_output = Dense(1, activation='sigmoid', name='main_output')(z)
        orig_model = Model([main_input], [main_output, aux_output])
    if gpu_num > 1:
        model = multi_gpu_model(orig_model, gpus=gpu_num)
    else:
        model = orig_model
    print(orig_model.summary())
    return model

seed = 0
results = {}
results['loss'] = []
results['auc'] = []
results['acc'] = []
results['auprc'] = []
earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0000, patience=20, verbose=0)
callbacks_list = [earlystop]

# number of rows in cat1(LOS<3) is less than the other 2 categories
if target == "3":
    cat1_num = len(cat1_ids)
    print("Cat1 num:",cat1_num )
    cat3_num = int(cat1_num/2)
    print("Cat3 num:",cat3_num )
    cat2_num = int(cat1_num/2)
    print("Cat2 num:",cat2_num )
    needed_ids= cat1_ids[0:cat1_num] + cat2_ids[0:cat2_num] + cat3_ids[:cat3_num]
elif target == "7":
    cat1_num = len(cat1_ids)
    print("Cat1 num:",cat1_num )
    cat3_num = int(cat1_num*2)
    print("Cat3 num:",cat3_num )
    cat2_num = int(cat1_num)
    print("Cat2 num:",cat2_num)
    needed_ids= cat1_ids[0:cat1_num] + cat2_ids[0:cat2_num] + cat3_ids[:cat3_num]
else:
    dead_ids = admission[admission["HOSPITAL_EXPIRE_FLAG"] == 1]['HADM_ID'].tolist()
    alive_ids = admission[admission["HOSPITAL_EXPIRE_FLAG"] == 0]['HADM_ID'].tolist()
    cat_num = min(len(dead_ids), len(alive_ids))
    needed_ids = dead_ids[0:cat_num] + alive_ids[0:cat_num]

  
temp_dbp = select_ids(item_dbp, needed_ids)
temp_glucose = select_ids(item_glucose, needed_ids)
temp_hr = select_ids(item_hr, needed_ids)
temp_ph = select_ids(item_ph, needed_ids)
temp_rr = select_ids(item_rr, needed_ids)
temp_sbp = select_ids(item_sbp, needed_ids)
temp_spo2 = select_ids(item_spo2, needed_ids)
temp_temp = select_ids(item_temp, needed_ids)
temp_tgcs = select_ids(item_tgcs, needed_ids)
temp_uo = select_ids(item_uo, needed_ids)

vitals_dict = {}
for adm_id in needed_ids:
    vitals_dict[adm_id] = [time_val_toLst(temp_sbp, adm_id)]
    vitals_dict[adm_id].append(time_val_toLst(temp_glucose, adm_id))
    vitals_dict[adm_id].append(time_val_toLst(temp_hr, adm_id))
    vitals_dict[adm_id].append(time_val_toLst(temp_ph, adm_id))
    vitals_dict[adm_id].append(time_val_toLst(temp_rr, adm_id))
    vitals_dict[adm_id].append(time_val_toLst(temp_sbp, adm_id))
    vitals_dict[adm_id].append(time_val_toLst(temp_spo2, adm_id))
    vitals_dict[adm_id].append(time_val_toLst(temp_temp, adm_id))
    vitals_dict[adm_id].append(time_val_toLst(temp_tgcs, adm_id))
    vitals_dict[adm_id].append(time_val_toLst(temp_uo, adm_id))

vitals = [vitals_dict[x] for x in needed_ids] # hadm_id(los>=48h): all the vitals values
if target == "3" or target == "7":
    label = [admission[admission['HADM_ID'] == adm_id]['LOS'].values[0] for adm_id in needed_ids]
    label = [y_trans(l) for l in label]
else :
    label = [admission[admission['HADM_ID'] == adm_id]['HOSPITAL_EXPIRE_FLAG'].values[0] for adm_id in needed_ids]

a, ts = flatten(vitals, hours_looks_ahead)
x, m, T = input_format(a, ts)
missing_mean(m, x)
X = np.concatenate((x, m, T, hold_out(m)), axis=1)  # input format
y = np.array(label)
timestamp = X.shape[2]
num_features = X.shape[1] // 4
print(X.shape, y.shape)

i = 0
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
for train, test in kfold.split(np.zeros(len(y)), y):
    print("Running Fold:", i+1)
    model1 = interp_net()  # re-initializing every time
    model1.compile(
        optimizer='adam',
        loss={'main_output': 'binary_crossentropy', 'aux_output': customloss},
        loss_weights={'main_output': 1., 'aux_output': 1.},
        metrics={'main_output': 'accuracy'})
    history1 = model1.fit(
        {'input': X[train]}, {'main_output': y[train], 'aux_output': X[train]},
        batch_size=128,
        callbacks=callbacks_list,
        nb_epoch=100,
        validation_split=0.20,
        verbose=2)
    y_pred = model1.predict(X[test], batch_size=batch)
    y_pred = y_pred[0]
    total_loss, score, reconst_loss, acc = model1.evaluate(
        {'input': X[test]},
        {'main_output': y[test], 'aux_output': X[test]},
        batch_size=batch,
        verbose=0)
    results['loss'].append(score)
    results['acc'].append(acc)
    results['auc'].append(auc_score(y[test], y_pred))
    results['auprc'].append(auprc(y[test], y_pred))
    print(results)
    i += 1

