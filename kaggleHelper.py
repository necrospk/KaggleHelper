# recomended use
# import kaggleHelper as kh

import numpy
import pandas
import sklearn
import os
import gc
from sklearn.metrics import *

def submit_result(df,id,target,path,name,score = 0,oof = None):
    """
    Function to from submit at kaggle.
    Create two folder in path folder, oof - with predict at train and submition - with predict at test.
    
    df - pandas DataFrame with id and target
    id - id field in df
    target - target feature in df
    path - path to save submit on local machine
    name - name for file with submit
    score - score at CV if exist
    oof - pandas DataFrame with predict at train if exist
    
    """
    arr = []
    sub_path = path + 'submition/'
    oof_path = path + 'oof/'
    
    if not os.path.exists(sub_path):
        os.makedirs(sub_path)
        print('Path:',sub_path,'does not exist. Creating path folder.')
    
    if not os.path.exists(oof_path):
        os.makedirs(oof_path)
        print('Path:',oof_path,'does not exist. Creating path folder.')
    
    #numeration for submits
    for a in os.listdir(sub_path):
        arr.append(a)
    arr = [re.sub("[^0-9_]",'',a) for a in arr ]
    arr = [a.split('_') for a in arr if a not in ['']]
    arr = [int(a[0]) for a in arr if a[0] not in ['']]
    try:
        pre = str(max(arr)+1)
    except:
        pre = '1'
    
    #submit time
    str_dat = str(time.localtime().tm_year)+'_'+str(time.localtime().tm_mon)+'_'+str(time.localtime().tm_mday)
    
    df.groupby(id)[target].mean().reset_index().to_csv(sub_path+pre+'_'+name+str_dat+'_CV_'+str(round(score,5))+'.csv',index=False)
    if oof != None:
      oof.groupby(id)[target].mean().reset_index().to_csv(oof_path+pre+'_'+name+str_dat+'_CV_'+str(round(score,5))+'.csv',index=False)
    
    
def smothed_aggregate(df, null_field, agg_field, alpha = 10):
    """
    Smothed aggregate, that use to reduce overfiting
    """
    d1 = df[[null_field, agg_field]].fillna(0).groupby([null_field])[agg_field].transform('mean')
    d2 = df[[null_field, agg_field]].fillna(0).groupby(null_field)[null_field].transform('count')
    d3 = df[agg_field].fillna(0).astype(np.float32).mean()
    
    result_series = (d1 * d2 + d3 * alpha) / (d2 + alpha)
    
    return result_series
    
def reduce_mem_usage(df, verbose=True, less_data = True):
    """
    Compresse DataFrame for low mem usage.
    !!!WARNING!!!
    The default parameter less_data = True, that mean,
    while you use this function, tou understand that while you compress 
    float value you may lose precision in decimal places.
    If you don't want it - set parameter less_data to False
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if less_data:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)    
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: 
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def ensemble_predictions(predictions, weights=None, type_="linear"):
    if weights != None:
        assert np.isclose(np.sum(weights), 1.0)
    if type_ == "linear":
        res = np.average(predictions, weights=weights, axis=0)
    elif type_ == "harmonic":
        res = np.average([1 / p for p in predictions], weights=weights, axis=0)
        return 1 / res
    elif type_ == "geometric":
        numerator = np.average(
            [np.log(p) for p in predictions], weights=weights, axis=0
        )
        res = np.exp(numerator / sum(weights))
        return res
    elif type_ == "rank":
        res = np.average([rankdata(p) for p in predictions], weights=weights, axis=0)
        return res / (len(res) + 1)
    return res
