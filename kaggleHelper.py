# recomended use
# import kaggleHelper as kh

import numpy
import pandas
import sklearn
import os
import gc
from sklearn.metrics import *
from catboost import CatBoostRegressor, Pool, CatBoostClassifier
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, GroupKFold
import lightgbm as lgb


def print_bold(string):
    """
    Function to display in output JupyterNotebook markdowns
    """
    display(Markdown(string))

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
    df - pd.DataFrame() with null_field and agg_field
    null_field - field, that will be used in groupby
    agg_fueld - field, that need be aggregate
    alpha - coefficent for smooth
    
    return  - result pd.Series() with aggregated values
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
    """
    Function to ansamble prediction.
    
    predictions - array with predictions
    weights - weight for prediction in array
    type_ - tpe of mix
        'linear' - simple mean stuck
        'harmonic' - ?
        'geometric' - ?
        'rank' - ranked stuck (vote method)
    """
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


def lgbm_calc(train,
              test,
              features,
              target,
              param,
              score_function = roc_auc_score,
              n_fold = 3, 
              seed = 11, 
              cat_features = []
              ):
    """
    Function for predicting with lgbm, that use KFold method
    
    train - train dataset
    test - test dataset
    features - features, that will be used for predict
    target - target feature
    param - param for LGBM (see doc. for lgbm)
    score_function  - score function from sklearn.metrics or you own function
    n_fold - number folds for KFold
    seed - seed for random
    cat_features - categorical feature if tou have it in dataset (default [])
    
    return :
    
    oof_df - dataframe with predict for train part
    submit - dataframe with predict for test part
    fi - feature importance of training
    
    """
    folds = KFold(n_splits=n_fold, shuffle=False, random_state = seed)
    #folds = GroupKFold(n_splits=n_fold)

    
    fi = pd.DataFrame(np.zeros(len(features)))
    fi.columns = ['importance']
    fi['feature'] = features

    submit = test[[test.columns[0]]]
    submit[target] = 0
    oof_glob = train[[test.columns[0],target]]

    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))

    print('Cnt features:',len(features))
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train,train[target])):#
        print_bold('***')
        print_bold('Fold №'+str(fold_+1))
                
        trn_data = lgb.Dataset(train.iloc[trn_idx][features],
                               label=train[target].iloc[trn_idx],
                               categorical_feature=cat_features
                              )
        val_data = lgb.Dataset(train.iloc[val_idx][features],
                               label=train[target].iloc[val_idx],
                               categorical_feature=cat_features
                              )
        num_round = 5000
        clf = lgb.train(
                        param,
                        trn_data,
                        num_round,
                        valid_sets = [trn_data, val_data],
                        verbose_eval=100,
                        early_stopping_rounds = 300
                       )
        fi['importance'] = fi['importance']+clf.feature_importance()/ folds.n_splits

        oof_pred = clf.predict(train.iloc[val_idx][features])
        oof[val_idx] = oof_pred#(oof_pred - oof_pred.min())/(oof_pred.max() - oof_pred.min())
        pred = clf.predict(test[features])
        predictions += pred/ folds.n_splits
        
        score = score_function(train[target][val_idx],oof[val_idx])
        print_bold('<b>Result flod'+str(fold_+1)+': AUC = '+str(score)+'</b>')

    submit[target] = predictions

    score = score_function(train[target],oof)
    oof_df = train[[test.columns[0]]]
    oof_df[target] = oof

    print_bold('<b>Result: AUC = '+str(score)+'</b>')
    #submit_result(submit,'LGBM_',score,oof_df)
    plt.figure(figsize=(14,25))
    sns.barplot(x="importance",
            y="feature",
            data=fi.sort_values(by="importance",
                                           ascending=False)[:100])
    return oof_df,submit,fi

def catboost_calc(train,
              test,
              features,
              target,
              param,
              score_function = roc_auc_score,
              n_fold = 3, 
              seed = 11, 
              cat_features = []
              ):
    """
    Function for predicting with CatBoost(Yandex), that use KFold method
    
    train - train dataset
    test - test dataset
    features - features, that will be used for predict
    target - target feature
    param - param for LGBM (see doc. for lgbm)
    score_function  - score function from sklearn.metrics or you own function
    n_fold - number folds for KFold
    seed - seed for random
    cat_features - categorical feature if tou have it in dataset (default [])
    
    return :
    
    oof_df - dataframe with predict for train part
    submit - dataframe with predict for test part    
    """
    
    folds = KFold(n_splits=5, shuffle=False, random_state = 22)
    #folds = GroupKFold(n_splits=n_fold)


    submit = test[[test.columns[0]]]
    submit[target] = 0
    oof_glob = train[[test.columns[0],target]]
    oof_glob['mean_all'] = 0
    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))

    print('Cnt features:',len(features))
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train,train[target])):#,train.DT_YM_cos
        print_bold('***')
        print_bold('Fold №'+str(fold_+1))

        trn_data = Pool(train.iloc[trn_idx][features],
                               label=train[target].iloc[trn_idx],
                                cat_features = cat_features
                              )
        val_data = Pool(train.iloc[val_idx][features],
                               label=train[target].iloc[val_idx],
                                cat_features = cat_features
                              )
        
        cb = CatBoost(param)
        clf = cb.fit(trn_data,eval_set = val_data)
        
        oof_pred = clf.predict_proba(train[features])
        oof_glob['mean_all'] += oof_pred[:,1]/ folds.n_splits
        oof[val_idx] = oof_pred[val_idx][:,1]
        
        pred = clf.predict_proba(test[features])
        predictions += pred[:,1]/ folds.n_splits

        score = score_function(train[target][val_idx],oof[val_idx])
        print_bold('<b>Result flod'+str(fold_+1)+': AUC = '+str(score)+'</b>')

    submit[target] = predictions

    score = score_function(train[target],oof_glob['mean_all'])
    oof_df = train[[test.columns[0]]]
    oof_df[target] = oof

    print_bold('<b>Result: AUC = '+str(score)+'</b>')
    submit_result(submit,'CatBoost_',score,oof_df)
    
    return oof_df,submit

