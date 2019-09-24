# KaggelHelper  
  
Self-writing library for more best practice when participating in competitions at Kaggle.com.  
Wrote for myself using.  
  
```python 
print_bold(string)
``` 
Function to display in output JupyterNotebook markdowns  
  
``` python 
submit_result(df,id,target,path,name,score = 0,oof = None)
```
Function to from submit at kaggle.  
Create two folder in path folder, oof - with predict at train and submition - with predict at test.  
  
- df - pandas DataFrame with id and target  
- id - id field in df  
- target - target feature in df  
- path - path to save submit on local machine  
- name - name for file with submit  
- score - score at CV if exist  
- oof - pandas DataFrame with predict at train if exist  
    
    
```python 
smoothed_aggregate(df, null_field, agg_field, alpha = 10)
```
Smoothed aggregate, that use to reduce overfiting  
- df - pd.DataFrame() with null_field and agg_field  
- null_field - field, that will be used in groupby  
- agg_fueld - field, that need be aggregate  
- alpha - coefficent for smooth  
  
- return  - result pd.Series() with aggregated values  
      
```python 
def reduce_mem_usage(df, verbose=True, less_data = True)
```
Compresse DataFrame for low mem usage.  
<b>!!!WARNING!!!  </b>
The default parameter less_data = True, that mean,  
while you use this function, tou understand that while you compress  
float value you may lose precision in decimal places.  
If you don't want it - set parameter less_data to False  

```python 
def ensemble_predictions(predictions, weights=None, type_="linear")
```
Function to ansamble prediction.  
  
- predictions - array with predictions  
- weights - weight for prediction in array  
- type_ - tpe of mix  
   - 'linear' - simple mean stuck  
   - 'harmonic' - ?  
   - 'geometric' - ?  
   - 'rank' - ranked stuck (vote method)  
  
- return result of stuck (array)  
  
```python 
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
```
Function for predicting with lgbm, that use KFold method  
  
- train - train dataset  
- test - test dataset  
- features - features, that will be used for predict  
- target - target feature  
- param - param for LGBM (see doc. for lgbm)  
- score_function  - score function from sklearn.metrics or you own function  
- n_fold - number folds for KFold  
- seed - seed for random  
- cat_features - categorical feature if tou have it in dataset (default [])  
  
<b>return :</b>  
  
- oof_df - dataframe with predict for train part  
- submit - dataframe with predict for test part  
- fi - feature importance of training  
  
   
