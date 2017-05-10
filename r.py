import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
import numpy as np
import xgboost as xgb

df=pd.read_csv("trainphoto_features.csv")
labels=pd.read_csv("train_labels.csv")


labels = labels.set_index(labels['name'])

df= df.set_index(df['img_num'])

df=df.drop('img_num.1',axis=1)

df= pd.merge(df, labels, how='left', left_on=['img_num'], right_on=['name'])

df=df.drop('name',axis=1)
y_train = df['invasive']

trainvars = df.columns[1:-1]
X_train=df[trainvars]
#X_train['ln_t4minBGRhist']=np.log1p(X_train['t4minBGRhist'])
X_train['bestBGRmin']=np.min((X_train['t11minBGRhist'],X_train['t10minBGRhist'],X_train['t9minBGRhist'],X_train['t8minBGRhist'],X_train['t7minBGRhist'],X_train['t6minBGRhist'],X_train['t4minBGRhist'],X_train['t1minBGRhist'],X_train['t2minBGRhist'],X_train['t3minBGRhist'],X_train['t5minBGRhist']))
X_train['avgBGRmin']=np.mean((X_train['t11minBGRhist'],X_train['t10minBGRhist'],X_train['t9minBGRhist'],X_train['t8minBGRhist'],X_train['t7minBGRhist'],X_train['t6minBGRhist'],X_train['t4minBGRhist'],X_train['t1minBGRhist'],X_train['t2minBGRhist'],X_train['t3minBGRhist'],X_train['t5minBGRhist']))
X_train['worstBGRmin']=np.max((X_train['t11minBGRhist'],X_train['t10minBGRhist'],X_train['t9minBGRhist'],X_train['t8minBGRhist'],X_train['t7minBGRhist'],X_train['t6minBGRhist'],X_train['t4minBGRhist'],X_train['t1minBGRhist'],X_train['t2minBGRhist'],X_train['t3minBGRhist'],X_train['t5minBGRhist']))

X_train['t4minBGRxminHSV']=X_train['t4minBGRhist']*X_train['t4minHSVhist']

#scaler = StandardScaler()

print "fix the minimum, and work on scaling/interactions"

rfc =RandomForestClassifier()


param_grid={'n_estimators':[10,100],'max_depth':[3,5,7,None]}

grid_search = GridSearchCV(rfc, param_grid=param_grid,n_jobs=6,cv=3,scoring='roc_auc')
grid_search.fit(X_train,y_train)

results = grid_search.cv_results_

print "Top Random Forest CV models"
for p in sorted(zip(results['mean_test_score'],results['params']),reverse=True)[:20]:
    print p

bestrfc = grid_search.best_estimator_

#Examine the top importance features

importances = bestrfc.feature_importances_
indices = np.argsort(importances)

feature_names = X_train.columns

a=importances[indices]

b=feature_names[indices]
ab= zip(a,b)[::-1]
# Print the feature ranking
print("Top 100 Feature importance ranking:")
for f in range(min(100,len(ab))):
    print ab[f]



xgb_params = {
    'eta': 0.01,
    'min_child_weight':1,
    'max_depth': 7,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'objective': 'binary:logistic',
    'reg_alpha':.0,
    'reg_lambda':.01,
    'silent': 1
}

dtrain = xgb.DMatrix(X_train, y_train)
#dtest = xgb.DMatrix(X_test)

cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=10000, early_stopping_rounds=50,
    verbose_eval=50, metrics='auc',show_stdv=False)


#Make a chart with a test-holdout

from sklearn.model_selection import train_test_split

train2, test2 = train_test_split(df, test_size = 0.2)

trainvars = df.columns[1:-1]
X_train=train2[trainvars]
X_test=test2[trainvars]
y_train = train2['invasive']


#xgtest=xgb.DMatrix(validation[trainvars],label=np.log(validation['SalePrice']))
xgtrain=xgb.DMatrix(X_train,label=y_train)
xgmodel = xgb.train( xgb_params, xgtrain, num_boost_round=15000,verbose_eval=1, obj = None)

y_pred = xgmodel.predict(X_test)

from sklearn.metrics import roc_auc_score

print roc_auc_score(test2['invasive'], y_pred)


'''
y_pred = bestrfc.predict(X_test)

y_pred[y_pred<0]=0

print "rf rmsle",rmsle(np.exp(np.array(y_test)),np.exp(np.array(y_pred)))

pred_rf = bestrfc.predict(X_score)

'''