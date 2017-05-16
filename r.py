import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
import numpy as np
import xgboost as xgb


#Load the feature data from the prior step

df=pd.read_csv("trainphoto_features.csv")
score=pd.read_csv("testphoto_features.csv")
labels=pd.read_csv("train_labels.csv")

labels = labels.set_index(labels['name'])

df= df.set_index(df['img_num'])
score= score.set_index(score['img_num'])
df=df.drop('img_num.1',axis=1)
score= score.drop('img_num.1',axis=1)

huVars = [x for x in df.columns if 'hu' in x]
#hsvVars = [x for x in df.columns if 'HSV' in x]

print "before",df.shape
score = score.drop(huVars,axis=1)
df = df.drop(huVars,axis=1)
#score = score.drop(hsvVars,axis=1)
#df = df.drop(hsvVars,axis=1)
print "after",df.shape

df= pd.merge(df, labels, how='left', left_on=['img_num'], right_on=['name'])

df=df.drop('name',axis=1)
y_train = df['invasive']

trainvars = df.columns[1:-1]
X_train=df[trainvars]
X_score = score[trainvars]
#X_train['ln_t4minBGRhist']=np.log1p(X_train['t4minBGRhist'])
#X_train['bestBGRmin']=np.min((X_train['t11minBGRhist'],X_train['t10minBGRhist'],X_train['t9minBGRhist'],X_train['t8minBGRhist'],X_train['t7minBGRhist'],X_train['t6minBGRhist'],X_train['t4minBGRhist'],X_train['t1minBGRhist'],X_train['t2minBGRhist'],X_train['t3minBGRhist'],X_train['t5minBGRhist']))
#X_train['avgBGRmin']=np.mean((X_train['t11minBGRhist'],X_train['t10minBGRhist'],X_train['t9minBGRhist'],X_train['t8minBGRhist'],X_train['t7minBGRhist'],X_train['t6minBGRhist'],X_train['t4minBGRhist'],X_train['t1minBGRhist'],X_train['t2minBGRhist'],X_train['t3minBGRhist'],X_train['t5minBGRhist']))
#X_train['worstBGRmin']=np.max((X_train['t11minBGRhist'],X_train['t10minBGRhist'],X_train['t9minBGRhist'],X_train['t8minBGRhist'],X_train['t7minBGRhist'],X_train['t6minBGRhist'],X_train['t4minBGRhist'],X_train['t1minBGRhist'],X_train['t2minBGRhist'],X_train['t3minBGRhist'],X_train['t5minBGRhist']))

#X_train['t4minBGRxminHSV']=X_train['t4minBGRhist']*X_train['t4minHSVhist']

#scaler = StandardScaler()

#print "fix the minimum, and work on scaling/interactions"



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
    'eta': 0.009,#1
    'min_child_weight':1,
    'max_depth': 5,#
    'subsample': 0.65,#.65
    'colsample_bytree': 0.8,#8
    'objective': 'binary:logistic',
    'reg_alpha':.25,#20
    'reg_lambda':.1,
    'silent': 1
}


dtrain = xgb.DMatrix(X_train, y_train)



cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=10000, early_stopping_rounds=20,
    verbose_eval=50, metrics='auc',show_stdv=False)



#Make a chart with a test-holdout

from sklearn.model_selection import train_test_split

train2, test2 = train_test_split(df, test_size = 0.2,random_state=200)

trainvars = df.columns[1:-1]
X_train=train2[trainvars]
X_test=test2[trainvars]
y_train = train2['invasive']


#xgtest=xgb.DMatrix(validation[trainvars],label=np.log(validation['SalePrice']))
xgtrain=xgb.DMatrix(X_train,label=y_train)
xgmodel = xgb.train( xgb_params, xgtrain, num_boost_round=1100,verbose_eval=1, obj = None)
xgmodel_full = xgb.train( xgb_params, dtrain, num_boost_round=1100,verbose_eval=1, obj = None)

y_pred_score  = xgmodel_full.predict(xgb.DMatrix(X_score))

score_df = pd.DataFrame(y_pred_score,index=X_score.index)
score_df.index.names = ['name']


score_df.to_csv("score_out4.csv",header=['invasive'])



try:
    d= xgmodel.get_fscore()
    for w in sorted(d, key=d.get, reverse=True):
        print w, d[w]
except:
    print "no fscore"


y_pred = xgmodel.predict(xgb.DMatrix(X_test))

y_pred_diff = y_pred - test2['invasive']

print y_pred_diff[np.abs(y_pred_diff)>.8]

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print roc_auc_score(test2['invasive'], y_pred)

z=y_pred.copy()
z[z>.5]=1
z[z<=.5]=0

print "acc score",accuracy_score(test2['invasive'], z)

print "acc score, no normalize",accuracy_score(test2['invasive'], z,normalize=False)

print confusion_matrix(test2['invasive'],z)


from sklearn import metrics


fpr, tpr, thresholds = metrics.roc_curve(test2['invasive'], y_pred, pos_label=1)

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig("roc_chart.png")
plt.show()

'''



y_pred = bestrfc.predict(X_test)

y_pred[y_pred<0]=0

print "rf rmsle",rmsle(np.exp(np.array(y_test)),np.exp(np.array(y_pred)))

pred_rf = bestrfc.predict(X_score)
'''