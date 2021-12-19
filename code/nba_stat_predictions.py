# -*- coding: utf-8 -*-
"""
Predict Next Season Points, Rebounds, and Assists per 36 minutes for players based on the following parameters:

Age of the current year
Years Pro
Play type features
Tracking features

training data from 2015-2020
test data 2020-21 season
    - minimum of 400 minutes played in either the current season or the next season
    (i.e. if a player played 600 minutes in 2016-17 but only 300 minutes in 17-18, that 2016-17 season for that player is removed)

i.e. {season: 2016, name:"LeBron James", age: 32, years_pro: 10, Drive_freq: ###, ...}
Points, Assists, Rebounds next year would be the dependent variables

Find the Ordinary Least Squared solution and measure the error in the 2020-21 season predictions.

t = [Points_next1;Points_next2;...Points_nextN] = Xw where X = [[1 playerseason1_features];[1 playerseason2_features];...[1 playerseasonN_features]]
and w = [w_0;w_1;w_2;...w_M]
Where N is the total number of player seasons in the dataset (i.e. many players will have up to 5 seasons of data in this set)
and where M is the number of features - 1

so t is a Nx1 matrix, X is a NxM matrix, and w is a Mx1 matrix
"""

from os import error
from nba_api.stats.endpoints import commonallplayers
import json
import numpy as np
from numpy.lib.utils import info
import pandas as pd
from sklearn import linear_model
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import nba_stat_predictions_functions as fnc


# acquire all the data
load_playtype = False # set to true if loading playtype data from NBA.com, False if loading from local file
load_tracking = False # set to true if loading tracking data from NBA.com, False if loading from local file
load_player_seasons = False # set to true if loading player season-by-season data from NBA.com, False if loading from local file

seasons = ['2015-16','2016-17','2017-18','2018-19','2019-20']
# 2019-20 will be the test data to predict 2020-21

nba_players = json.loads(commonallplayers.CommonAllPlayers(league_id='00').get_json())['resultSets'][0]['rowSet']

[play_types,tracking_data] = fnc.get_training_data(seasons,load_playtype,load_tracking)

# get the list of player seasons that qualify (min 400 minutes this season and the next season to qualify)
players_seasons = fnc.get_players_seasons(seasons,nba_players,load_player_seasons)

# Organize this data into the design matrix, X, and the output, (t_pts, t_reb, or t_ast)
X = [] # Design matrix
X_test = [] # test data
w = [] # weight vector
t_pts = [] # next season points per 36 minutes
t_reb = [] # next season rebounds per 36 minutes
t_ast = [] # next season assists per 36 minutes
t_pts_test = [] # next season points per 36 minutes for TEST DATA. Not to be used in training the model!
t_reb_test = [] # next season rebounds per 36 minutes for TEST DATA. Not to be used in training the model!
t_ast_test = [] # next season assists per 36 minutes for TEST DATA. Not to be used in training the model!


features = ['CONSTANT','AGE','AGE_SQUARED','YEARS_PRO','TRANS_FREQ','TRANS_PPP','ISO_FREQ','ISO_PPP',
            'PNR_BH_FREQ','PNR_BH_PPP','PNR_R_FREQ','PNR_R_PPP','POST_UP_FREQ',
            'POST_UP_PPP', 'SPOT_UP_FREQ','SPOT_UP_PPP','HANDOFF_FREQ','HANDOFF_PPP',
            'CUT_FREQ','CUT_PPP','OFF_SCREEN_FREQ','OFF_SCREEN_PPP',
            'PUTBACK_FREQ','PUTBACK_PPP','MISC_FREQ','MISC_PPP','DRIVES','DRIVES_TS','DRIVES_PTS_PERC','DRIVES_AST_PERC','DRIVES_TO_PERC',
            'CATCH_SHOOT_FGA_2PT','CATCH_SHOOT_FGA_3PT','CATCH_SHOOT_EFG',
            'PASSES_MADE','PASSES_RECEIVED','POTENTIAL_AST','AST',
            'PULL_UP_FGA_2PT','PULL_UP_FGA_3PT','PULL_UP_EFG',
            'REB_CHANCES','CONTESTED_DREB','CONTESTED_OREB',
            'TOUCHES','ELBOW_TOUCHES','ELBOW_TS','ELBOW_PTS_PERC','ELBOW_AST_PERC','ELBOW_TO_PERC',
            'POST_TOUCHES','POST_TS','POST_PTS_PERC','POST_AST_PERC','POST_TO_PERC',
            'PAINT_TOUCHES','PAINT_TS','PAINT_PTS_PERC','PAINT_AST_PERC','PAINT_TO_PERC']

# iterate over the two training seasons, and add all features to the player vectors
info_vect = [] # vector containing name, id, and season
for i in range(len(seasons)-1):
    for p in players_seasons[i]:
        pt = fnc.get_playtype_vector(i,p[0],play_types)
        info_vect.append([p[0],p[1],seasons[i]])
        feat_vect = [1,p[2],np.square(p[2]),p[3]] # Age, Age squared, and Years Pro
        for j in range(len(pt)):
            feat_vect.append(pt[j])    # playtype features
        tr = fnc.get_tracking_vector(i,p[0],tracking_data)
        for j in range(len(tr)):
            feat_vect.append(tr[j])     # tracking features
        X.append(feat_vect)
        t_pts.append(p[4])
        t_reb.append(p[5])
        t_ast.append(p[6])

# iterate over the one test season to get the test data
info_vect_test = [] # vector containing name, id, and season
for p in players_seasons[len(seasons)-1]:
    pt = fnc.get_playtype_vector(len(seasons)-1,p[0],play_types)
    # info vector for test data: ID, Name, Season, current season Points/36min, Rebounds/36min,Assists/36min
    info_vect_test.append([p[0],p[1],seasons[len(seasons)-1],p[7],p[8],p[9]])
    feat_vect = [1,p[2],np.square(p[2]),p[3]] # Age, Age squared, and Years Pro
    for j in range(len(pt)):
        feat_vect.append(pt[j])    # playtype features
    tr = fnc.get_tracking_vector(len(seasons)-1,p[0],tracking_data)
    for j in range(len(tr)):
        feat_vect.append(tr[j])     # tracking features
    X_test.append(feat_vect)
    t_pts_test.append(p[4])     # Actual results!!! Not to be used in training!
    t_reb_test.append(p[5])
    t_ast_test.append(p[6])

df_info_training = pd.DataFrame(data=info_vect,columns=['ID','NAME','SEASON'])
df_X_training = pd.DataFrame(data=X,columns=features)

df_test_info = pd.DataFrame(data=info_vect_test,columns=['ID','NAME','SEASON','PP36_prev','RP36_prev','AP36_prev'])
df_X_test = pd.DataFrame(data=X_test,columns=features)

df_info_features = df_info_training.join(df_X_training).drop(labels='CONSTANT',axis=1)
df_test_info_features = df_test_info.join(df_X_test).drop(labels='CONSTANT',axis=1)


# Using the Simple imputer to replace all NaN values with the average of the rest of the set
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X)
X_transform = imp.transform(X)
X_test_transform = imp.transform(X_test)

# Use MinMaxScaler to scale the data between 0 and 1
min_max_scaler = MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_transform)
X_test_minmax = min_max_scaler.transform(X_test_transform)

# Using sci-kit-learn to perform least squares linear regression
reg_pts = linear_model.LinearRegression()
reg_reb = linear_model.LinearRegression()
reg_ast = linear_model.LinearRegression()

# fit the regression model with the training data after processing
reg_pts.fit(X_train_minmax,t_pts)
reg_reb.fit(X_train_minmax,t_reb)
reg_ast.fit(X_train_minmax,t_ast)

df_coef = pd.DataFrame(data=np.transpose([reg_pts.coef_,reg_reb.coef_,reg_ast.coef_]),index=np.transpose(features))

# predict next season stats using test data as the input!
pts_pred = reg_pts.predict(X_test_minmax)
reb_pred = reg_reb.predict(X_test_minmax)
ast_pred = reg_ast.predict(X_test_minmax)

# calculate the squared errors
squared_errors = fnc.calc_squared_error(pts_pred,reb_pred,ast_pred,t_pts_test,t_reb_test,t_ast_test)

# organize data into a dataframe and save in Excel!
df_pred = pd.DataFrame(data=np.transpose([df_test_info['NAME'],df_test_info_features['AGE'],df_test_info_features['YEARS_PRO'],
                                          df_test_info['PP36_prev'],pts_pred,t_pts_test,squared_errors[0],
                                          df_test_info['RP36_prev'],reb_pred,t_reb_test,squared_errors[1],
                                          df_test_info['AP36_prev'],ast_pred,t_ast_test,squared_errors[2],
                                          np.mean(squared_errors,axis=0)]),
                                         columns=['Name','Age','Years Pro',
                                                  'Previous PTS','Predicted PTS','Actual PTS','PTS square error',
                                                  'Previous REB','Predicted REB','Actual REB','REB square error',
                                                  'Previous AST','Predicted AST','Actual AST','AST square error',
                                                  'Average squared error'])

# We can try regularization as well (ridge regression)
# Using sci-kit-learn to perform Ridge regression
ridge_pts1 = linear_model.Ridge(alpha = 1.0)
ridge_reb1 = linear_model.Ridge(alpha = 1.0)
ridge_ast1 = linear_model.Ridge(alpha = 1.0)

ridge_pts10 = linear_model.Ridge(alpha = 10.0)
ridge_reb10 = linear_model.Ridge(alpha = 10.0)
ridge_ast10 = linear_model.Ridge(alpha = 10.0)

ridge_pts30 = linear_model.Ridge(alpha = 30.0)
ridge_reb30 = linear_model.Ridge(alpha = 30.0)
ridge_ast30 = linear_model.Ridge(alpha = 30.0)

# fit the regression model with the training data after processing
ridge_pts1.fit(X_train_minmax,t_pts)
ridge_reb1.fit(X_train_minmax,t_reb)
ridge_ast1.fit(X_train_minmax,t_ast)

ridge_pts10.fit(X_train_minmax,t_pts)
ridge_reb10.fit(X_train_minmax,t_reb)
ridge_ast10.fit(X_train_minmax,t_ast)

ridge_pts30.fit(X_train_minmax,t_pts)
ridge_reb30.fit(X_train_minmax,t_reb)
ridge_ast30.fit(X_train_minmax,t_ast)

df_coef = pd.DataFrame(data=np.transpose([reg_pts.coef_,ridge_pts1.coef_,ridge_pts10.coef_,ridge_pts30.coef_,
                                          reg_reb.coef_,ridge_reb1.coef_,ridge_reb10.coef_,ridge_reb30.coef_,
                                          reg_ast.coef_,ridge_ast1.coef_,ridge_ast10.coef_,ridge_ast30.coef_]),
                        columns=['PTS','PTS (Alpha=1)','PTS (Alpha=10)','PTS (Alpha=30)',
                                 'REB','REB (Alpha=1)','REB (Alpha=10)','REB (Alpha=30)',
                                 'AST','AST (Alpha=1)','AST (Alpha=10)','AST (Alpha=30)'],index=np.transpose(features))

# predict next season stats using test data as the input!
pts_pred_ridge1 = ridge_pts1.predict(X_test_minmax)
reb_pred_ridge1 = ridge_reb1.predict(X_test_minmax)
ast_pred_ridge1 = ridge_ast1.predict(X_test_minmax)

pts_pred_ridge10 = ridge_pts10.predict(X_test_minmax)
reb_pred_ridge10 = ridge_reb10.predict(X_test_minmax)
ast_pred_ridge10 = ridge_ast10.predict(X_test_minmax)

pts_pred_ridge30 = ridge_pts30.predict(X_test_minmax)
reb_pred_ridge30 = ridge_reb30.predict(X_test_minmax)
ast_pred_ridge30 = ridge_ast30.predict(X_test_minmax)

# calculate the squared errors
squared_errors_ridge1 = fnc.calc_squared_error(pts_pred_ridge1,reb_pred_ridge1,ast_pred_ridge1,t_pts_test,t_reb_test,t_ast_test)
squared_errors_ridge10 = fnc.calc_squared_error(pts_pred_ridge10,reb_pred_ridge10,ast_pred_ridge10,t_pts_test,t_reb_test,t_ast_test)
squared_errors_ridge30 = fnc.calc_squared_error(pts_pred_ridge30,reb_pred_ridge30,ast_pred_ridge30,t_pts_test,t_reb_test,t_ast_test)

# organize data into a dataframe and save in Excel!
df_pred_ridge1 = pd.DataFrame(data=np.transpose([df_test_info['NAME'],df_test_info_features['AGE'],df_test_info_features['YEARS_PRO'],
                                          df_test_info['PP36_prev'],pts_pred_ridge1,t_pts_test,squared_errors_ridge1[0],
                                          df_test_info['RP36_prev'],reb_pred_ridge1,t_reb_test,squared_errors_ridge1[1],
                                          df_test_info['AP36_prev'],ast_pred_ridge1,t_ast_test,squared_errors_ridge1[2],
                                          np.mean(squared_errors_ridge1,axis=0)]),
                                         columns=['Name','Age','Years Pro',
                                                  'Previous PTS','Predicted PTS','Actual PTS','PTS square error',
                                                  'Previous REB','Predicted REB','Actual REB','REB square error',
                                                  'Previous AST','Predicted AST','Actual AST','AST square error',
                                                  'Average squared error'])

df_pred_ridge10 = pd.DataFrame(data=np.transpose([df_test_info['NAME'],df_test_info_features['AGE'],df_test_info_features['YEARS_PRO'],
                                          df_test_info['PP36_prev'],pts_pred_ridge10,t_pts_test,squared_errors_ridge10[0],
                                          df_test_info['RP36_prev'],reb_pred_ridge10,t_reb_test,squared_errors_ridge10[1],
                                          df_test_info['AP36_prev'],ast_pred_ridge10,t_ast_test,squared_errors_ridge10[2],
                                          np.mean(squared_errors_ridge10,axis=0)]),
                                         columns=['Name','Age','Years Pro',
                                                  'Previous PTS','Predicted PTS','Actual PTS','PTS square error',
                                                  'Previous REB','Predicted REB','Actual REB','REB square error',
                                                  'Previous AST','Predicted AST','Actual AST','AST square error',
                                                  'Average squared error'])

df_pred_ridge30 = pd.DataFrame(data=np.transpose([df_test_info['NAME'],df_test_info_features['AGE'],df_test_info_features['YEARS_PRO'],
                                          df_test_info['PP36_prev'],pts_pred_ridge30,t_pts_test,squared_errors_ridge30[0],
                                          df_test_info['RP36_prev'],reb_pred_ridge30,t_reb_test,squared_errors_ridge30[1],
                                          df_test_info['AP36_prev'],ast_pred_ridge30,t_ast_test,squared_errors_ridge30[2],
                                          np.mean(squared_errors_ridge30,axis=0)]),
                                         columns=['Name','Age','Years Pro',
                                                  'Previous PTS','Predicted PTS','Actual PTS','PTS square error',
                                                  'Previous REB','Predicted REB','Actual REB','REB square error',
                                                  'Previous AST','Predicted AST','Actual AST','AST square error',
                                                  'Average squared error'])



with pd.ExcelWriter('..\\data\\2020-21_predictions.xlsx') as writer: 
    df_info_features.to_excel(writer, sheet_name='Training Data Features')
    df_test_info_features.to_excel(writer, sheet_name='Test Data Features')
    df_coef.to_excel(writer, sheet_name='Regression Coefficients')
    df_pred.to_excel(writer, sheet_name='Predictions')
    df_pred_ridge1.to_excel(writer, sheet_name='Ridge (alpha=1) Predictions')
    df_pred_ridge10.to_excel(writer, sheet_name='Ridge (alpha=10) Predictions')
    df_pred_ridge30.to_excel(writer, sheet_name='Ridge (alpha=30) Predictions')