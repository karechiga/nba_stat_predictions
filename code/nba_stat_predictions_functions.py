"""
This .py file is meant to be used with the nba_stat_prediction.py file. The nba_stat_prediction.py file utilizes the functions from this file.
"""

from os import error
import nba_api
from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.endpoints import synergyplaytypes
from nba_api.stats.endpoints import commonallplayers
from nba_api.stats.endpoints import commonplayerinfo
from nba_stats_tracking import tracking
import json
import numpy as np
from numpy.lib.utils import info
import pandas as pd
import time
from sklearn import linear_model
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

def get_career_stats(p):
    while(1):
        try:
            time.sleep(0.650)
            season_totals = playercareerstats.PlayerCareerStats(p).season_totals_regular_season.data['data']
            break
        except:
            print("Connection refused by the server..")
            print("retrying after 30 seconds")
            time.sleep(30)
            continue

    return season_totals

def get_common_player_data(p):
    while(1):
        try:
            time.sleep(0.650)
            info = commonplayerinfo.CommonPlayerInfo(p).common_player_info.data['data'][0]
            break
        except:
            print("Connection refused by the server..")
            print("retrying after 30 seconds")
            time.sleep(30)
            continue

    return info

def get_playtype_data(season_id,play_type):
    while(1):
        try:
            time.sleep(0.650)
            playtype_data = synergyplaytypes.SynergyPlayTypes(type_grouping_nullable='offensive',player_or_team_abbreviation='P',season=season_id,play_type_nullable=play_type,league_id='00')
            break
        except:
            print("Connection refused by the server..")
            print("retrying after 30 seconds")
            time.sleep(30)
            continue

    return playtype_data.synergy_play_type.data

def get_tracking_data(stat_measure,season,season_types,entity_type):
    while(1):
        try:
            time.sleep(0.650)
            tracking_data = tracking.aggregate_full_season_tracking_stats_for_seasons(stat_measure,[season],season_types,entity_type)
            break
        except:
            print("Connection refused by the server..")
            print("retrying after 30 seconds")
            time.sleep(30)
            continue

    return tracking_data

def get_playtype_features(seasons):
    play_types = ['Transition','Isolation','PRBallHandler','PRRollman','Postup','Spotup','Handoff','Cut','OffScreen','OffRebound','Misc']
    seasons_arr = []
    for season in seasons:
        play_types_arr = []
        for play_type in play_types:
            play_types_arr.append(get_playtype_data(season,play_type))
        seasons_arr.append(play_types_arr)
    return seasons_arr

def get_tracking_features(seasons):
    stat_measures = ['Drives','CatchShoot','Passing','Possessions','PullUpShot','Rebounding','ElbowTouch','PostTouch','PaintTouch']
    season_types = ['Regular Season']
    entity_type = 'player'
    season_tracking_stats = []
    for season in seasons:
        tracking_stats = []
        for stat_measure in stat_measures:
            time.sleep(0.650)
            tracking_stats.append(get_tracking_data(stat_measure,season,season_types,entity_type))
        season_tracking_stats.append(tracking_stats)
    return season_tracking_stats

def get_training_data(seasons,load_playtype,load_tracking):
    # loading playtype features by season
    if load_playtype:
        play_types_training = get_playtype_features(seasons) # gets every season's playtype stats
        with open('..\\data\\nba_playtypes.txt','w') as json_file: # dump to a .txt file
            json.dump(play_types_training,json_file)
    else: # load from .txt file
        with open('..\\data\\nba_playtypes.txt') as json_file:
            play_types_training = json.load(json_file)

    if load_tracking:
        tracking_training = get_tracking_features(seasons) # gets all tracking stats for multiple seasons
        with open('..\\data\\nba_tracking.txt','w') as json_file: # dump to a .txt file
            json.dump(tracking_training,json_file)
    else: # load from .txt file
        with open('..\\data\\nba_tracking.txt') as json_file:
            tracking_training = json.load(json_file)
    return play_types_training,tracking_training

def get_minutes_pts_reb_ast(stats,season_year):
    # get the number of minutes played this season and next as well as points, rebounds, and assists
    if not stats:
        # if a player has no stats at all, return 0's
        return 0,0,0,0,0,0,0,0
    minutes = 0
    curr_pts = 0
    curr_reb = 0
    curr_ast = 0
    next_pts = 0
    next_reb = 0
    next_ast = 0
    next_minutes = 0
    if int(stats[len(stats)-1][1][0:4])==season_year:   
        # if their last year in the league was this season, 
        # return 0's since they didn't play next season.
        # they will be omitted from the training data.
        return 0,0,0,0,0,0,0,0
    for j in range(len(stats)):
        if int(stats[j][1][0:4])==season_year:  
            # if players were traded midseason, the same season will show up multiple times on their career stats. 
            # Want to get their combined minutes.
            while (int(stats[j+1][1][0:4])==season_year):
                j += 1
            minutes = stats[j][8]   # total minutes played in the current season
            curr_pts = stats[j][26]
            curr_reb = stats[j][20]
            curr_ast = stats[j][21]
            if int(stats[len(stats)-1][1][0:4])==season_year+1: # if their last season is next season, take that minutes value
                next_minutes = stats[len(stats)-1][8]
                next_pts = stats[len(stats)-1][26]
                next_reb = stats[len(stats)-1][20]
                next_ast = stats[len(stats)-1][21]
                break
            if not (int(stats[j+1][1][0:4])==season_year+1):   
                # if their next playing year, isn't next season, the player will be removed from the dataset
                # this would occur in rare instances where players have to miss a whole season due to injury.
                return minutes,0,curr_pts,curr_reb,curr_ast,0,0,0
            while (int(stats[j+1][1][0:4])==season_year+1):
                j += 1
            
            next_minutes = stats[j][8]
            next_pts = stats[j][26]
            next_reb = stats[j][20]
            next_ast = stats[j][21]
    return minutes,next_minutes,curr_pts,curr_reb,curr_ast,next_pts,next_reb,next_ast

def get_players_seasons(seasons,nba_players,load_player_seasons):
    # load_player_seasons is a boolean input. deciding whether we are loading from NBA.com or locally
    # outputs an array which represents all the id's of players that played in that season AND the next season
    if not load_player_seasons:
        with open('..\\data\\nba_player_seasons.txt') as json_file:
            players_seasons = json.load(json_file)
        return players_seasons

    players_seasons = []
    first_season_year = int(seasons[0][0:4])    # integer of the first season year in the list of seasons
    last_season_year = int(seasons[len(seasons)-1][0:4])    # integer of the last season year in the list of seasons
    for i in range(len(seasons)):
        players_seasons.append([])
    for p in nba_players:
        if (int(p[4]) <= last_season_year) & (int(p[5]) >= first_season_year+1):   # if player was active during at least one of the seasons
            player_career_stats = get_career_stats(p[0])
            # If a player played less than 400 minutes this season or the next season, don't include them in this dataset
            for i in range(len(seasons)):
                [minutes,next_minutes,curr_pts,curr_reb,curr_ast,next_pts,next_reb,next_ast] = get_minutes_pts_reb_ast(player_career_stats,int(seasons[i][0:4]))
                if (minutes >= 400) & (next_minutes >= 400):
                    player_info = get_common_player_data(p[0])
                    # Getting player id, name, age, years pro, next season pts/36min, next season reb/36min, next season ast/36min, 
                    # this season pts/36min, this season reb/36min, this season ast/36min
                    pl = [player_info[0],player_info[3],int(seasons[i][0:4])-int(player_info[7][0:4]),
                          int(seasons[i][0:4])-int(player_info[24]),next_pts*36/(next_minutes),next_reb*36/(next_minutes),next_ast*36/(next_minutes),
                          curr_pts*36/(minutes),curr_reb*36/(minutes),curr_ast*36/(minutes)]
                    players_seasons[i].append(pl)
    with open('..\\data\\nba_player_seasons.txt','w') as json_file: # dump to a .txt file
        json.dump(players_seasons,json_file)
    return players_seasons

def get_playtype_vector(i,id,play_types):
    vector = []
    for pt in play_types[i]:
        df = pd.DataFrame(data=pt['data'],columns=pt['headers'])
        freq = df.loc[df['PLAYER_ID'] == id]['POSS_PCT'].values
        ppp = df.loc[df['PLAYER_ID'] == id]['PPP'].values
        if len(ppp) == 0:
            ppp = np.NaN
            freq = 0
        elif len(ppp) == 1:
            ppp = ppp[0]
            freq = freq[0]
        else:      
            # Players who were traded will show up more than once. Want to combine their features into one
            points = 0
            poss = 0
            total_poss = 0
            for i in range(len(ppp)):
                points += df.loc[df['PLAYER_ID'] == id]['PTS'].values[i]
                poss += df.loc[df['PLAYER_ID'] == id]['POSS'].values[i]
                total_poss += np.round(poss / df.loc[df['PLAYER_ID'] == id]['POSS_PCT'].values[i])
            ppp = points/poss
            freq = poss/total_poss
        vector.append(freq)
        vector.append(ppp)
    return vector

def calc_true_shooting(points, fga, fta):
    return 0.5*points/(fga+0.44*fta)

def get_tracking_vector(i,id,tracking_stats):
    vector = []
    # Add Driving stats
    # 'DRIVES','DRIVES_TS','DRIVES_PTS_PERC','DRIVES_AST_PERC','DRIVES_TO_PERC'
    df = pd.DataFrame(tracking_stats[i][0][0])
    drives = df.loc[df['PLAYER_ID'] == id]['DRIVES'].values
    drive_pts = df.loc[df['PLAYER_ID'] == id]['DRIVE_PTS'].values
    drive_fga = df.loc[df['PLAYER_ID'] == id]['DRIVE_FGA'].values
    drive_fta = df.loc[df['PLAYER_ID'] == id]['DRIVE_FTA'].values
    drive_ast = df.loc[df['PLAYER_ID'] == id]['DRIVE_AST'].values
    drive_to = df.loc[df['PLAYER_ID'] == id]['DRIVE_TOV'].values
    if (len(drives) == 0) | (drives[0] == 0):
        drives = 0
        drives_ts = np.nan
        drives_pts_perc = np.nan
        drives_ast_perc = np.nan
        drive_to_perc = np.nan
    elif len(drives) == 1:
        drives_ts = calc_true_shooting(drive_pts[0],drive_fga[0],drive_fta[0])
        drives_pts_perc = drive_pts[0] / drives[0]
        drives_ast_perc = drive_ast[0] / drives[0]
        drive_to_perc = drive_to[0] / drives[0]
        drives = 36*drives[0] / df.loc[df['PLAYER_ID'] == id]['MIN'].values[0] # normalized to per 36 minutes
    vector.append(drives),vector.append(drives_ts),vector.append(drives_pts_perc)
    vector.append(drives_ast_perc),vector.append(drive_to_perc)
    
    # Add Catch and Shoot stats
    # 'CATCH_SHOOT_FGA_2PT','CATCH_SHOOT_FGA_3PT','CATCH_SHOOT_EFG'
    df = pd.DataFrame(tracking_stats[i][1][0])
    cns_fga = df.loc[df['PLAYER_ID'] == id]['CATCH_SHOOT_FGA'].values
    if (len(cns_fga) == 0) | (cns_fga[0] == 0):
        cns_2fga = 0
        cns_3fga = 0
        cns_efg = np.nan
    elif len(cns_fga) == 1:
        cns_2fga = (df.loc[df['PLAYER_ID'] == id]['CATCH_SHOOT_FGA'].values[0] - df.loc[df['PLAYER_ID'] == id]['CATCH_SHOOT_FG3A'].values[0]) * 36 / df.loc[df['PLAYER_ID'] == id]['MIN'].values[0]
        cns_3fga = df.loc[df['PLAYER_ID'] == id]['CATCH_SHOOT_FG3A'].values[0]*36/df.loc[df['PLAYER_ID'] == id]['MIN'].values[0]
        cns_efg = calc_true_shooting(df.loc[df['PLAYER_ID'] == id]['CATCH_SHOOT_PTS'].values[0],cns_fga[0],0)
    vector.append(cns_2fga),vector.append(cns_3fga),vector.append(cns_efg)
    
    # Add Passing stats
    # 'PASSES_MADE','PASSES_RECEIVED','POTENTIAL_AST','AST'
    df = pd.DataFrame(tracking_stats[i][2][0])
    vector.append(df.loc[df['PLAYER_ID'] == id]['PASSES_MADE'].values[0]*36/df.loc[df['PLAYER_ID'] == id]['MIN'].values[0])
    vector.append(df.loc[df['PLAYER_ID'] == id]['PASSES_RECEIVED'].values[0]*36/df.loc[df['PLAYER_ID'] == id]['MIN'].values[0])
    vector.append(df.loc[df['PLAYER_ID'] == id]['POTENTIAL_AST'].values[0]*36/df.loc[df['PLAYER_ID'] == id]['MIN'].values[0])
    vector.append(df.loc[df['PLAYER_ID'] == id]['AST'].values[0]*36/df.loc[df['PLAYER_ID'] == id]['MIN'].values[0])
    
    # Add Touches stats
    # 'TOUCHES'
    df = pd.DataFrame(tracking_stats[i][3][0])
    vector.append(df.loc[df['PLAYER_ID'] == id]['TOUCHES'].values[0]*36/df.loc[df['PLAYER_ID'] == id]['MIN'].values[0])

    # Add Pull-up stats
    # 'PULL_UP_FGA_2PT','PULL_UP_FGA_3PT','PULL_UP_EFG'
    df = pd.DataFrame(tracking_stats[i][4][0])
    pullup_fga = df.loc[df['PLAYER_ID'] == id]['PULL_UP_FGA'].values
    if (len(pullup_fga) == 0) | (pullup_fga[0] == 0):
        pullup_2fga = 0
        pullup_3fga = 0
        pullup_efg = np.nan
    elif len(pullup_fga) == 1:
        pullup_2fga = (df.loc[df['PLAYER_ID'] == id]['PULL_UP_FGA'].values[0] - df.loc[df['PLAYER_ID'] == id]['PULL_UP_FG3A'].values[0]) * 36 / df.loc[df['PLAYER_ID'] == id]['MIN'].values[0]
        pullup_3fga = df.loc[df['PLAYER_ID'] == id]['PULL_UP_FG3A'].values[0]*36/df.loc[df['PLAYER_ID'] == id]['MIN'].values[0]
        pullup_efg = calc_true_shooting(df.loc[df['PLAYER_ID'] == id]['PULL_UP_PTS'].values[0],pullup_fga[0],0)
    vector.append(pullup_2fga),vector.append(pullup_3fga),vector.append(pullup_efg)

    
    # Add Rebounding stats
    # 'REB_CHANCES','CONTESTED_DREB','CONTESTED_OREB'
    df = pd.DataFrame(tracking_stats[i][5][0])
    vector.append(df.loc[df['PLAYER_ID'] == id]['REB_CHANCES'].values[0]*36/df.loc[df['PLAYER_ID'] == id]['MIN'].values[0])
    vector.append(df.loc[df['PLAYER_ID'] == id]['DREB_CONTEST'].values[0]*36/df.loc[df['PLAYER_ID'] == id]['MIN'].values[0])
    vector.append(df.loc[df['PLAYER_ID'] == id]['OREB_CONTEST'].values[0]*36/df.loc[df['PLAYER_ID'] == id]['MIN'].values[0])

    
    # Add Elbow Touches stats
    # 'ELBOW_TOUCHES','ELBOW_TS','ELBOW_PTS_PERC','ELBOW_AST_PERC','ELBOW_TO_PERC',
    df = pd.DataFrame(tracking_stats[i][6][0])
    elbow = df.loc[df['PLAYER_ID'] == id]['ELBOW_TOUCHES'].values
    elbow_pts = df.loc[df['PLAYER_ID'] == id]['ELBOW_TOUCH_PTS'].values
    elbow_fga = df.loc[df['PLAYER_ID'] == id]['ELBOW_TOUCH_FGA'].values
    elbow_fta = df.loc[df['PLAYER_ID'] == id]['ELBOW_TOUCH_FTA'].values
    elbow_ast = df.loc[df['PLAYER_ID'] == id]['ELBOW_TOUCH_AST'].values
    elbow_to = df.loc[df['PLAYER_ID'] == id]['ELBOW_TOUCH_TOV'].values
    if (len(elbow) == 0) | (elbow[0] == 0):
        elbow = 0
        elbow_ts = np.nan
        elbow_pts_perc = np.nan
        elbow_ast_perc = np.nan
        elbow_to_perc = np.nan
    elif len(elbow) == 1:
        elbow_ts = calc_true_shooting(elbow_pts[0],elbow_fga[0],elbow_fta[0])
        elbow_pts_perc = elbow_pts[0] / elbow[0]
        elbow_ast_perc = elbow_ast[0] / elbow[0]
        elbow_to_perc = elbow_to[0] / elbow[0]
        elbow = 36*elbow[0] / df.loc[df['PLAYER_ID'] == id]['MIN'].values[0] # normalized to per 36 minutes
    vector.append(elbow),vector.append(elbow_ts),vector.append(elbow_pts_perc)
    vector.append(elbow_ast_perc),vector.append(elbow_to_perc)
    
    # Add Post Touches stats
    # 'POST_TOUCHES','POST_TS','POST_PTS_PERC','POST_AST_PERC','POST_TO_PERC',
    df = pd.DataFrame(tracking_stats[i][7][0])
    post = df.loc[df['PLAYER_ID'] == id]['POST_TOUCHES'].values
    post_pts = df.loc[df['PLAYER_ID'] == id]['POST_TOUCH_PTS'].values
    post_fga = df.loc[df['PLAYER_ID'] == id]['POST_TOUCH_FGA'].values
    post_fta = df.loc[df['PLAYER_ID'] == id]['POST_TOUCH_FTA'].values
    post_ast = df.loc[df['PLAYER_ID'] == id]['POST_TOUCH_AST'].values
    post_to = df.loc[df['PLAYER_ID'] == id]['POST_TOUCH_TOV'].values
    if (len(post) == 0) | (post[0] == 0):
        post = 0
        post_ts = np.nan
        post_pts_perc = np.nan
        post_ast_perc = np.nan
        post_to_perc = np.nan
    elif len(post) == 1:
        post_ts = calc_true_shooting(post_pts[0],post_fga[0],post_fta[0])
        post_pts_perc = post_pts[0] / post[0]
        post_ast_perc = post_ast[0] / post[0]
        post_to_perc = post_to[0] / post[0]
        post = 36*post[0] / df.loc[df['PLAYER_ID'] == id]['MIN'].values[0] # normalized to per 36 minutes
    vector.append(post),vector.append(post_ts),vector.append(post_pts_perc)
    vector.append(post_ast_perc),vector.append(post_to_perc)

    # Add Paint Touches stats
    # 'PAINT_TOUCHES','PAINT_TS','PAINT_PTS_PERC','PAINT_AST_PERC','PAINT_TO_PERC'
    df = pd.DataFrame(tracking_stats[i][8][0])
    paint = df.loc[df['PLAYER_ID'] == id]['PAINT_TOUCHES'].values
    paint_pts = df.loc[df['PLAYER_ID'] == id]['PAINT_TOUCH_PTS'].values
    paint_fga = df.loc[df['PLAYER_ID'] == id]['PAINT_TOUCH_FGA'].values
    paint_fta = df.loc[df['PLAYER_ID'] == id]['PAINT_TOUCH_FTA'].values
    paint_ast = df.loc[df['PLAYER_ID'] == id]['PAINT_TOUCH_AST'].values
    paint_to = df.loc[df['PLAYER_ID'] == id]['PAINT_TOUCH_TOV'].values
    if (len(paint) == 0) | (paint[0] == 0):
        paint = 0
        paint_ts = np.nan
        paint_pts_perc = np.nan
        paint_ast_perc = np.nan
        paint_to_perc = np.nan
    elif len(paint) == 1:
        paint_ts = calc_true_shooting(paint_pts[0],paint_fga[0],paint_fta[0])
        paint_pts_perc = paint_pts[0] / paint[0]
        paint_ast_perc = paint_ast[0] / paint[0]
        paint_to_perc = paint_to[0] / paint[0]
        paint = 36*paint[0] / df.loc[df['PLAYER_ID'] == id]['MIN'].values[0] # normalized to per 36 minutes
    vector.append(paint),vector.append(paint_ts),vector.append(paint_pts_perc)
    vector.append(paint_ast_perc),vector.append(paint_to_perc)

    return vector

def calc_squared_error(pts_pred,reb_pred,ast_pred,t_pts_test,t_reb_test,t_ast_test):
    return [np.square((pts_pred-t_pts_test)/t_pts_test),np.square((reb_pred-t_reb_test)/t_reb_test),np.square((ast_pred-t_ast_test)/t_ast_test)]
