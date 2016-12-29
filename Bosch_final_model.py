
# coding: utf-8

# In[66]:

################################ DATE PROCESSING ##################################
import time
import pandas as pd
import numpy as np
import xgboost as xgb


tr = pd.read_csv('training_date_processed.csv',header=0, sep=',', quotechar='"')
te = pd.read_csv('test_date_processed.csv', header=0, sep=',', quotechar='"')

date_df = tr.append(te)

del tr
del te

# #get sequence for each production line
unique_l = date_df[['prod_line_key']].drop_duplicates()   
unique_l['prod_line_seq'] = np.arange(len(unique_l))

date_df = pd.merge (date_df, unique_l, on = 'prod_line_key')

# #get sequence for each station line
unique_s = date_df[['station_line_key']].drop_duplicates()   
unique_s['station_line_seq'] = np.arange(len(unique_s))

date_df = pd.merge (date_df, unique_s, on = 'station_line_key')

###get presence for each station
num_underscore = 49
for i in range(0,num_underscore):
    df = pd.DataFrame(date_df.station_line_key.str.split('_',1).tolist(),columns = ['S'+str(i),'station_line_key'])   
    date_df = date_df.drop(['station_line_key'], axis = 1)
    date_df = pd.concat([date_df, df], axis=1)

del df
date_df['S49'] = date_df['station_line_key']

s0_s11_cols = ['S0','S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11']
s12_s23_cols = ['S12','S13','S14','S15','S16','S17','S18','S19','S20','S21','S22','S23']
s24_s25_cols = ['S24','S25']
s26_s28_cols = ['S26','S27','S28']
s29_s38_cols = ['S29','S30','S31','S32','S33','S34','S35','S36','S37','S38']
s39_s51_cols = ['S39','S40','S41','S42','S43','S44','S45','S46','S47','S48','S49']

key_list = [s0_s11_cols,s12_s23_cols,s24_s25_cols,s26_s28_cols,s29_s38_cols,s39_s51_cols]
###
###
##loop for creating the 'presence' key -- starts
for i in range(0,len(key_list)):
    
    col_key = key_list[i]

    for col in col_key:
        date_df[str(i)+'_presence'] = date_df[col_key].max(axis = 1).astype(int)
            
    date_df = date_df.drop(col_key, axis = 1)
##loop for creating the 'presence' key -- ends
###
###

presence_keys = [col for col in date_df if '_presence' in col]
date_df['new_station_key']  = date_df[presence_keys].apply(lambda x: '_'.join(map(str,x)), axis = 1)

unique_l = date_df[['new_station_key']].drop_duplicates()
unique_l['new_station_seq'] = np.arange(len(unique_l))

date_df = pd.merge(date_df, unique_l, on = ['new_station_key'])
        

##drop the line keys
date_df = date_df.drop(['prod_line_key','station_line_key','new_station_key'],axis = 1)


date_df['starttime'] = date_df['starttime'].fillna(-1)
date_df = date_df.sort_values(by = ['starttime','Id'],ascending = True)

date_df['diff_id_1_previous'] = date_df['Id'].diff().fillna(9999999).astype(int) #the difference from previous ID
date_df['diff_id_2_previous'] = date_df['Id'].diff(periods=2).fillna(9999999).astype(int) #the difference from previous ID

date_df['diff_id_1_next'] = date_df['Id'].iloc[::-1].diff().fillna(9999999).astype(int) # the difference from next ID
date_df['diff_id_2_next'] = date_df['Id'].iloc[::-1].diff(periods=2).fillna(9999999).astype(int) # the difference from next ID
date_df['diff_id_3_next'] = date_df['Id'].iloc[::-1].diff(periods=3).fillna(9999999).astype(int) # the difference from next ID

date_df['weight_of_seq1'] = 1 + 2 * (date_df['diff_id_1_previous'] > 1) + 1 * (date_df['diff_id_1_next'] < -1)
date_df['weight_of_seq2'] = 1 + 2 * (date_df['diff_id_1_previous'] < 1) + 1 * (date_df['diff_id_1_next'] > -1)
date_df['weight_of_seq3'] = 1 + 3 * (date_df['diff_id_1_previous'] > 1) + 1 * (date_df['diff_id_1_next'] < -1)
date_df['weight_of_seq4'] = 1 + 3 * (date_df['diff_id_1_previous'] < 1) + 1 * (date_df['diff_id_1_next'] > -1)


##get the response information
interim_cols = ['Id','Response']
train_numeric_interim = pd.read_csv('train_numeric_bosch_csv.zip', compression='zip', header=0, sep=',', quotechar='"'
                            , usecols = interim_cols)
date_df = pd.merge(date_df, train_numeric_interim, on = 'Id',how = 'left')
del train_numeric_interim
date_df['Response'] = date_df['Response'].fillna(0)


# batch_response_seen : value indicating how many times positive response is previously seen in the batch
date_df['cumsum1'] = date_df.groupby(['starttime'])['Response'].cumsum().astype(int)
date_df['batch_response_seen'] = (date_df['cumsum1'] - date_df['Response']).where(date_df['Response'] == 1)
date_df['batch_response_seen'] = date_df['batch_response_seen'].fillna(date_df['cumsum1']).astype(int)

## sequence number within starttime
date_df['starttime_seq_row'] = date_df.groupby(['starttime'])['Id'].rank(method = 'dense').astype(int)

# rank order of the starttime
date_df = date_df.sort_values(by = ['starttime','station_line_seq','Id'],ascending = True)
date_df['starttime_gp'] = date_df['starttime'].rank(method='dense').astype(int)

# rank of id's within the group of starttime and station_line_seq
tmp = date_df.groupby(['starttime','station_line_seq']).size()
rank = tmp.map(range)
rank =[item for sublist in rank for item in sublist]
date_df['sttime_station_seq'] = rank
date_df['sttime_station_seq'] = date_df['sttime_station_seq']+1

# Batch : Based on start time
# Batch_First : Binary value indicating if the observation is the first observation of the batch
# Batch_last : Binary value indicating if the observation is the last observation of the batch
date_df['batch_first'] = (date_df.starttime != date_df.starttime.shift(1)).astype(int)
date_df['batch_last'] = (date_df.starttime != date_df.starttime.shift(-1)).astype(int)


# Group : Based on combination of start time and station_line_seq
# Group_change : Binary value if the observation is the first observation of a new group in a batch
# Group_First: Binary value indicating if the observation is the first observation of the group
# Group_Last: Binary value indicating if the observation is the last observation of the group

# count of positive responses in the group

date_df['group_change'] = date_df['sttime_station_seq'].map(lambda x:0 if x > 1 else x)
date_df['group_first'] = (date_df.station_line_seq != date_df.station_line_seq.shift(1)).astype(int)
date_df['group_last'] = (date_df.station_line_seq != date_df.station_line_seq.shift(-1)).astype(int)

# Group_Count: Total number of observations in the group
group_count = date_df.groupby(['starttime','station_line_seq'])['Id'].agg([('group_count',np.size)]).reset_index()
date_df = date_df.merge(group_count,how = 'left', on =['starttime','station_line_seq'])
del group_count

#starttime count : Total Number of observations taken at in a particular time
starttime_count = date_df.groupby(['starttime'])['Id'].agg([('sttime_count',np.size)]).reset_index()
date_df = date_df.merge(starttime_count, how = 'left', on = ['starttime'])
del starttime_count

#starttime sequence count : Total number of station lines in a particular time period
starttime_st_count = date_df.groupby(['starttime'])['station_line_seq'].agg([('starttime_st_count', pd.Series.nunique)]).reset_index()
date_df = date_df.merge(starttime_st_count, how = 'left',on = ['starttime'])
del starttime_st_count

# group_response_seen : value indicating how many times positive response is previously seen in the group
date_df['cumsum2'] = date_df.groupby(['starttime','station_line_seq'])['Response'].cumsum().astype(int)
date_df['group_response_seen'] = (date_df['cumsum2'] - date_df['Response']).where(date_df['Response'] == 1)
date_df['group_response_seen'] = date_df['group_response_seen'].fillna(date_df['cumsum2']).astype(int)


# In[67]:


date_df = date_df.drop(['cumsum1','cumsum2'], axis = 1)

# response_prior : Binary value indicating if positive response is seen in previous observation within the group
# response_next : Binary value indicating if positive response is seen in next observation within the group
date_df['response_prior_gp'] = date_df.Response.shift(1).fillna(999).astype(int)
date_df['response_prior_gp'][(date_df['group_first'] == 1)] = 0 
date_df['response_next_gp'] = date_df.Response.shift(-1).fillna(999).astype(int)
date_df['response_next_gp'][(date_df['group_last'] == 1)] = 0 

date_df['response_prior_batch'] = date_df.Response.shift(1).fillna(999).astype(int)
date_df['response_prior_batch'][(date_df['batch_first'] == 1)] = 0 
date_df['response_next_batch'] = date_df.Response.shift(-1).fillna(999).astype(int)
date_df['response_next_batch'][(date_df['batch_last'] == 1)] = 0 


# weekday = Day of the week when the observation was taken
# 0.01 == 6 mins -- Picked up from one of the forum discussions
date_df ['day'] = (((date_df['starttime']/0.01)*6)/60)/24
date_df['actual_day'] = date_df.day.apply(np.ceil)
date_df['weekday'] = date_df['actual_day'].map(lambda x: 7 if x % 7 == 0 else x % 7)
date_df.loc[date_df['actual_day'] <=0, 'weekday'] = -1 ##this is because the mod doesn't work for negative values

date_df ['hour'] = ((date_df['starttime']/0.01)*6)/60
date_df['actual_hour'] = date_df.hour.apply(np.ceil)
date_df['hour_of_day'] = date_df['actual_hour'].map(lambda x: 24 if x % 24 == 0 else x % 24)
date_df.loc[date_df['actual_hour'] < 0, 'hour_of_day'] = -1 ##this is because the mod doesn't work for negative values

date_df = date_df.drop(['day','actual_day','hour','actual_hour'], axis = 1)

#obs_weekday : Number of observations per weekday
obs_weekday = date_df.groupby(['weekday'])['Id'].agg([('obs_weekday', np.size)]).reset_index()
date_df = date_df.merge(obs_weekday,how = 'left', on =['weekday'])

date_df = date_df.drop(['Response'], axis = 1)


##get sequence for each S32,33,34 station lines
date_df['s32_s33_s34_key'] = date_df.L3_S32_KEY.astype(str) + date_df.L3_S33_KEY.astype(str)+ date_df.L3_S34_KEY.astype(str)
unique_l = date_df[['s32_s33_s34_key']].drop_duplicates()   
unique_l['s32_s33_s34_seq'] = np.arange(len(unique_l))
date_df = pd.merge (date_df, unique_l, on = 's32_s33_s34_key')

date_df['w1_s32_s33_key'] = date_df.weight_of_seq1.astype(str) + date_df.L3_S32_KEY.astype(str) + date_df.L3_S33_KEY.astype(str)
unique_l = date_df[['w1_s32_s33_key']].drop_duplicates()   
unique_l['w1_s32_s33_seq'] = np.arange(len(unique_l))
date_df = pd.merge (date_df, unique_l, on = 'w1_s32_s33_key')

date_df['w2_s32_s33_key'] = date_df.weight_of_seq2.astype(str) + date_df.L3_S32_KEY.astype(str) + date_df.L3_S33_KEY.astype(str)
unique_l = date_df[['w2_s32_s33_key']].drop_duplicates()   
unique_l['w2_s32_s33_seq'] = np.arange(len(unique_l))
date_df = pd.merge (date_df, unique_l, on = 'w2_s32_s33_key')

date_df['w3_s32_s33_key'] = date_df.weight_of_seq3.astype(str) + date_df.L3_S32_KEY.astype(str) + date_df.L3_S33_KEY.astype(str)
unique_l = date_df[['w3_s32_s33_key']].drop_duplicates()   
unique_l['w3_s32_s33_seq'] = np.arange(len(unique_l))
date_df = pd.merge (date_df, unique_l, on = 'w3_s32_s33_key')

date_df['w4_s32_s33_key'] = date_df.weight_of_seq4.astype(str) + date_df.L3_S32_KEY.astype(str) + date_df.L3_S33_KEY.astype(str)
unique_l = date_df[['w4_s32_s33_key']].drop_duplicates()   
unique_l['w4_s32_s33_seq'] = np.arange(len(unique_l))
date_df = pd.merge (date_df, unique_l, on = 'w4_s32_s33_key')

date_df = date_df.drop(['w1_s32_s33_key','w2_s32_s33_key','w3_s32_s33_key','w4_s32_s33_key'], axis = 1)

##dropped these features after the feature importance results
date_var_drop_list = ['L3_S34_KEY','L2_KEY','batch_last','L3_SECOND_HALF_KEY','group_first','L3_S32_KEY','L0_KEY'
                        ,'L1_KEY','obs_weekday','L3_S33_KEY','group_change','s32_s33_s34_key']
date_df = date_df.drop(date_var_drop_list, axis = 1)


# In[68]:

###################### CATEGORICAL DATA PREPROCESSING


##this list of categorical variables was decided following these 4 steps
# step 1: drop all duplicated columns
# step 2: keep only those columns where we see a positive response
# step 3: ran a quick XGB using only categorical fields to pick the highly important fields (Top 10 fields)
# step 4: these were added back to the full data (numeric + date) and then the ones who contributed at an overall level were picked

cat_cols =['Id','L3_S32_F3854','L3_S35_F3902','L1_S25_F1852','L3_S35_F3907','L1_S24_F675']

cat_train = pd.read_csv('train_categorical_bosch_csv.zip', compression='zip'
                         , header=0, sep=',', quotechar='"'
                         , usecols = cat_cols)


cat_test = pd.read_csv('test_categorical_bosch_csv.zip', compression='zip'
                         , header=0, sep=',', quotechar='"'
                         , usecols = cat_cols)

cat_train = cat_train.fillna('AA')
cat_test = cat_test.fillna('AA')

id_col = ['Id']

cat_value_cols = np.setdiff1d(cat_cols, id_col)

for cols in cat_value_cols:
    cat_train[cols] = cat_train[cols].str.replace('-','')
    cat_test[cols] = cat_test[cols].str.replace('-','')

cat_total = cat_train.append(cat_test)

cat_total = pd.get_dummies(cat_total, columns = cat_value_cols)


# In[69]:


#prominent numeric features - these are in descending order of significance/importance (90%)
train_cols = ['Id', 'Response','L0_S0_F0','L0_S0_F10','L0_S0_F16','L0_S0_F18','L0_S0_F2','L0_S0_F20','L0_S0_F22','L0_S0_F6'
              ,'L0_S0_F8','L0_S1_F24','L0_S1_F28','L0_S10_F219','L0_S10_F234','L0_S10_F249','L0_S10_F264','L0_S11_F286'
              ,'L0_S11_F294','L0_S11_F306','L0_S11_F314','L0_S11_F326','L0_S12_F330','L0_S12_F340','L0_S12_F346','L0_S12_F348'
              ,'L0_S12_F350','L0_S13_F356','L0_S15_F397','L0_S15_F418','L0_S16_F421','L0_S16_F426','L0_S17_F433','L0_S19_F459'
              ,'L0_S2_F44','L0_S2_F48','L0_S2_F64','L0_S23_F671','L0_S3_F100','L0_S3_F72','L0_S3_F84','L0_S3_F96','L0_S4_F104'
              ,'L0_S4_F109','L0_S5_F116','L0_S6_F122','L0_S6_F132','L0_S7_F142','L0_S9_F155','L0_S9_F160','L0_S9_F165'
              ,'L0_S9_F170','L0_S9_F180','L0_S9_F185','L0_S9_F190','L0_S9_F195','L0_S9_F200','L0_S9_F210','L1_S24_F1102'
              ,'L1_S24_F1212','L1_S24_F1270','L1_S24_F1326','L1_S24_F1366','L1_S24_F1401','L1_S24_F1406','L1_S24_F1463'
              ,'L1_S24_F1467','L1_S24_F1494','L1_S24_F1498','L1_S24_F1512','L1_S24_F1514','L1_S24_F1516','L1_S24_F1518'
              ,'L1_S24_F1520','L1_S24_F1539','L1_S24_F1544','L1_S24_F1565','L1_S24_F1567','L1_S24_F1571','L1_S24_F1575'
              ,'L1_S24_F1578','L1_S24_F1581','L1_S24_F1609','L1_S24_F1622','L1_S24_F1632','L1_S24_F1637','L1_S24_F1652'
              ,'L1_S24_F1662','L1_S24_F1667','L1_S24_F1672','L1_S24_F1685','L1_S24_F1700','L1_S24_F1713','L1_S24_F1723'
              ,'L1_S24_F1728','L1_S24_F1743','L1_S24_F1753','L1_S24_F1758','L1_S24_F1763','L1_S24_F1773','L1_S24_F1778'
              ,'L1_S24_F1783','L1_S24_F1788','L1_S24_F1793','L1_S24_F1798','L1_S24_F1803','L1_S24_F1808','L1_S24_F1812'
              ,'L1_S24_F1816','L1_S24_F1818','L1_S24_F1820','L1_S24_F1822','L1_S24_F1824','L1_S24_F1829','L1_S24_F1831'
              ,'L1_S24_F1834','L1_S24_F1836','L1_S24_F1842','L1_S24_F1844','L1_S24_F1846','L1_S24_F1848','L1_S24_F1850'
              ,'L1_S24_F700','L1_S24_F867','L1_S24_F902','L1_S25_F1938','L1_S25_F2016','L1_S25_F2036','L1_S25_F2161'
              ,'L1_S25_F2167','L1_S25_F2247','L1_S25_F2307','L1_S25_F2449','L2_S26_F3036','L2_S26_F3040','L2_S26_F3047'
              ,'L2_S26_F3051','L2_S26_F3062','L2_S26_F3069','L2_S26_F3073','L2_S26_F3106','L2_S26_F3113','L2_S26_F3117'
              ,'L2_S26_F3121','L2_S27_F3129','L2_S27_F3133','L2_S27_F3140','L2_S27_F3144','L2_S27_F3155','L2_S27_F3162'
              ,'L2_S27_F3166','L2_S27_F3199','L2_S27_F3206','L2_S27_F3210','L2_S27_F3214','L2_S28_F3255','L3_S29_F3315'
              ,'L3_S29_F3318','L3_S29_F3321','L3_S29_F3324','L3_S29_F3327','L3_S29_F3330','L3_S29_F3333','L3_S29_F3336'
              ,'L3_S29_F3339','L3_S29_F3342','L3_S29_F3345','L3_S29_F3348','L3_S29_F3351','L3_S29_F3354','L3_S29_F3357'
              ,'L3_S29_F3360','L3_S29_F3367','L3_S29_F3370','L3_S29_F3373','L3_S29_F3376','L3_S29_F3379','L3_S29_F3382'
              ,'L3_S29_F3385','L3_S29_F3388','L3_S29_F3395','L3_S29_F3401','L3_S29_F3404','L3_S29_F3407','L3_S29_F3412'
              ,'L3_S29_F3421','L3_S29_F3424','L3_S29_F3427','L3_S29_F3430','L3_S29_F3433','L3_S29_F3436','L3_S29_F3439'
              ,'L3_S29_F3449','L3_S29_F3452','L3_S29_F3455','L3_S29_F3458','L3_S29_F3461','L3_S29_F3464','L3_S29_F3467'
              ,'L3_S29_F3473','L3_S29_F3476','L3_S29_F3479','L3_S30_F3494','L3_S30_F3499','L3_S30_F3504','L3_S30_F3509'
              ,'L3_S30_F3514','L3_S30_F3519','L3_S30_F3524','L3_S30_F3529','L3_S30_F3534','L3_S30_F3544','L3_S30_F3554'
              ,'L3_S30_F3564','L3_S30_F3569','L3_S30_F3574','L3_S30_F3579','L3_S30_F3584','L3_S30_F3589','L3_S30_F3604'
              ,'L3_S30_F3609','L3_S30_F3624','L3_S30_F3629','L3_S30_F3634','L3_S30_F3639','L3_S30_F3644','L3_S30_F3649'
              ,'L3_S30_F3664','L3_S30_F3669','L3_S30_F3674','L3_S30_F3684','L3_S30_F3689','L3_S30_F3704','L3_S30_F3709'
              ,'L3_S30_F3744','L3_S30_F3749','L3_S30_F3754','L3_S30_F3759','L3_S30_F3764','L3_S30_F3769','L3_S30_F3774'
              ,'L3_S30_F3784','L3_S30_F3794','L3_S30_F3799','L3_S30_F3804','L3_S30_F3809','L3_S30_F3819','L3_S30_F3829'
              ,'L3_S32_F3850','L3_S33_F3855','L3_S33_F3857','L3_S33_F3859','L3_S33_F3865','L3_S33_F3873','L3_S34_F3876'
              ,'L3_S34_F3880','L3_S34_F3882','L3_S35_F3889','L3_S35_F3894','L3_S35_F3896','L3_S36_F3920','L3_S36_F3924'
              ,'L3_S36_F3938','L3_S38_F3952','L3_S38_F3956','L3_S38_F3960','L3_S40_F3982','L3_S47_F4158','L3_S49_F4236'
              ,'L3_S32_F3850','L3_S33_F3855','L3_S33_F3857','L3_S33_F3859','L3_S33_F3865','L3_S33_F3867','L3_S33_F3873'
              ,'L3_S34_F3876','L3_S34_F3880','L3_S34_F3882','L3_S35_F3889','L3_S35_F3894','L3_S35_F3896','L3_S36_F3920'
              ,'L3_S36_F3924','L3_S36_F3938','L3_S38_F3952','L3_S38_F3956','L3_S38_F3960','L3_S40_F3982','L3_S47_F4158'
              ,'L3_S49_F4236']

target_col = ['Response']

test_cols = np.setdiff1d(train_cols, target_col)

#### LOAD TRAINING DATA
train_1 = pd.read_csv('train_numeric_bosch_csv.zip', compression='zip', header=0, sep=',', quotechar='"', usecols = train_cols)
train_feature_cols = [col for col in train_1 if '_' in col]
train_1 = train_1.fillna(99)
train_1['feat_concat'] = train_1[train_feature_cols].apply(lambda x: '_'.join(map(str, x)), axis=1)
train_1 = train_1[['Id','feat_concat']]

#### LOAD TEST DATA
test_1 = pd.read_csv('test_numeric_bosch_csv.zip', compression='zip', header=0, sep=',', quotechar='"', usecols = test_cols)
test_feature_cols = [col for col in test_1 if '_' in col]
test_1 = test_1.fillna(99)
test_1['feat_concat'] = test_1[test_feature_cols].apply(lambda x: '_'.join(map(str, x)), axis=1)
test_1 = test_1[['Id','feat_concat']]


dup_data_check = train_1.append(test_1)
del train_1
del test_1

## relevant data from date table
date_df_1 = date_df[['Id','starttime']]
dup_data_check = pd.merge(dup_data_check, date_df_1, on = ['Id'])
del date_df_1

##create flag for duplicate rows
dup_data_check = dup_data_check.sort_values(by = ['starttime','Id'], ascending = True)
dup_data_check['duplicate_row'] = (dup_data_check.feat_concat == dup_data_check.feat_concat.shift(-1)).astype(int)

dup_data_check = dup_data_check[['Id','duplicate_row']]


# In[70]:



### LOAD TRAINING DATA
train_numeric = pd.read_csv('train_numeric_bosch_csv.zip', compression='zip', header=0, sep=',', quotechar='"'
                            , usecols = train_cols)


train_numeric = pd.merge(train_numeric,date_df, on = 'Id')
# train_numeric = pd.merge(train_numeric,cat_train, on = 'Id')
train_numeric = pd.merge(train_numeric,cat_total, on = 'Id')
train_numeric = pd.merge(train_numeric,dup_data_check, on = 'Id')

## 2 way interaction
train_numeric['multi17_done']= train_numeric['L1_S24_F1723']*train_numeric['L1_S24_F1844']
train_numeric['multi20_done']= train_numeric['L3_S33_F3857']*train_numeric['L3_S29_F3327']
train_numeric['multi22_done']= train_numeric['L3_S33_F3857']*train_numeric['L3_S29_F3373']
train_numeric['multi31_done']= train_numeric['L3_S38_F3956']*train_numeric['L3_S38_F3952']
train_numeric['multi60_done']= train_numeric['L1_S24_F1632']*train_numeric['L3_S29_F3476']
train_numeric['multi88_done']= train_numeric['L0_S9_F180']*train_numeric['L3_S33_F3865']
train_numeric['multi90_done']= train_numeric['L0_S9_F180']*train_numeric['L3_S29_F3476']
train_numeric['multi92_done']= train_numeric['L3_S29_F3373']*train_numeric['L3_S30_F3709']
train_numeric['multi104_done']= train_numeric['L3_S33_F3865']*train_numeric['L3_S29_F3476']
train_numeric['multi_new_12']=train_numeric['L1_S24_F1723'] * train_numeric['L1_S24_F1844']
train_numeric['multi_new_15']=train_numeric['L1_S24_F1723'] * train_numeric['L2_S26_F3106']
train_numeric['multi_new_25']=train_numeric['L1_S24_F1844'] * train_numeric['L3_S29_F3407']
train_numeric['multi_new_32']=train_numeric['L1_S24_F1846'] * train_numeric['L2_S26_F3106']
train_numeric['multi_new_46']=train_numeric['L2_S26_F3106'] * train_numeric['L3_S29_F3407']
train_numeric['multi_new_47']=train_numeric['L2_S26_F3106'] * train_numeric['L3_S30_F3754']
train_numeric['multi_new_48']=train_numeric['L2_S26_F3106'] * train_numeric['L3_S32_F3850']
train_numeric['multi_new_52']=train_numeric['L3_S29_F3407'] * train_numeric['L3_S30_F3754']
train_numeric['multi_new_65']=train_numeric['L3_S38_F3952'] * train_numeric['L3_S38_F3960']
train_numeric['multi_new_66']=train_numeric['L3_S38_F3956'] * train_numeric['L3_S38_F3960']

###########new added later
train_numeric['multi_comb_1'] = train_numeric['L3_S30_F3554']*train_numeric['L3_S33_F3859']
train_numeric['multi_comb_2'] = train_numeric['L1_S24_F1844']*train_numeric['L1_S24_F1846']
train_numeric['multi_comb_3'] = train_numeric['L3_S30_F3524']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_4'] = train_numeric['L3_S29_F3412']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_5'] = train_numeric['L3_S29_F3407']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_6'] = train_numeric['L1_S24_F1632']*train_numeric['L1_S24_F1844']
train_numeric['multi_comb_7'] = train_numeric['L1_S24_F1846']*train_numeric['L3_S29_F3464']
train_numeric['multi_comb_8'] = train_numeric['L3_S30_F3799']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_9'] = train_numeric['L3_S30_F3774']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_10'] = train_numeric['L3_S29_F3430']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_11'] = train_numeric['L3_S30_F3744']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_12'] = train_numeric['L3_S29_F3360']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_13'] = train_numeric['L3_S30_F3519']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_14'] = train_numeric['L3_S30_F3794']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_15'] = train_numeric['L3_S30_F3784']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_16'] = train_numeric['L3_S30_F3759']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_17'] = train_numeric['L3_S30_F3544']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_18'] = train_numeric['L3_S29_F3436']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_19'] = train_numeric['L1_S24_F1609']*train_numeric['L1_S24_F1632']
train_numeric['multi_comb_20'] = train_numeric['L3_S29_F3330']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_21'] = train_numeric['L3_S30_F3509']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_22'] = train_numeric['L3_S29_F3424']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_23'] = train_numeric['L3_S29_F3348']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_24'] = train_numeric['L3_S30_F3804']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_25'] = train_numeric['L3_S30_F3554']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_26'] = train_numeric['L3_S30_F3829']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_27'] = train_numeric['L3_S29_F3339']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_28'] = train_numeric['L3_S29_F3455']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_29'] = train_numeric['L3_S29_F3395']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_30'] = train_numeric['L3_S30_F3764']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_31'] = train_numeric['L3_S29_F3327']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_32'] = train_numeric['L3_S29_F3354']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_33'] = train_numeric['L3_S29_F3373']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_34'] = train_numeric['L3_S30_F3809']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_35'] = train_numeric['L3_S29_F3433']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_36'] = train_numeric['L3_S30_F3529']*train_numeric['L3_S33_F3857']
train_numeric['multi_comb_37'] = train_numeric['L3_S30_F3819']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_38'] = train_numeric['L3_S29_F3452']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_39'] = train_numeric['L0_S13_F356']*train_numeric['L2_S26_F3073']
train_numeric['multi_comb_40'] = train_numeric['L1_S24_F1581']*train_numeric['L1_S24_F1609']
train_numeric['multi_comb_41'] = train_numeric['L3_S29_F3449']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_42'] = train_numeric['L3_S29_F3452']*train_numeric['L3_S33_F3857']
train_numeric['multi_comb_43'] = train_numeric['L3_S29_F3401']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_44'] = train_numeric['L3_S29_F3357']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_45'] = train_numeric['L3_S29_F3427']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_46'] = train_numeric['L3_S32_F3850']*train_numeric['L3_S35_F3894']
train_numeric['multi_comb_47'] = train_numeric['L3_S30_F3504']*train_numeric['L3_S33_F3857']
train_numeric['multi_comb_48'] = train_numeric['L3_S29_F3342']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_49'] = train_numeric['L0_S10_F249']*train_numeric['L3_S30_F3609']
train_numeric['multi_comb_50'] = train_numeric['L3_S30_F3769']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_51'] = train_numeric['L3_S29_F3467']*train_numeric['L3_S33_F3859']
train_numeric['multi_comb_52'] = train_numeric['L1_S24_F1667']*train_numeric['L1_S24_F1846']
train_numeric['multi_comb_53'] = train_numeric['L3_S29_F3318']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_54'] = train_numeric['L3_S29_F3376']*train_numeric['L3_S33_F3859']
train_numeric['multi_comb_55'] = train_numeric['L3_S30_F3754']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_56'] = train_numeric['L3_S29_F3388']*train_numeric['L3_S33_F3859']
train_numeric['multi_comb_57'] = train_numeric['L3_S30_F3514']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_58'] = train_numeric['L3_S29_F3439']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_59'] = train_numeric['L3_S29_F3336']*train_numeric['L3_S33_F3859']
train_numeric['multi_comb_60'] = train_numeric['L3_S29_F3336']*train_numeric['L3_S35_F3894']
train_numeric['multi_comb_61'] = train_numeric['L3_S29_F3345']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_62'] = train_numeric['L3_S29_F3449']*train_numeric['L3_S33_F3859']
train_numeric['multi_comb_63'] = train_numeric['L3_S29_F3333']*train_numeric['L3_S33_F3859']
train_numeric['multi_comb_64'] = train_numeric['L3_S29_F3455']*train_numeric['L3_S33_F3859']
train_numeric['multi_comb_65'] = train_numeric['L3_S29_F3315']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_66'] = train_numeric['L3_S29_F3370']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_67'] = train_numeric['L3_S29_F3351']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_68'] = train_numeric['L3_S29_F3388']*train_numeric['L3_S35_F3894']
train_numeric['multi_comb_69'] = train_numeric['L3_S29_F3458']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_70'] = train_numeric['L3_S30_F3499']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_71'] = train_numeric['L3_S29_F3382']*train_numeric['L3_S33_F3859']
train_numeric['multi_comb_72'] = train_numeric['L3_S29_F3360']*train_numeric['L3_S33_F3859']
train_numeric['multi_comb_73'] = train_numeric['L3_S29_F3342']*train_numeric['L3_S33_F3859']
train_numeric['multi_comb_74'] = train_numeric['L3_S29_F3333']*train_numeric['L3_S35_F3894']
train_numeric['multi_comb_75'] = train_numeric['L3_S29_F3421']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_76'] = train_numeric['L3_S29_F3464']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_77'] = train_numeric['L3_S29_F3382']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_78'] = train_numeric['L0_S4_F104']*train_numeric['L3_S29_F3318']
train_numeric['multi_comb_79'] = train_numeric['L3_S29_F3336']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_80'] = train_numeric['L3_S29_F3395']*train_numeric['L3_S35_F3894']
train_numeric['multi_comb_81'] = train_numeric['L3_S29_F3461']*train_numeric['L3_S33_F3859']
train_numeric['multi_comb_82'] = train_numeric['L3_S29_F3339']*train_numeric['L3_S33_F3859']
train_numeric['multi_comb_83'] = train_numeric['L3_S29_F3404']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_84'] = train_numeric['L3_S30_F3529']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_85'] = train_numeric['L3_S29_F3370']*train_numeric['L3_S33_F3859']
train_numeric['multi_comb_86'] = train_numeric['L1_S24_F1622']*train_numeric['L3_S30_F3744']
train_numeric['multi_comb_87'] = train_numeric['L3_S30_F3584']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_88'] = train_numeric['L3_S29_F3458']*train_numeric['L3_S29_F3479']
train_numeric['multi_comb_89'] = train_numeric['L3_S29_F3467']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_90'] = train_numeric['L3_S30_F3759']*train_numeric['L3_S35_F3894']
train_numeric['multi_comb_91'] = train_numeric['L3_S29_F3382']*train_numeric['L3_S35_F3894']
train_numeric['multi_comb_92'] = train_numeric['L0_S9_F170']*train_numeric['L3_S30_F3584']
train_numeric['multi_comb_93'] = train_numeric['L3_S32_F3850']*train_numeric['L3_S35_F3889']
train_numeric['multi_comb_94'] = train_numeric['L3_S30_F3749']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_95'] = train_numeric['L3_S29_F3421']*train_numeric['L3_S33_F3859']
train_numeric['multi_comb_96'] = train_numeric['L3_S30_F3504']*train_numeric['L3_S32_F3850']
train_numeric['multi_comb_97'] = train_numeric['L3_S29_F3376']*train_numeric['L3_S35_F3894']
train_numeric['multi_comb_98'] = train_numeric['L3_S30_F3494']*train_numeric['L3_S33_F3865']
train_numeric['multi_comb_99'] = train_numeric['L1_S24_F1753']*train_numeric['L1_S24_F1773']
train_numeric['multi_comb_100'] = train_numeric['L3_S29_F3324']*train_numeric['L3_S33_F3857']


# In[144]:

#define  MCC -- additional 10ct Oct

from sklearn.metrics import matthews_corrcoef
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import scipy.sparse

def mcc(tp, tn, fp, fn):
    sup = tp * tn - fp * fn
    inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if inf==0:
        return 0
    else:
        return sup / np.sqrt(inf)


def eval_mcc(y_true, y_prob, show=False):
    idx = np.argsort(y_prob)
    y_true_sort = y_true[idx]
    n = y_true.shape[0]
    nump = 1.0 * np.sum(y_true) # number of positive
    numn = n - nump # number of negative
    tp = nump
    tn = 0.0
    fp = numn
    fn = 0.0
    best_mcc = 0.0
    best_id = -1
    prev_proba = -1
    best_proba = -1
    mccs = np.zeros(n)
    for i in range(n):
        # all items with idx < i are predicted negative while others are predicted positive
        # only evaluate mcc when probability changes
        proba = y_prob[idx[i]]
        if proba != prev_proba:
            prev_proba = proba
            new_mcc = mcc(tp, tn, fp, fn)
            if new_mcc >= best_mcc:
                best_mcc = new_mcc
                best_id = i
                best_proba = proba
        mccs[i] = new_mcc
        if y_true_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0
    if show:
        y_pred = (y_prob >= best_proba).astype(int)
        score = matthews_corrcoef(y_true, y_pred)
        return best_proba, best_mcc
    else:
        return best_mcc

def mcc_eval_xgb(y_prob_xgb, xgtrain):
    y_true = xgtrain.get_label()
    best_proba, best_mcc = eval_mcc(y_true, y_prob_xgb,True)
    print ('Cutoff probability is %s and best MCC is %s ' %(best_proba,best_mcc))
    return best_proba

def mcc_eval_rf(y_prob_rf, y):
    y_true = y.as_matrix()
    best_proba, best_mcc = eval_mcc(y_true, y_prob_rf,True)
    print ('Cutoff probability is %s and best MCC is %s ' %(best_proba,best_mcc))
    return best_proba

train_numeric = pd.merge(train_numeric,cfold_info, how = 'left', on = ['Id'])


# In[168]:


#################################################################################################################################
############################################ -----------XGBOOST MODEL----------------##########################################
#################################################################################################################################
###creating model data
id_x_train = train_numeric['Id']

y_train = train_numeric['Response']


X_train = train_numeric.drop(['Id','Response'], axis = 1)


xgtrain   = xgb.DMatrix(X_train, label=y_train)

ROUNDS=200
params = {}

params["max_depth"] = 14
params["min_child_weight"] = 5
params["eta"] = 0.02
params["gamma"] = 0
params["objective"] = "binary:logistic"
params["subsample"] = 0.8
params["colsample_bytree"] = 0.8
params["colsample_bylevel"] = 0.9
params["eval_metric"] = "auc"
params["silent"] = 1
params["scale_pos_weight"] = 1
params["seed"] = 1729

plst      = list(params.items())
watchlist = [(xgtrain, 'train'),(xgtest, 'test')]

#XGB on single fold
model     = xgb.train(plst, xgtrain, ROUNDS, watchlist,early_stopping_rounds=30,verbose_eval = 10)
y_prob_xgb = model.predict(xgtrain)
best_proba_xgb = mcc_eval_xgb(y_prob_xgb, xgtrain)

#checking for accuracy
train_check_xgb = pd.DataFrame({'Id':id_x, 'y_actual':y, 'y_pred': y_prob_xgb})
train_check_xgb['y_final_xgb'] = train_check_xgb['y_pred'].map(lambda x: 0 if x < best_proba_xgb else 1)

train_check_xgb['flag'] = 'A'
train_check_xgb['flag'][(train_check_xgb['y_actual'] == 1) & (train_check_xgb['y_final_xgb'] == 0)] = 'FN'
train_check_xgb['flag'][(train_check_xgb['y_actual'] == 1) & (train_check_xgb['y_final_xgb'] == 1)] = 'TP'
train_check_xgb['flag'][(train_check_xgb['y_actual'] == 0) & (train_check_xgb['y_final_xgb'] == 0)] = 'TN'
train_check_xgb['flag'][(train_check_xgb['y_actual'] == 0) & (train_check_xgb['y_final_xgb'] == 1)] = 'FP'

tcheck = train_check_xgb.groupby(['flag']).size()
print tcheck

################# Checking feature importance
feat_importance = pd.DataFrame(model.get_fscore().items(),columns=['feature', 'value'])
feat_importance = feat_importance.sort_values(['value'], ascending = False)
print feat_importance


# In[23]:



## LOAD TEST DATA
test_numeric = pd.read_csv('test_numeric_bosch_csv.zip', compression='zip', header=0, sep=',', quotechar='"'
                           , usecols = test_cols)

test_numeric = pd.merge(test_numeric,date_df, on = 'Id')
# test_numeric = pd.merge(test_numeric,cat_test, on = 'Id')
test_numeric = pd.merge(test_numeric,cat_total, on = 'Id')
test_numeric = pd.merge(test_numeric,dup_data_check, on = 'Id')

## 2 way interaction
test_numeric['multi17_done']= test_numeric['L1_S24_F1723']*test_numeric['L1_S24_F1844']
test_numeric['multi20_done']= test_numeric['L3_S33_F3857']*test_numeric['L3_S29_F3327']
test_numeric['multi22_done']= test_numeric['L3_S33_F3857']*test_numeric['L3_S29_F3373']
test_numeric['multi31_done']= test_numeric['L3_S38_F3956']*test_numeric['L3_S38_F3952']
test_numeric['multi60_done']= test_numeric['L1_S24_F1632']*test_numeric['L3_S29_F3476']
test_numeric['multi88_done']= test_numeric['L0_S9_F180']*test_numeric['L3_S33_F3865']
test_numeric['multi90_done']= test_numeric['L0_S9_F180']*test_numeric['L3_S29_F3476']
test_numeric['multi92_done']= test_numeric['L3_S29_F3373']*test_numeric['L3_S30_F3709']
test_numeric['multi104_done']= test_numeric['L3_S33_F3865']*test_numeric['L3_S29_F3476']
test_numeric['multi_new_12']=test_numeric['L1_S24_F1723'] * test_numeric['L1_S24_F1844']
test_numeric['multi_new_15']=test_numeric['L1_S24_F1723'] * test_numeric['L2_S26_F3106']
test_numeric['multi_new_25']=test_numeric['L1_S24_F1844'] * test_numeric['L3_S29_F3407']
test_numeric['multi_new_32']=test_numeric['L1_S24_F1846'] * test_numeric['L2_S26_F3106']
test_numeric['multi_new_46']=test_numeric['L2_S26_F3106'] * test_numeric['L3_S29_F3407']
test_numeric['multi_new_47']=test_numeric['L2_S26_F3106'] * test_numeric['L3_S30_F3754']
test_numeric['multi_new_48']=test_numeric['L2_S26_F3106'] * test_numeric['L3_S32_F3850']
test_numeric['multi_new_52']=test_numeric['L3_S29_F3407'] * test_numeric['L3_S30_F3754']
test_numeric['multi_new_65']=test_numeric['L3_S38_F3952'] * test_numeric['L3_S38_F3960']
test_numeric['multi_new_66']=test_numeric['L3_S38_F3956'] * test_numeric['L3_S38_F3960']

######added later
test_numeric['multi_comb_1'] = test_numeric['L3_S30_F3554']*test_numeric['L3_S33_F3859']
test_numeric['multi_comb_2'] = test_numeric['L1_S24_F1844']*test_numeric['L1_S24_F1846']
test_numeric['multi_comb_3'] = test_numeric['L3_S30_F3524']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_4'] = test_numeric['L3_S29_F3412']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_5'] = test_numeric['L3_S29_F3407']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_6'] = test_numeric['L1_S24_F1632']*test_numeric['L1_S24_F1844']
test_numeric['multi_comb_7'] = test_numeric['L1_S24_F1846']*test_numeric['L3_S29_F3464']
test_numeric['multi_comb_8'] = test_numeric['L3_S30_F3799']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_9'] = test_numeric['L3_S30_F3774']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_10'] = test_numeric['L3_S29_F3430']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_11'] = test_numeric['L3_S30_F3744']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_12'] = test_numeric['L3_S29_F3360']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_13'] = test_numeric['L3_S30_F3519']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_14'] = test_numeric['L3_S30_F3794']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_15'] = test_numeric['L3_S30_F3784']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_16'] = test_numeric['L3_S30_F3759']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_17'] = test_numeric['L3_S30_F3544']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_18'] = test_numeric['L3_S29_F3436']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_19'] = test_numeric['L1_S24_F1609']*test_numeric['L1_S24_F1632']
test_numeric['multi_comb_20'] = test_numeric['L3_S29_F3330']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_21'] = test_numeric['L3_S30_F3509']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_22'] = test_numeric['L3_S29_F3424']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_23'] = test_numeric['L3_S29_F3348']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_24'] = test_numeric['L3_S30_F3804']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_25'] = test_numeric['L3_S30_F3554']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_26'] = test_numeric['L3_S30_F3829']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_27'] = test_numeric['L3_S29_F3339']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_28'] = test_numeric['L3_S29_F3455']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_29'] = test_numeric['L3_S29_F3395']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_30'] = test_numeric['L3_S30_F3764']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_31'] = test_numeric['L3_S29_F3327']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_32'] = test_numeric['L3_S29_F3354']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_33'] = test_numeric['L3_S29_F3373']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_34'] = test_numeric['L3_S30_F3809']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_35'] = test_numeric['L3_S29_F3433']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_36'] = test_numeric['L3_S30_F3529']*test_numeric['L3_S33_F3857']
test_numeric['multi_comb_37'] = test_numeric['L3_S30_F3819']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_38'] = test_numeric['L3_S29_F3452']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_39'] = test_numeric['L0_S13_F356']*test_numeric['L2_S26_F3073']
test_numeric['multi_comb_40'] = test_numeric['L1_S24_F1581']*test_numeric['L1_S24_F1609']
test_numeric['multi_comb_41'] = test_numeric['L3_S29_F3449']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_42'] = test_numeric['L3_S29_F3452']*test_numeric['L3_S33_F3857']
test_numeric['multi_comb_43'] = test_numeric['L3_S29_F3401']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_44'] = test_numeric['L3_S29_F3357']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_45'] = test_numeric['L3_S29_F3427']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_46'] = test_numeric['L3_S32_F3850']*test_numeric['L3_S35_F3894']
test_numeric['multi_comb_47'] = test_numeric['L3_S30_F3504']*test_numeric['L3_S33_F3857']
test_numeric['multi_comb_48'] = test_numeric['L3_S29_F3342']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_49'] = test_numeric['L0_S10_F249']*test_numeric['L3_S30_F3609']
test_numeric['multi_comb_50'] = test_numeric['L3_S30_F3769']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_51'] = test_numeric['L3_S29_F3467']*test_numeric['L3_S33_F3859']
test_numeric['multi_comb_52'] = test_numeric['L1_S24_F1667']*test_numeric['L1_S24_F1846']
test_numeric['multi_comb_53'] = test_numeric['L3_S29_F3318']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_54'] = test_numeric['L3_S29_F3376']*test_numeric['L3_S33_F3859']
test_numeric['multi_comb_55'] = test_numeric['L3_S30_F3754']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_56'] = test_numeric['L3_S29_F3388']*test_numeric['L3_S33_F3859']
test_numeric['multi_comb_57'] = test_numeric['L3_S30_F3514']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_58'] = test_numeric['L3_S29_F3439']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_59'] = test_numeric['L3_S29_F3336']*test_numeric['L3_S33_F3859']
test_numeric['multi_comb_60'] = test_numeric['L3_S29_F3336']*test_numeric['L3_S35_F3894']
test_numeric['multi_comb_61'] = test_numeric['L3_S29_F3345']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_62'] = test_numeric['L3_S29_F3449']*test_numeric['L3_S33_F3859']
test_numeric['multi_comb_63'] = test_numeric['L3_S29_F3333']*test_numeric['L3_S33_F3859']
test_numeric['multi_comb_64'] = test_numeric['L3_S29_F3455']*test_numeric['L3_S33_F3859']
test_numeric['multi_comb_65'] = test_numeric['L3_S29_F3315']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_66'] = test_numeric['L3_S29_F3370']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_67'] = test_numeric['L3_S29_F3351']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_68'] = test_numeric['L3_S29_F3388']*test_numeric['L3_S35_F3894']
test_numeric['multi_comb_69'] = test_numeric['L3_S29_F3458']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_70'] = test_numeric['L3_S30_F3499']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_71'] = test_numeric['L3_S29_F3382']*test_numeric['L3_S33_F3859']
test_numeric['multi_comb_72'] = test_numeric['L3_S29_F3360']*test_numeric['L3_S33_F3859']
test_numeric['multi_comb_73'] = test_numeric['L3_S29_F3342']*test_numeric['L3_S33_F3859']
test_numeric['multi_comb_74'] = test_numeric['L3_S29_F3333']*test_numeric['L3_S35_F3894']
test_numeric['multi_comb_75'] = test_numeric['L3_S29_F3421']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_76'] = test_numeric['L3_S29_F3464']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_77'] = test_numeric['L3_S29_F3382']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_78'] = test_numeric['L0_S4_F104']*test_numeric['L3_S29_F3318']
test_numeric['multi_comb_79'] = test_numeric['L3_S29_F3336']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_80'] = test_numeric['L3_S29_F3395']*test_numeric['L3_S35_F3894']
test_numeric['multi_comb_81'] = test_numeric['L3_S29_F3461']*test_numeric['L3_S33_F3859']
test_numeric['multi_comb_82'] = test_numeric['L3_S29_F3339']*test_numeric['L3_S33_F3859']
test_numeric['multi_comb_83'] = test_numeric['L3_S29_F3404']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_84'] = test_numeric['L3_S30_F3529']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_85'] = test_numeric['L3_S29_F3370']*test_numeric['L3_S33_F3859']
test_numeric['multi_comb_86'] = test_numeric['L1_S24_F1622']*test_numeric['L3_S30_F3744']
test_numeric['multi_comb_87'] = test_numeric['L3_S30_F3584']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_88'] = test_numeric['L3_S29_F3458']*test_numeric['L3_S29_F3479']
test_numeric['multi_comb_89'] = test_numeric['L3_S29_F3467']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_90'] = test_numeric['L3_S30_F3759']*test_numeric['L3_S35_F3894']
test_numeric['multi_comb_91'] = test_numeric['L3_S29_F3382']*test_numeric['L3_S35_F3894']
test_numeric['multi_comb_92'] = test_numeric['L0_S9_F170']*test_numeric['L3_S30_F3584']
test_numeric['multi_comb_93'] = test_numeric['L3_S32_F3850']*test_numeric['L3_S35_F3889']
test_numeric['multi_comb_94'] = test_numeric['L3_S30_F3749']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_95'] = test_numeric['L3_S29_F3421']*test_numeric['L3_S33_F3859']
test_numeric['multi_comb_96'] = test_numeric['L3_S30_F3504']*test_numeric['L3_S32_F3850']
test_numeric['multi_comb_97'] = test_numeric['L3_S29_F3376']*test_numeric['L3_S35_F3894']
test_numeric['multi_comb_98'] = test_numeric['L3_S30_F3494']*test_numeric['L3_S33_F3865']
test_numeric['multi_comb_99'] = test_numeric['L1_S24_F1753']*test_numeric['L1_S24_F1773']
test_numeric['multi_comb_100'] = test_numeric['L3_S29_F3324']*test_numeric['L3_S33_F3857']


id_test = test_numeric['Id']
test_numeric = test_numeric.drop(['Id'], axis = 1)  

##### MODEL STEP 3: predict on actual test data using XGB
xgb_test = xgb.DMatrix(test_numeric)   
preds_test = model.predict(xgb_test)
del xgb_test
submission_xgb = pd.DataFrame({'Id':id_test, 'pred_response': preds_test})
submission_xgb['Response'] = submission_xgb['pred_response'].map(lambda x: 0 if x < best_proba_xgb else 1)
submission_xgb = submission_xgb.drop(['pred_response'], axis = 1)
submission_xgb.to_csv('submission_bosch_06nov.csv',index = False)


