
# coding: utf-8

# In[1]:

################################ DATE PROCESSING ##################################
import time
import pandas as pd
import numpy as np
import xgboost as xgb


#steps to pick only de-duplicate columns    
date_sample = pd.read_csv('train_date_bosch_csv.zip', compression='zip', header=0, sep=',', quotechar='"',nrows = 12000)
date_new = date_sample.T.drop_duplicates().T
date_cols = date_new.columns.values.tolist()
del date_sample
del date_new


# In[3]:


#### traininng date data processing
start_time = time.time()

#multiple stations
unique_stations = ['L0_S0_D1', 'L0_S1_D26', 'L0_S2_D34', 'L0_S3_D70','L0_S4_KEY','L0_S5_D115', 'L0_S6_D120', 'L0_S7_D137'
                   , 'L0_S8_D145','L0_S9_D152', 'L0_S10_D216', 'L0_S11_D280', 'L0_S12_D331',
           'L0_S13_D355', 'L0_S14_D360', 'L0_S15_D395', 'L0_S16_D423','L0_S17_D432', 'L0_S18_D437', 'L0_S19_D454'
                   , 'L0_S20_D462','L0_S21_KEY',
           'L0_S22_D543', 'L0_S23_D617','L1_S24_KEY','L1_S25_KEY','L2_S26_D3037', 'L2_S27_D3130',
           'L2_S28_D3223', 'L3_S29_KEY', 'L3_S30_KEY', 'L3_S31_D3836', 'L3_S32_D3852', 'L3_S33_D3856',
           'L3_S34_D3875', 'L3_S35_KEY', 'L3_S36_D3919', 'L3_S37_D3942', 'L3_S38_D3953',
           'L3_S39_D3966', 'L3_S40_D3981', 'L3_S41_D3997', 'L3_S43_D4062',
           'L3_S44_D4101', 'L3_S45_D4125', 'L3_S47_D4140', 'L3_S48_D4194',
           'L3_S49_D4208', 'L3_S50_D4242', 'L3_S51_D4255']


tr = pd.read_csv('train_date_bosch_csv.zip', compression='zip', header=0, sep=',', quotechar='"'
                 , usecols = date_cols)
             
tr_cols = [col for col in tr if '_' in col]
for col in tr_cols:
    tr['starttime'] = tr[tr_cols].min(axis = 1)

# ------ check for overall presence
val_cols = [col for col in tr if '_S' in col]
for col in val_cols:
    tr[col] = tr[col] > 0

tr[val_cols] = tr[val_cols].astype(int)

####----------------------- check for production line presence ---------------    
#check for Product Line 0 presence
val_l0_cols = [col for col in tr if 'L0' in col]
for col in val_l0_cols:
    tr['L0_KEY'] = tr[val_l0_cols].max(axis = 1)

#check for Product Line 1 presence
val_l1_cols = [col for col in tr if 'L1' in col]
for col in val_l1_cols:
    tr['L1_KEY'] = tr[val_l1_cols].max(axis = 1)
    

#check for Product Line 2 presence
val_l2_cols = [col for col in tr if 'L2' in col]
for col in val_l2_cols:
    tr['L2_KEY'] = tr[val_l2_cols].max(axis = 1)
    
    
#check for Product Line 3 presence
val_l3_cols = [col for col in tr if 'L3' in col]
for col in val_l3_cols:
    tr['L3_KEY'] = tr[val_l3_cols].max(axis = 1)

#CREATE A PRODUCT LINE KEY
tr['prod_line_key'] = tr.L0_KEY.astype(str) + tr.L1_KEY.astype(str)+ tr.L2_KEY.astype(str)+ tr.L3_KEY.astype(str)

tr = tr.drop(['L0_KEY','L1_KEY','L2_KEY','L3_KEY'],axis =1)



##### S4,S21,S24,S29,S30,S35 had different time stamps and different parts had different features checked on this stations
#### hence this unique way of getting presence of parts on this stations.

val_s4_cols = [col for col in tr if '_S4_' in col]#check for S4 presence
for col in val_s4_cols:
    tr['L0_S4_KEY'] = tr[val_s4_cols].max(axis = 1)

val_s21_cols = [col for col in tr if '_S21_' in col]#check for S21 presence
for col in val_s21_cols:
    tr['L0_S21_KEY'] = tr[val_s21_cols].max(axis = 1)

val_s24_cols = [col for col in tr if '_S24_' in col]#check for S24 presence
for col in val_s24_cols:
    tr['L1_S24_KEY'] = tr[val_s24_cols].max(axis = 1) #this key to be used at station 24 presence key elsewhere also

val_s25_cols = [col for col in tr if '_S25_' in col]#check for S25 presence
for col in val_s25_cols:
    tr['L1_S25_KEY'] = tr[val_s25_cols].max(axis = 1)#this key to be used at station 25 presence key elsewhere also

val_s29_cols = [col for col in tr if '_S29_' in col]#check for S29 presence
for col in val_s29_cols:
    tr['L3_S29_KEY'] = tr[val_s29_cols].max(axis = 1)

val_s30_cols = [col for col in tr if '_S30_' in col]#check for S30 presence
for col in val_s30_cols:
    tr['L3_S30_KEY'] = tr[val_s30_cols].max(axis = 1)

val_s32_cols = [col for col in tr if '_S32_' in col]#check for S32 presence
for col in val_s32_cols:
    tr['L3_S32_KEY'] = tr[val_s32_cols].max(axis = 1)
    
val_s33_cols = [col for col in tr if '_S33_' in col]#check for S33 presence
for col in val_s33_cols:
    tr['L3_S33_KEY'] = tr[val_s33_cols].max(axis = 1)
    
val_s34_cols = [col for col in tr if '_S34_' in col]#check for S34 presence
for col in val_s34_cols:
    tr['L3_S34_KEY'] = tr[val_s34_cols].max(axis = 1)    
       
val_s35_cols = [col for col in tr if '_S35_' in col]#check for S35 presence
for col in val_s35_cols:
    tr['L3_S35_KEY'] = tr[val_s35_cols].max(axis = 1)

#multiple stations
unique_stations = ['L0_S0_D1', 'L0_S1_D26', 'L0_S2_D34', 'L0_S3_D70','L0_S4_KEY','L0_S5_D115', 'L0_S6_D120', 'L0_S7_D137', 'L0_S8_D145','L0_S9_D152', 'L0_S10_D216', 'L0_S11_D280', 'L0_S12_D331',
           'L0_S13_D355', 'L0_S14_D360', 'L0_S15_D395', 'L0_S16_D423','L0_S17_D432', 'L0_S18_D437', 'L0_S19_D454', 'L0_S20_D462','L0_S21_KEY',
           'L0_S22_D543', 'L0_S23_D617','L1_S24_KEY','L1_S25_KEY','L2_S26_D3037', 'L2_S27_D3130',
           'L2_S28_D3223', 'L3_S29_KEY', 'L3_S30_KEY', 'L3_S31_D3836', 'L3_S32_D3852', 'L3_S33_D3856',
           'L3_S34_D3875', 'L3_S35_KEY', 'L3_S36_D3919', 'L3_S37_D3942', 'L3_S38_D3953',
           'L3_S39_D3966', 'L3_S40_D3981', 'L3_S41_D3997', 'L3_S43_D4062',
           'L3_S44_D4101', 'L3_S45_D4125', 'L3_S47_D4140', 'L3_S48_D4194',
           'L3_S49_D4208', 'L3_S50_D4242', 'L3_S51_D4255']


    
tr['station_line_key'] = tr[['L0_S0_D1', 'L0_S1_D26', 'L0_S2_D34', 'L0_S3_D70','L0_S4_KEY','L0_S5_D115', 'L0_S6_D120', 'L0_S7_D137', 'L0_S8_D145','L0_S9_D152', 'L0_S10_D216', 'L0_S11_D280', 'L0_S12_D331',
       'L0_S13_D355', 'L0_S14_D360', 'L0_S15_D395', 'L0_S16_D423','L0_S17_D432', 'L0_S18_D437', 'L0_S19_D454', 'L0_S20_D462','L0_S21_KEY',
       'L0_S22_D543', 'L0_S23_D617','L1_S24_KEY','L1_S25_KEY','L2_S26_D3037', 'L2_S27_D3130',
       'L2_S28_D3223', 'L3_S29_KEY', 'L3_S30_KEY', 'L3_S31_D3836', 'L3_S32_D3852', 'L3_S33_D3856',
       'L3_S34_D3875', 'L3_S35_KEY', 'L3_S36_D3919', 'L3_S37_D3942', 'L3_S38_D3953',
       'L3_S39_D3966', 'L3_S40_D3981', 'L3_S41_D3997', 'L3_S43_D4062',
       'L3_S44_D4101', 'L3_S45_D4125', 'L3_S47_D4140', 'L3_S48_D4194',
       'L3_S49_D4208', 'L3_S50_D4242', 'L3_S51_D4255']].apply(lambda x: '_'.join(map(str, x)), axis=1)

tr = tr.drop(['L0_S0_D1', 'L0_S1_D26', 'L0_S2_D34', 'L0_S3_D70','L0_S4_KEY','L0_S5_D115', 'L0_S6_D120', 'L0_S7_D137', 'L0_S8_D145','L0_S9_D152', 'L0_S10_D216', 'L0_S11_D280', 'L0_S12_D331',
       'L0_S13_D355', 'L0_S14_D360', 'L0_S15_D395', 'L0_S16_D423','L0_S17_D432', 'L0_S18_D437', 'L0_S19_D454', 'L0_S20_D462','L0_S21_KEY',
       'L0_S22_D543', 'L0_S23_D617','L1_S24_KEY','L1_S25_KEY','L2_S26_D3037', 'L2_S27_D3130',
       'L2_S28_D3223', 'L3_S29_KEY', 'L3_S30_KEY', 'L3_S31_D3836', 'L3_S32_D3852', 'L3_S33_D3856',
       'L3_S34_D3875', 'L3_S35_KEY', 'L3_S36_D3919', 'L3_S37_D3942', 'L3_S38_D3953',
       'L3_S39_D3966', 'L3_S40_D3981', 'L3_S41_D3997', 'L3_S43_D4062',
       'L3_S44_D4101', 'L3_S45_D4125', 'L3_S47_D4140', 'L3_S48_D4194',
       'L3_S49_D4208', 'L3_S50_D4242', 'L3_S51_D4255'], axis = 1)

    
##### all products either start from L0 (S00 to S11 OR S12 to S243) Or L1 (S24/S25) or L2; Almost all part go through L3
##### in L3 either they go through S29 to S38 OR S38 to later
l0_first_half_col = ['L0_S0_D1', 'L0_S1_D26', 'L0_S2_D34', 'L0_S3_D70','L0_S4_KEY','L0_S5_D115', 'L0_S6_D120'
                     , 'L0_S7_D137', 'L0_S8_D145','L0_S9_D152', 'L0_S10_D216', 'L0_S11_D280']
for col in l0_first_half_col:
    tr['L0_FIRST_HALF_KEY'] = tr[l0_first_half_col].max(axis = 1)

l0_second_half_col = ['L0_S12_D331','L0_S13_D355', 'L0_S14_D360', 'L0_S15_D395', 'L0_S16_D423','L0_S17_D432', 
                      'L0_S18_D437', 'L0_S19_D454', 'L0_S20_D462','L0_S21_KEY','L0_S22_D543', 'L0_S23_D617']
for col in l0_second_half_col:
    tr['L0_SECOND_HALF_KEY'] = tr[l0_second_half_col].max(axis = 1)

l3_first_half_col = ['L3_S29_KEY', 'L3_S30_KEY', 'L3_S31_D3836', 'L3_S32_D3852', 'L3_S33_D3856',
       'L3_S34_D3875', 'L3_S35_KEY', 'L3_S36_D3919', 'L3_S37_D3942']
for col in l3_first_half_col:
    tr['L3_FIRST_HALF_KEY'] = tr[l3_first_half_col].max(axis = 1)

l3_second_half_col = ['L3_S38_D3953','L3_S39_D3966', 'L3_S40_D3981', 'L3_S41_D3997', 'L3_S43_D4062',
       'L3_S44_D4101', 'L3_S45_D4125', 'L3_S47_D4140', 'L3_S48_D4194',
       'L3_S49_D4208', 'L3_S50_D4242', 'L3_S51_D4255']
for col in l3_second_half_col:
    tr['L3_SECOND_HALF_KEY'] = tr[l3_second_half_col].max(axis = 1)

    
tr = tr[['Id','starttime','L0_FIRST_HALF_KEY','L0_SECOND_HALF_KEY',
             'L3_FIRST_HALF_KEY','L3_SECOND_HALF_KEY','L3_S32_KEY','L3_S33_KEY','L3_S34_KEY']]

print ('-- done in %s seconds' %(time.time()-start_time))


tr.to_csv('training_date_processed.csv', index = False)


# In[4]:


#### test date data processing
start_time = time.time()

CHUNKSIZE = 50000

#multiple stations
unique_stations = ['L0_S0_D1', 'L0_S1_D26', 'L0_S2_D34', 'L0_S3_D70','L0_S4_KEY','L0_S5_D115', 'L0_S6_D120', 'L0_S7_D137', 'L0_S8_D145','L0_S9_D152', 'L0_S10_D216', 'L0_S11_D280', 'L0_S12_D331',
           'L0_S13_D355', 'L0_S14_D360', 'L0_S15_D395', 'L0_S16_D423','L0_S17_D432', 'L0_S18_D437', 'L0_S19_D454', 'L0_S20_D462','L0_S21_KEY',
           'L0_S22_D543', 'L0_S23_D617','L1_S24_KEY','L1_S25_KEY','L2_S26_D3037', 'L2_S27_D3130',
           'L2_S28_D3223', 'L3_S29_KEY', 'L3_S30_KEY', 'L3_S31_D3836', 'L3_S32_D3852', 'L3_S33_D3856',
           'L3_S34_D3875', 'L3_S35_KEY', 'L3_S36_D3919', 'L3_S37_D3942', 'L3_S38_D3953',
           'L3_S39_D3966', 'L3_S40_D3981', 'L3_S41_D3997', 'L3_S43_D4062',
           'L3_S44_D4101', 'L3_S45_D4125', 'L3_S47_D4140', 'L3_S48_D4194',
           'L3_S49_D4208', 'L3_S50_D4242', 'L3_S51_D4255']


te = pd.read_csv('test_date_bosch_csv.zip', compression='zip', header=0, sep=',', quotechar='"'
                 ,usecols = date_cols)
             
te_cols = [col for col in te if '_' in col]
for col in te_cols:
    te['starttime'] = te[te_cols].min(axis = 1)

val_cols = [col for col in te if '_S' in col] #check for overall presence
for col in val_cols:
    te[col] = te[col] > 0

te[val_cols] = te[val_cols].astype(int)


####----------------------- check for production line presence ---------------    
#check for Product Line 0 presence
val_l0_cols = [col for col in te if 'L0' in col]
for col in val_l0_cols:
    te['L0_KEY'] = te[val_l0_cols].max(axis = 1)

#check for Product Line 1 presence
val_l1_cols = [col for col in te if 'L1' in col]
for col in val_l1_cols:
    te['L1_KEY'] = te[val_l1_cols].max(axis = 1)
    

#check for Product Line 2 presence
val_l2_cols = [col for col in te if 'L2' in col]
for col in val_l2_cols:
    te['L2_KEY'] = te[val_l2_cols].max(axis = 1)
    
    
#check for Product Line 3 presence
val_l3_cols = [col for col in te if 'L3' in col]
for col in val_l3_cols:
    te['L3_KEY'] = te[val_l3_cols].max(axis = 1)

#CREATE A PRODUCT LINE KEY
te['prod_line_key'] = te.L0_KEY.astype(str) + te.L1_KEY.astype(str)+ te.L2_KEY.astype(str)+ te.L3_KEY.astype(str)

te = te.drop(['L0_KEY','L1_KEY','L2_KEY','L3_KEY'],axis =1)



##### S4,S21,S24,S29,S30,S35 had different time stamps and different parts had different features checked on this stations
#### hence this unique way of getting presence of parts on this stations.

val_s4_cols = [col for col in te if '_S4_' in col]#check for S4 presence
for col in val_s4_cols:
    te['L0_S4_KEY'] = te[val_s4_cols].max(axis = 1)

val_s21_cols = [col for col in te if '_S21_' in col]#check for S21 presence
for col in val_s21_cols:
    te['L0_S21_KEY'] = te[val_s21_cols].max(axis = 1)

val_s24_cols = [col for col in te if '_S24_' in col]#check for S24 presence
for col in val_s24_cols:
    te['L1_S24_KEY'] = te[val_s24_cols].max(axis = 1) #this key to be used at station 24 presence key elsewhere also

val_s25_cols = [col for col in te if '_S25_' in col]#check for S25 presence
for col in val_s25_cols:
    te['L1_S25_KEY'] = te[val_s25_cols].max(axis = 1)#this key to be used at station 25 presence key elsewhere also

val_s29_cols = [col for col in te if '_S29_' in col]#check for S29 presence
for col in val_s29_cols:
    te['L3_S29_KEY'] = te[val_s29_cols].max(axis = 1)

val_s30_cols = [col for col in te if '_S30_' in col]#check for S30 presence
for col in val_s30_cols:
    te['L3_S30_KEY'] = te[val_s30_cols].max(axis = 1)

val_s32_cols = [col for col in te if '_S32_' in col]#check for S32 presence
for col in val_s32_cols:
    te['L3_S32_KEY'] = te[val_s32_cols].max(axis = 1)
    
val_s33_cols = [col for col in te if '_S33_' in col]#check for S33 presence
for col in val_s33_cols:
    te['L3_S33_KEY'] = te[val_s33_cols].max(axis = 1)
    
val_s34_cols = [col for col in te if '_S34_' in col]#check for S34 presence
for col in val_s34_cols:
    te['L3_S34_KEY'] = te[val_s34_cols].max(axis = 1)    
    
val_s35_cols = [col for col in te if '_S35_' in col]#check for S35 presence
for col in val_s35_cols:
    te['L3_S35_KEY'] = te[val_s35_cols].max(axis = 1)
    
    
te['station_line_key'] = te[['L0_S0_D1', 'L0_S1_D26', 'L0_S2_D34', 'L0_S3_D70','L0_S4_KEY','L0_S5_D115', 'L0_S6_D120', 'L0_S7_D137', 'L0_S8_D145','L0_S9_D152', 'L0_S10_D216', 'L0_S11_D280', 'L0_S12_D331',
       'L0_S13_D355', 'L0_S14_D360', 'L0_S15_D395', 'L0_S16_D423','L0_S17_D432', 'L0_S18_D437', 'L0_S19_D454', 'L0_S20_D462','L0_S21_KEY',
       'L0_S22_D543', 'L0_S23_D617','L1_S24_KEY','L1_S25_KEY','L2_S26_D3037', 'L2_S27_D3130',
       'L2_S28_D3223', 'L3_S29_KEY', 'L3_S30_KEY', 'L3_S31_D3836', 'L3_S32_D3852', 'L3_S33_D3856',
       'L3_S34_D3875', 'L3_S35_KEY', 'L3_S36_D3919', 'L3_S37_D3942', 'L3_S38_D3953',
       'L3_S39_D3966', 'L3_S40_D3981', 'L3_S41_D3997', 'L3_S43_D4062',
       'L3_S44_D4101', 'L3_S45_D4125', 'L3_S47_D4140', 'L3_S48_D4194',
       'L3_S49_D4208', 'L3_S50_D4242', 'L3_S51_D4255']].apply(lambda x: '_'.join(map(str, x)), axis=1)

te = te.drop(['L0_S0_D1', 'L0_S1_D26', 'L0_S2_D34', 'L0_S3_D70','L0_S4_KEY','L0_S5_D115', 'L0_S6_D120', 'L0_S7_D137', 'L0_S8_D145','L0_S9_D152', 'L0_S10_D216', 'L0_S11_D280', 'L0_S12_D331',
       'L0_S13_D355', 'L0_S14_D360', 'L0_S15_D395', 'L0_S16_D423','L0_S17_D432', 'L0_S18_D437', 'L0_S19_D454', 'L0_S20_D462','L0_S21_KEY',
       'L0_S22_D543', 'L0_S23_D617','L1_S24_KEY','L1_S25_KEY','L2_S26_D3037', 'L2_S27_D3130',
       'L2_S28_D3223', 'L3_S29_KEY', 'L3_S30_KEY', 'L3_S31_D3836', 'L3_S32_D3852', 'L3_S33_D3856',
       'L3_S34_D3875', 'L3_S35_KEY', 'L3_S36_D3919', 'L3_S37_D3942', 'L3_S38_D3953',
       'L3_S39_D3966', 'L3_S40_D3981', 'L3_S41_D3997', 'L3_S43_D4062',
       'L3_S44_D4101', 'L3_S45_D4125', 'L3_S47_D4140', 'L3_S48_D4194',
       'L3_S49_D4208', 'L3_S50_D4242', 'L3_S51_D4255'], axis = 1)


##### all products either start from L0 (S00 to S11 OR S12 to S243) Or L1 (S24/S25) or L2; Almost all part go through L3
##### in L3 either they go through S29 to S38 OR S38 to later
l0_first_half_col = ['L0_S0_D1', 'L0_S1_D26', 'L0_S2_D34', 'L0_S3_D70','L0_S4_KEY','L0_S5_D115', 'L0_S6_D120'
                     , 'L0_S7_D137', 'L0_S8_D145','L0_S9_D152', 'L0_S10_D216', 'L0_S11_D280']
for col in l0_first_half_col:
    te['L0_FIRST_HALF_KEY'] = te[l0_first_half_col].max(axis = 1)

l0_second_half_col = ['L0_S12_D331','L0_S13_D355', 'L0_S14_D360', 'L0_S15_D395', 'L0_S16_D423','L0_S17_D432', 
                      'L0_S18_D437', 'L0_S19_D454', 'L0_S20_D462','L0_S21_KEY','L0_S22_D543', 'L0_S23_D617']
for col in l0_second_half_col:
    te['L0_SECOND_HALF_KEY'] = te[l0_second_half_col].max(axis = 1)

l3_first_half_col = ['L3_S29_KEY', 'L3_S30_KEY', 'L3_S31_D3836', 'L3_S32_D3852', 'L3_S33_D3856',
       'L3_S34_D3875', 'L3_S35_KEY', 'L3_S36_D3919', 'L3_S37_D3942']
for col in l3_first_half_col:
    te['L3_FIRST_HALF_KEY'] = te[l3_first_half_col].max(axis = 1)

l3_second_half_col = ['L3_S38_D3953','L3_S39_D3966', 'L3_S40_D3981', 'L3_S41_D3997', 'L3_S43_D4062',
       'L3_S44_D4101', 'L3_S45_D4125', 'L3_S47_D4140', 'L3_S48_D4194',
       'L3_S49_D4208', 'L3_S50_D4242', 'L3_S51_D4255']
for col in l3_second_half_col:
    te['L3_SECOND_HALF_KEY'] = te[l3_second_half_col].max(axis = 1)


te = te[['Id','starttime','L0_FIRST_HALF_KEY','L0_SECOND_HALF_KEY',
             'L3_FIRST_HALF_KEY','L3_SECOND_HALF_KEY','L3_S32_KEY','L3_S33_KEY','L3_S34_KEY']]

print ('-- done in %s seconds' %(time.time()-start_time))


te.to_csv('test_date_processed.csv', index = False)

