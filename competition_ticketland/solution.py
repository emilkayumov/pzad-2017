# coding: utf-8

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb


click_no_impr = pd.read_csv('../input/clicks_no_impressions.сsv')
impr = pd.read_csv('../input/impressions.сsv')

client_data = pd.read_csv('../input/client_data.сsv')
show_data = pd.read_csv('../input/show_data.сsv')
show_images = pd.read_csv('../input/show_images.сsv')
show_rating = pd.read_csv('../input/show_rating.сsv')

show_data = show_data.groupby('id_show').first()


is_valid = False

if is_valid:
    impr.event_datetime_m = impr.event_datetime_m.apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    test = impr.loc[impr.event_datetime_m >= datetime.datetime(2017, 3, 10)].copy()
    impr = impr.loc[impr.event_datetime_m < datetime.datetime(2017, 3, 10)].copy()
    click_no_impr = click_no_impr.loc[click_no_impr.event_datetime_m < '2017-03-10 00:00:00'].copy()
    impr.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
else:
    test = pd.read_csv('../input/test.csv')


# feature extraction 
show_clicks = click_no_impr.groupby('id_show').size()
show_clicks = show_clicks.reset_index().rename(columns={0:'num_clicks'})
show_clicks['ratio_clicks'] = show_clicks.num_clicks / show_clicks.num_clicks.sum()
show_clicks['rank_clicks'] = show_clicks.num_clicks.rank(ascending=False)
show_clicks['rank_clicks'] = show_clicks.rank_clicks / show_clicks.rank_clicks.max()


impr['ratio_clicks'] = impr.id_show.map(show_clicks.ratio_clicks).fillna(show_clicks.ratio_clicks.mean())
impr['rank_clicks'] = impr.id_show.map(show_clicks.rank_clicks).fillna(show_clicks.rank_clicks.mean())
test['ratio_clicks'] = test.id_show.map(show_clicks.ratio_clicks).fillna(show_clicks.ratio_clicks.mean())
test['rank_clicks'] = test.id_show.map(show_clicks.rank_clicks).fillna(show_clicks.rank_clicks.mean())

show_clicks = impr.groupby('id_show').size()
show_clicks = show_clicks.reset_index().rename(columns={0:'num_clicks'})
show_clicks['ratio_clicks'] = show_clicks.num_clicks / show_clicks.num_clicks.sum()
show_clicks['rank_clicks'] = show_clicks.num_clicks.rank(ascending=False)
show_clicks['rank_clicks'] = show_clicks.rank_clicks / show_clicks.rank_clicks.max()

impr['ratio_clicks_impr'] = impr.id_show.map(show_clicks.ratio_clicks).fillna(show_clicks.ratio_clicks.mean())
impr['rank_clicks_impr'] = impr.id_show.map(show_clicks.rank_clicks).fillna(show_clicks.rank_clicks.mean())
test['ratio_clicks_impr'] = test.id_show.map(show_clicks.ratio_clicks).fillna(show_clicks.ratio_clicks.mean())
test['rank_clicks_impr'] = test.id_show.map(show_clicks.rank_clicks).fillna(show_clicks.rank_clicks.mean())


impr['inv_rank'] = 1.0 / impr['rank']
test['inv_rank'] = 1.0 / test['rank']

client_data.set_index('id_user', inplace=True)
client_data.sex.fillna('unknown', inplace=True)

show_data['parent_genre_id'] = show_data.parent_genre_id.apply(
    lambda x: -1 if type(x) is float else x.replace('[', '').replace(']', '').split(' ')[0]
)

show_data.IdBuilding.fillna(-1, inplace=True)
show_data[show_data.parent_genre_id == ''] = -1


impr['id_organizer'] = impr.id_show.map(show_data.organizer_id).astype(int)
impr['id_show_building'] = impr.id_show.map(show_data.IdBuilding).astype(int)
impr['id_parent_genre'] = impr.id_show.map(show_data.parent_genre_id).astype(int)
impr['show_age_category'] = impr.id_show.map(show_data.age_category)
impr['show_duration'] = impr.id_show.map(show_data.duration)

impr['show_maxprice'] = impr.id_show.map(show_data.show_maxprice)
impr['show_minprice'] = impr.id_show.map(show_data.show_minprice)
impr['show_meanprice'] = impr.id_show.map(show_data.show_meanprice)
impr['show_stdprice'] = impr.id_show.map(show_data.show_stdprice)


test['id_organizer'] = test.id_show.map(show_data.organizer_id).astype(int)
test['id_show_building'] = test.id_show.map(show_data.IdBuilding).astype(int)
test['id_parent_genre'] = test.id_show.map(show_data.parent_genre_id).astype(int)
test['show_age_category'] = test.id_show.map(show_data.age_category)
test['show_duration'] = test.id_show.map(show_data.duration)

test['show_maxprice'] = test.id_show.map(show_data.show_maxprice)
test['show_minprice'] = test.id_show.map(show_data.show_minprice)
test['show_meanprice'] = test.id_show.map(show_data.show_meanprice)
test['show_stdprice'] = test.id_show.map(show_data.show_stdprice)

impr['id_client_sex'] = impr.id_user.map(client_data.sex)
impr['client_age'] = impr.id_user.map(client_data.age)

test['id_client_sex'] = test.id_user.map(client_data.sex)
test['client_age'] = test.id_user.map(client_data.age)


impr['show_rating'] = impr.id_show.map(show_rating.groupby('id_show')['rating'].mean())
impr['show_num_rating'] = impr.id_show.map(show_rating.groupby('id_show').size())
impr['show_num_review'] = impr.id_show.map(show_rating.groupby('id_show')['review_count'].max())

test['show_rating'] = test.id_show.map(show_rating.groupby('id_show')['rating'].mean())
test['show_num_rating'] = test.id_show.map(show_rating.groupby('id_show').size())
test['show_num_review'] = test.id_show.map(show_rating.groupby('id_show')['review_count'].max())


impr['show_x_dt'] = impr.event_datetime_m.astype(str)+'_'+impr.id_show.astype(str)
impr['user_x_dt'] = impr.event_datetime_m.astype(str)+'_'+impr.id_user.astype(str)
impr['user_x_show_x_dt'] = impr.event_datetime_m.astype(str)+'_'+impr.id_user.astype(str)+'_'+impr.id_show.astype(str)

test['show_x_dt'] = test.event_datetime_m.astype(str)+'_'+test.id_show.astype(str)
test['user_x_dt'] = test.event_datetime_m.astype(str)+'_'+test.id_user.astype(str)
test['user_x_show_x_dt'] = test.event_datetime_m.astype(str)+'_'+test.id_user.astype(str)+'_'+test.id_show.astype(str)

impr['show_x_dt_count'] = -1.0/impr.show_x_dt.map(impr.groupby('show_x_dt').size())
impr['user_x_dt_count'] = -1.0/impr.user_x_dt.map(impr.groupby('user_x_dt').size())
impr['user_x_show_x_dt_count'] = -1.0/impr.user_x_show_x_dt.map(impr.groupby('user_x_show_x_dt').size())

test['show_x_dt_count'] = -1.0/test.show_x_dt.map(test.groupby('show_x_dt').size())
test['user_x_dt_count'] = -1.0/test.user_x_dt.map(test.groupby('user_x_dt').size())
test['user_x_show_x_dt_count'] = -1.0/test.user_x_show_x_dt.map(test.groupby('user_x_show_x_dt').size())


impr['user_x_dt_rank_count'] = impr.user_x_dt.map(impr.groupby('user_x_dt')['rank'].aggregate(
    lambda x: len(np.unique(x)))
)
test['user_x_dt_rank_count'] = test.user_x_dt.map(test.groupby('user_x_dt')['rank'].aggregate(
    lambda x: len(np.unique(x)))
)


# ctr features with smoothing
cv = KFold(8, shuffle=True, random_state=140)
global_mean = impr.is_clicked.mean()
alpha = 10.0

for col in impr.columns:
    if not col.startswith('id'):
        continue
    impr['ctr_'+col] = 0
    test['ctr_'+col] = 0
    
    for i_tr, i_ts in cv.split(impr):
        counts = impr.iloc[i_tr].groupby(col).size()
        means = impr.iloc[i_tr].groupby(col)['is_clicked'].mean()
        impr.loc[i_ts, 'ctr_'+col] = impr.iloc[i_ts][col].map(
            (counts*means+alpha*global_mean)/(counts+alpha)
        ).fillna(global_mean)
        test['ctr_'+col] += test[col].map(
            (counts*means+alpha*global_mean)/(counts+alpha)
        ).fillna(global_mean) / 8.0


# logreg meta features
for col in ['show_x_dt_count', 'user_x_dt_count', 'user_x_show_x_dt_count']:
    impr['lr_'+col] = 0
    test['lr_'+col] = 0
    
    for i_tr, i_ts in cv.split(impr):
        clf = LogisticRegression()
        clf.fit(impr.iloc[i_tr][col].values.reshape((-1, 1)), impr.iloc[i_tr]['is_clicked'])
        impr.loc[i_ts, 'lr_'+col] = clf.predict_proba(impr.iloc[i_ts][col].values.reshape((-1, 1)))[:, 1]
        test['lr_'+col] += clf.predict_proba(test[col].values.reshape((-1, 1)))[:, 1] / 8.0


params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.05,
    'num_leaves': 100,
    'lambda_l1': 1,
    'lambda_l2': 1,
    'feature_fraction': 0.75,
}


# base features
use_cols = ['ctr_id_show', 'ctr_id_user', 'ctr_id_organizer', 'ctr_id_show_building', 'ctr_id_parent_genre',
            'show_age_category', 'show_duration', 'show_maxprice', 'show_minprice', 'show_meanprice', 'show_stdprice',
            'ratio_clicks', 'ratio_clicks_impr', 'rank_clicks', 'rank_clicks_impr', 'inv_rank',
            'client_age', 'show_rating', 'show_num_rating', 'show_num_review']

# model 1
ltrain = lgb.Dataset(impr[use_cols], impr.is_clicked)
bst = lgb.train(params, train_set=ltrain, num_boost_round=80)
test['s1'] = bst.predict(test[use_cols])    

# model 2
scaler = StandardScaler()
tr = scaler.fit_transform(impr[use_cols].fillna(0))
ts = scaler.transform(test[use_cols].fillna(0))

clf = LogisticRegression(C=1)
clf.fit(tr, impr.is_clicked)
test['s2'] = clf.predict_proba(ts)[:, 1]


# features with leak
use_cols = ['ctr_id_show', 'ctr_id_user', 'ctr_id_organizer', 'ctr_id_show_building', 'ctr_id_parent_genre',
            'show_age_category', 'show_duration', 'show_maxprice', 'show_minprice', 'show_meanprice', 'show_stdprice',
            'ratio_clicks', 'ratio_clicks_impr', 'rank_clicks', 'rank_clicks_impr', 'inv_rank',
            'client_age', 'show_rating', 'show_num_rating', 'show_num_review',
                
            'show_x_dt_count', 'user_x_dt_count', 'user_x_show_x_dt_count',
            'lr_show_x_dt_count', 'lr_user_x_dt_count', 'lr_user_x_show_x_dt_count']

# model 3
ltrain = lgb.Dataset(impr[use_cols], impr.is_clicked)
bst = lgb.train(params, train_set=ltrain, num_boost_round=80)
test['s3'] = bst.predict(test[use_cols])    

# model 4
scaler = StandardScaler()
tr = scaler.fit_transform(impr[use_cols].fillna(0))
ts = scaler.transform(test[use_cols].fillna(0))

clf = LogisticRegression(C=1)
clf.fit(tr, impr.is_clicked)
test['s4'] = clf.predict_proba(ts)[:, 1]


# features with leak + 1
use_cols = ['ctr_id_show', 'ctr_id_user', 'ctr_id_organizer', 'ctr_id_show_building', 'ctr_id_parent_genre',
            'show_age_category', 'show_duration', 'show_maxprice', 'show_minprice', 'show_meanprice', 'show_stdprice',
            'ratio_clicks', 'ratio_clicks_impr', 'rank_clicks', 'rank_clicks_impr', 'inv_rank',
            'client_age', 'show_rating', 'show_num_rating', 'show_num_review',
                
            'show_x_dt_count', 'user_x_dt_count', 'user_x_show_x_dt_count',
            'lr_show_x_dt_count', 'lr_user_x_dt_count', 'lr_user_x_show_x_dt_count', 

            'user_x_dt_rank_count']

# model 5
ltrain = lgb.Dataset(impr[use_cols], impr.is_clicked)
bst = lgb.train(params, train_set=ltrain, num_boost_round=80)
test['s5'] = bst.predict(test[use_cols])    

# model 6
scaler = StandardScaler()
tr = scaler.fit_transform(impr[use_cols].fillna(0))
ts = scaler.transform(test[use_cols].fillna(0))

clf = LogisticRegression(C=1)
clf.fit(tr, impr.is_clicked)
test['s6'] = clf.predict_proba(ts)[:, 1]

# blending with handcrafting weights
test['prob'] = 0.1 * test['s1'] + 0.1 * test['s2'] + 0.3 * test['s3'] + 0.3 * test['s4'] + 0.1 * test['s5'] + 0.1 * test['s6']   
test.rename(columns={'id':'_ID_', 'prob':'_VAL_'})[['_ID_', '_VAL_']].to_csv('../output/submit_blend.csv', index=False)
