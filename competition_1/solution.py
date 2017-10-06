import numpy as np
import pandas as pd

from scipy.stats import mode


INPUT_PATH = '../input/'
OUTPUT_PATH = '../output/'

# for validation
def split_data(data, week):
    train, test = data.loc[data.week < week], data.loc[data.week == week]
    
    test = test.groupby('id').head(1).reset_index(drop=True)
    test.set_index('id', inplace=True)
    true = pd.DataFrame.from_dict({'id':np.unique(data.id)})
    true['sum'] = true.id.map(test['sum']).fillna(0).astype(int)
    
    return train, true


# load data
data = pd.read_csv(INPUT_PATH + 'train2.csv.xls')
data['week'] = ((data['date'] - 5) / 7).astype(int)

# parameters
f_delta = -4
f_minus = -7
f_plus = 7
f_pow = 1.2
last_day = 438

train = data.copy()

# weights by date for weighted mode 
train['weights'] = (train.date / train.date.max()) ** f_pow
byw = train.groupby(['id', 'sum'])['weights'].sum()
ind = byw.reset_index().groupby('id')['weights'].apply(np.argmax)
tmp_prediction = byw.reset_index().iloc[ind].reset_index(drop=True)
tmp_prediction.set_index('id', inplace=True)

# preparing prediction df
train_gr = train.groupby('id')
prediction = pd.DataFrame.from_dict({'id': np.unique(data.id)})
prediction['sum'] = prediction.id.map(tmp_prediction['sum']).fillna(0).astype(int)

# calculate last day of visit and mean day between visits
train['delta_day'] = train_gr['date'].diff()
prediction['delta_day'] = prediction.id.map(train_gr['delta_day'].mean())
prediction['last_day'] = train_gr.tail(1).reset_index(drop=True)['date']

# filter: long time without visits
# filter: estimated visit before our week
# filters: estimated visit after our week
def make_zero(row):
    if row['last_day'] - last_day < f_minus \
            or row['last_day'] + row['delta_day'] - last_day < f_delta \
            or row['last_day'] + row['delta_day'] - last_day > f_plus:
        return 0
    else:
        return int(row['sum'])

prediction['sum'] = prediction.apply(make_zero, axis=1)

# saving
prediction[['id', 'sum']].to_csv(OUTPUT_PATH + 'submission_3.csv', index=False)