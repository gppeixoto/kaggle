import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
import time, re, gc, sys, os, gzip, copy
from dateutil.parser import parse
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import LogNorm
from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss
from keras.layers.core import Dense, Dropout, Activation
from keras.utils.layer_utils import BatchNormalization
from keras.models import Sequential
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adagrad
from keras.utils import np_utils

t0 = time.time()
# read data
print "reading .csv data from file..."
df = pd.read_csv(gzip.open('train.csv.gz'))

#drop useless columns and save categories for later
df = df.drop('Descript', 1)
df = df.drop('Resolution', 1)

#transform dates
def parse_dates(series):
    hour, day, month = [], [], []
    for date in series:
        date_parsed = parse(date)
        hour.append(date_parsed.hour)
        month.append(date_parsed.month)
        day.append(date_parsed.day)
    hour = pd.Series(hour)
    day = pd.Series(day)
    month = pd.Series(month)
    dates = pd.concat([hour, day, month], axis=1)
    dates.columns = ['Hour', 'Day', 'Month']
    return dates

print "parsing dates..."
dates = parse_dates(df.Dates)
df = df.drop('Dates', 1)
df = pd.concat([df, dates], axis=1)

# get intersections
def get_address_features(addresses):
    isCorner, blockNumber = [], []
    for addr in addresses:
        isCorner.append(1 if '/' in addr else 0)
        match = re.match(r'[0-9]+', addr.split()[0])
        blockNumber.append(-1 if not match else match.group())
    isCorner = pd.Series(isCorner)
    blockNumber = pd.Series(blockNumber)
    addr_features = pd.concat([isCorner, blockNumber], axis=1)
    addr_features.columns = ['Intersection', 'BlockNumber']
    return addr_features

print "parsing addresses..."
feats = get_address_features(df.Address)
df = pd.concat([df, feats], axis=1)

df[['X', 'Y']] = StandardScaler().fit_transform(df[['X', 'Y']])
df = df[abs(df.Y) < 100]
df.index = range(len(df))

print "generating dummy features..."
pd_district_dummy = pd.get_dummies(df.PdDistrict)
pd_district_dummy.columns = ['PD_'+i for i in pd_district_dummy.columns]
day_week_dummies = pd.get_dummies(df.DayOfWeek)
day_week_dummies.columns = ['DAY_'+i.upper() for i in day_week_dummies.columns]
df = df.drop('PdDistrict', 1)
df = df.drop('DayOfWeek', 1)
df = pd.concat([df, day_week_dummies, pd_district_dummy], axis=1)

print "featurizing addresses..."
t1=time.time()
aux = df[['Category', 'Address']]
cat_prior = aux.Category.value_counts() / aux.Category.size
cat_default_logodds = (np.log(cat_prior)-np.log(1-cat_prior)).sort_index()
addr_cat_logodds = {}
addr_prior_logodds = {}
addr_counts = aux.Address.value_counts().sort_index()
T_addr_counts = addr_counts.sum() + .0
addr_cat_counts = aux.groupby(['Address', 'Category']).size()
MIN_COUNT = 5
del aux

for addr in df.Address.unique():
    proba = addr_counts[addr] / T_addr_counts
    addr_prior_logodds[addr] = np.log(proba)-np.log(1-proba)
    addr_cat_logodds[addr] = copy.deepcopy(cat_default_logodds)
    if len(addr_cat_counts[addr]) > MIN_COUNT:
        for cat in addr_cat_counts[addr].keys():
            proba = addr_cat_counts[addr][cat] / (addr_counts[addr].sum()+.0)
            addr_cat_logodds[addr][cat] = np.log(proba)-np.log(1-proba)
aux = df.Address.apply(lambda addr: addr_cat_logodds[addr])
cols = ['LOGODD_'+'_'.join(col.split()) for col in aux.columns]
aux2 = df.Address.apply(lambda addr: addr_prior_logodds[addr])
aux = pd.concat([aux, aux2], axis=1)
cols.append("LOGODD_PRIOR")
aux.columns = cols
df = df.drop('Address', 1)
df = pd.concat([df, aux], axis=1)
del aux
gc.collect()

labels = df.Category
labels_map = sorted(labels.unique())
labels_map = dict(zip(labels_map, range(len(labels_map))))
labels = labels.apply(lambda x: labels_map[x])
df = df.drop('Category', 1)
labels = np_utils.to_categorical(
    labels, nb_classes=(len(labels.unique()))
    )

# Model generation
print "compiling model..."
input_dim = 64
hidden_dim = 32
nb_epoch = 20
dp = .5
n_layers = 2
output_dim = 39

model = Sequential()
model.add(Dense(hidden_dim, init="glorot_uniform", input_dim=input_dim))
model.add(PReLU())
model.add(BatchNormalization())
model.add(Dropout(dp))

for i in xrange(n_layers):
    model.add(Dense(hidden_dim, input_dim=hidden_dim)) #fully connected layer
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(.5))

model.add(Dense(output_dim, input_dim=hidden_dim))
model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam")

# Data splitting
print "splitting data..."
split = 65
idxs = range(len(df))
np.random.shuffle(idxs)
cut = len(df)*split/100
df = df.as_matrix()
x_train, y_train = df[idxs[:cut]], labels[idxs[:cut]]
x_val, y_val = df[idxs[cut:]], labels[idxs[cut:]]

# fitting model
print "fitting model..."
fitting = model.fit(x_train, y_train, nb_epoch=nb_epoch, batch_size=64, verbose=2, validation_data=(x_val, y_val))

t0 = time.time()-t0
print 'total time elapsed: %.0fs' % t0