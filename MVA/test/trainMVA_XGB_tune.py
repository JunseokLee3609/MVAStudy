import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK 
from hyperopt.pyll import scope
import torch
import sys
# adding Folder_2 to the system path
sys.path.insert(0, '../lib')
from treeIO import varData, varDataTraining, ROOTDataset, getFile, rangeData
import numpy as np
import uproot
import matplotlib.pyplot as plt
import pickle
from sklearn import metrics as mt
# from torch.utils.data import DataLoader
from typing import Dict
import matplotlib.pyplot as plt

# custom_callback = CustomCallback()
custom_callback = xgb.callback.TrainingCallback()
# Define your model architecture
hyperSpace ={
    'tree_method': "hist",
    'n_estimators': scope.int(hp.quniform('n_estimators', 300, 500 ,100)), # 3
    'max_depth': scope.int(hp.quniform('max_depth', 7, 11, 2)), # 3
    'learning_rate': hp.loguniform('learning_rate', -4, -1), # 3
    'reg_lambda' : 5., # 3
    'reg_alpha': 2. , # 3
    'min_split_loss' : hp.qloguniform('min_split_loss', -5, -3, 1), #3
    'min_child_weight' : hp.qloguniform('min_child_weight', -7, -4 ,1), # 4
    'sampling_method' : 'uniform',
    'objective' : 'binary:logistic',
    'booster' : 'gbtree',
}
fit = False
fit = True
# saveTree = True
saveTree = False
doNotLoad = True
doNotLoad = False

# Define your loss function and optimizer
# (Note: XGBoost does not use these, but some packages like PyTorch require them)
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters())
nFileDATA = getFile('Data','Jpsi')[0]
nFileMCPR = getFile('MC','Prompt')[0]
nFileMCNR = getFile('MC','NonPrompt')[0]

batch_size = "50 MB"
DataLoader = ROOTDataset(nFileDATA, "mvaTupler/muonTuple", varDataTraining, '50 MB')
MCLoaderPR = ROOTDataset(nFileMCPR, "mvaTupler/muonTuple", varDataTraining, '30 MB')
MCLoaderNP = ROOTDataset(nFileMCPR, "mvaTupler/muonTuple", varDataTraining, '30 MB')

variables_dict = {var: 'float64' for var in varDataTraining}
vardim = len(varDataTraining)
print(vardim)
num_epochs = 1
var_start_from = 10;
simTypeSignal = [-6,-7,-8,-9-10,6,7,8,9,10]

# train_dataset_signal1 = ROOTDataset(file_path_signal1, tree_name, variables, batch_size)
# train_dataset_background1 = ROOTDataset(file_path_background1, tree_name, variables, batch_size)#, cut = "((mass > 3.8) | (mass < 2.8))")

# Train the model

# Preparing data for training and testing
# Baskets for data aux will include non training variables and raw will be a copy of the full data
inputsBKG = []
inputsSIG = []
# Labels will be 0 for signal and 1 for background
inputsAuxBKG = []
labelsBKG = []
inputsAuxSIG = []
labelsSIG = []
for idx, inputchunk in enumerate(MCLoaderPR):
    # Convert inputs and labels to NumPy arrays
    keyList = list(inputchunk[1].keys())
    dataValues = np.array(list(inputchunk[1].values())).T
    if idx == 0:
        print(keyList)
    dataValuesSIG= dataValues[np.isin(dataValues[:,keyList.index('matching')], simTypeSignal)].squeeze()
    dataValuesBKG= dataValues[~np.isin(dataValues[:,keyList.index('matching')], simTypeSignal)].squeeze()
    labelchunkSIG = np.full((dataValuesSIG.shape[0],1), 1)
    labelchunkBKG = np.full((dataValuesBKG.shape[0],1), 0)
    # Append data chun to inputs, index 0 is excluded (dimuon mass variable)
    inputsSIG.append(dataValuesSIG[:,var_start_from:])
    inputsAuxSIG.append(dataValuesSIG[:,:var_start_from-1])
    inputsBKG.append(dataValuesBKG[:,var_start_from:])
    inputsAuxBKG.append(dataValuesBKG[:,:var_start_from-1])
    # Append labels
    labelsSIG.append(labelchunkSIG)
    labelsBKG.append(labelchunkBKG)
    if idx > 5:
        break
for idx, inputchunk in enumerate(MCLoaderNP):
    # Convert inputs and labels to NumPy arrays
    keyList = list(inputchunk[1].keys())
    dataValues = np.array(list(inputchunk[1].values())).T
    if idx == 0:
        print(keyList)
    dataValuesSIG= dataValues[np.isin(dataValues[:,keyList.index('matching')], simTypeSignal)].squeeze()
    dataValuesBKG= dataValues[~np.isin(dataValues[:,keyList.index('matching')], simTypeSignal)].squeeze()
    labelchunkSIG = np.full((dataValuesSIG.shape[0],1), 1)
    labelchunkBKG = np.full((dataValuesBKG.shape[0],1), 0)
    # Append data chun to inputs, index 0 is excluded (dimuon mass variable)
    inputsSIG.append(dataValuesSIG[:,var_start_from:])
    inputsAuxSIG.append(dataValuesSIG[:,:var_start_from-1])
    inputsBKG.append(dataValuesBKG[:,var_start_from:])
    inputsAuxBKG.append(dataValuesBKG[:,:var_start_from-1])
    # Append labels
    labelsSIG.append(labelchunkSIG)
    labelsBKG.append(labelchunkBKG)
    if idx > 5:
        break

# Vertical stack all the data
inputs_sig = np.vstack(inputsSIG)
inputsAux_sig = np.vstack(inputsAuxSIG)
labels_sig = np.vstack(labelsSIG)

# for idx, inputchunk in enumerate(DataLoader):
#     # Convert inputs and labels to NumPy arrays
#     keyList = list(inputchunk[1].keys())
#     dataValues = np.array(list(inputchunk[1].values())).T
#     labelchunk = np.full((dataValues.shape[0],1), 0)
#     inputsBKG.append(dataValues[:,var_start_from:])
#     inputsAuxBKG.append(dataValues[:,:var_start_from-1])
#     labelsBKG.append(labelchunk)
#     if idx > 5:
#         break

# Vertical stack all the data
inputs_bkg = np.vstack(inputsBKG)
inputsAux_bkg = np.vstack(inputsAuxBKG)
labels_bkg = np.vstack(labelsBKG)



print( inputs_sig.shape, inputs_bkg.shape)
print( labels_sig.shape, labels_bkg.shape)

# Now the data include both signal and background data with column of label of each row
# Stack and shuffle the data
# TODO: Random sampling on signal and background cause imbalance in training better idea?
inputs = np.vstack((inputs_sig, inputs_bkg))
inputsAux = np.vstack((inputsAux_sig, inputsAux_bkg))
labels = np.vstack((labels_sig, labels_bkg))
nrows = inputs.shape[0]
# Random shuffle data
pm_idx = np.random.permutation(nrows)
inputs = inputs[pm_idx]
inputsAux = inputsAux[pm_idx]
labels = labels[pm_idx]

print(inputs.shape, labels.shape)
# print(inputs.shape)
# # exit()
print(nrows//2)
# Split sampels as train and test half and half
inputs_train, inputs_test = np.split(inputs, [nrows//2])
labels_train, labels_test = np.split(labels, [nrows//2])
inputsAux_train, inputsAux_test = np.split(inputsAux, [nrows//2])
# weights_train = np.ones(labels_train.shape)
# weights_test = np.ones(labels_test.shape)

dtrain = xgb.DMatrix(inputs_train, label=labels_train, weight = inputsAux_train[:,1])
dtest = xgb.DMatrix(inputs_test, label=labels_test, weight = inputsAux_test[:,1])

# Train model using XGBoost.fit()
# Define boost round
if fit:
    def objective(params):
        model = xgb.XGBClassifier(**params)
        model.verbosity = 2
        model.fit(inputs_train, labels_train, verbose = True)
        y_pred = model.predict(inputs_test)
        print(y_pred[:10].T)
        print(labels_test[:10].T)
        score = mt.accuracy_score(labels_test, y_pred )
        print(score)
        return {'loss': -score, 'status': STATUS_OK}

    best_params = fmin(objective, hyperSpace, algo=tpe.suggest, max_evals=200)
    print("Best set of hyperparameters: ", best_params)
