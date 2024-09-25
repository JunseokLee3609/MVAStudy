import xgboost as xgb
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
model = xgb.XGBClassifier(
    n_estimators=40,
    max_depth=14,
    learning_rate=0.16,
    reg_lambda=5.,
    reg_alpha=2,
    min_split_loss = 0.01,
    min_child_weight = 0.01,
    sampling_method = 'uniform',
    objective = 'binary:logistic',
    booster = 'gbtree',
    num_parallel_tree = 10,
    callbacks=[custom_callback],
)
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
DataLoader = ROOTDataset(nFileDATA, "mvaTupler/muonTuple", varDataTraining, '20 MB')
MCLoaderPR = ROOTDataset(nFileMCPR, "mvaTupler/muonTuple", varDataTraining, '10 MB')
MCLoaderNP = ROOTDataset(nFileMCPR, "mvaTupler/muonTuple", varDataTraining, '10 MB')

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
    model.verbosity = 2
    for epoch in range(num_epochs):
        print("Fiiting loop %d" % epoch)
        # Train the model on this batch
        model.fit(inputs_train, labels_train, eval_set=[(inputs_train, labels_train), (inputs_test, labels_test)], verbose = True)
    #     break;

    # Dump model to pickle format, recommend using saving as json
    # pickle model
    # pickle.dump(model, open("xgb_model.pkl", "wb"))

    # Save trained model as json
    model.save_model("xgb_model.json")
    # evals_result = custom_callback.evals_result
    evals_result = model.evals_result()
    train_auc = evals_result['validation_0']['logloss']
    val_auc = evals_result['validation_1']['logloss']
    # Generate the x-axis values (iterations)
    iterations = range(1, len(train_auc) + 1)

    # Create the plot
    plt.figure(figsize=(10, 6))
    # Set Y axis to logarithmic scale
    plt.yscale('log')
    plt.plot(iterations, train_auc, label='Train AUC')
    plt.plot(iterations, val_auc, label='Validation AUC')
    plt.savefig('auc.pdf')



# Load model
# TODO: Better code to separate loading and trianing?
if not fit:
    model = xgb.XGBClassifier()
    model.load_model("xgb_model.json")

print(model.n_features_in_)

# Plot the tree
fig, ax = plt.subplots(figsize=(30, 30))
xgb.plot_tree(model, num_trees=1, ax=ax)
plt.savefig("tree.png")

total_correct = 0
total_samples = 0


inputsData = []
for idx, inputchunk in enumerate(DataLoader):
    # Convert inputs and labels to NumPy arrays
    keyList = list(inputchunk[1].keys())
    dataValues = np.array(list(inputchunk[1].values())).T
    print(dataValues.shape)
    labelchunk = np.full((dataValues.shape[0],1), 0)
    inputsData.append(dataValues[:,:])
    if idx > 5:
        break

inputs_Data = np.vstack(inputsData)

# Make predictions on this batch
input_labeled = np.hstack((inputs_test,labels_test))
predictions = model.predict(inputs_test)
pred_prob = model.predict_proba(inputs_test)
pred_labeled = np.hstack((pred_prob,labels_test))
pred_prob_data = model.predict_proba(inputs_Data[:,var_start_from:])


plt.figure(figsize=(10, 6))
fpr, tpr, _  = mt.roc_curve(labels_test, pred_prob[:,1], sample_weight = inputs_test[:,1])
plt.plot(fpr,tpr)
lpt = np.array([0,1])
plt.plot(lpt, linestyle = 'dotted' )
plt.savefig('pred_ROC.pdf')

plt.clf()
# Create histogram to plot pred_prob[0] and pred_prob[1]
plt.figure(figsize=(10, 6))
plt.hist(pred_labeled[pred_labeled[:,-1]==0, 1], bins=1000, label='Background', alpha=0.5)
plt.hist(pred_labeled[pred_labeled[:,-1]==1, 1], bins=1000, label='Signal', alpha=0.5)
plt.hist(pred_prob_data[:,1], bins=1000, label='Signal prob data', alpha=0.5)
plt.legend(loc='upper right')
plt.savefig('pred_prob.pdf')


print("stack shape (aux, nom")
print(inputs_test.shape)
print("input shape:")
print(inputs_test.shape)
inputsAux_train = np.hstack((inputsAux_train, inputs_train, labels_train))
inputsAux_test = np.hstack((inputsAux_test, inputs_test, labels_test))
inputsAll = np.vstack((inputsAux_train, inputsAux_test))
# Add variables to save
variables_dict.update({'prob0': 'float64', 'prob1': 'float64', 'label': 'int32'})
# Save the data to root file
if saveTree:
    with uproot.recreate('test_xgb_test.root') as f:
        print(variables_dict)
        f.mktree("newT", variables_dict)
        print(vardim)
        entry = { varDataTraining[i] : inputsAll[:,i] for i in range(vardim)}
        entry['prob0'] = pred_prob[:,0]
        entry['prob1'] = pred_prob[:,1]
        entry['label'] = inputsAll[:,-1]
        f['newT'].extend(entry)
    # for i in range(vardim):
    #     print(i)
    #     f["newT"].extend({variables[i]: inputs_test[:,i]})

# Compute the number of correct predictions
print("predictions shape")
print( predictions.shape)
print(labels_test.shape)
total_correct += (predictions.reshape(-1,1) == labels_test).sum()

# Compute the total number of samples
total_samples += labels_test.size

# # Compute the accuracy of the model on the test set
accuracy = total_correct / total_samples
print("Accuracy: {:.2f}%".format(accuracy * 100))