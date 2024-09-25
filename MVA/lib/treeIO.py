import uproot
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import json
varMC = [
    'matchGEN',
    'pT',
    'y',
    'eta',
    'phi',
    'mass',
]
varData = [
    'pT',
    'y',
    'eta',
    'phi',
    'mass',
]
varpair = {
# ['VtxProb',
# 'VtxChi2',
# 'VtxNDF',
# '3DCosPointingAngle',
'3DPointingAngle':'matchGen3DPointingAngle',
'2DPointingAngle':'matchGen2DPointingAngle',
# '3DDecayLengthSignificance',
'3DDecayLength':'matchGen3DDecayLength',
# '2DDecayLengthSignificance':
'2DDecayLength': 'matchGen2DDecayLength',
'massDaugther1': 'matchGen_D0mass',
'pTD1':'matchGen_D0pT',  #D1 (dStar->D0)
'EtaD1':'matchGen_D0eta',
'PhiD1':'matchGen_D0phi',
# 'VtxProbDaugther1':
# 'VtxChi2Daugther1':
# 'VtxNDFDaugther1':
# '3DCosPointingAngleDaugther1': #daughter1 (dstar->d0))
# '3DPointingAngleDaugther1':
# '2DCosPointingAngleDaugther1':
# '2DPointingAngleDaugther1':
# '3DDecayLengthSignificanceDaugther1':
'3DDecayLengthDaugther1':'matchGen_D1decayLength3D_',
# '3DDecayLengthErrorDaugther1':
# '2DDecayLengthSignificanceDaugther1':
# 'zDCASignificanceDaugther2':
# 'xyDCASignificanceDaugther2':
# 'NHitD2':
# 'HighPurityDaugther2':
'pTD2':'matchGen_D1pT', #D2 (dstar->pion)
'EtaD2':'matchGen_D1eta',
'PhiD2':'matchGen_D1phi',
# 'pTerrD1':
# 'pTerrD2':
# 'dedxHarmonic2D2':
# 'zDCASignificanceGrandDaugther1':
# 'zDCASignificanceGrandDaugther2':
# 'xyDCASignificanceGrandDaugther1':
# 'xyDCASignificanceGrandDaugther2':
# 'NHitGrandD1':
# 'NHitGrandD2':
# 'HighPurityGrandDaugther1':
# 'HighPurityGrandDaugther2':
'pTGrandD1': 'matchGen_D0Dau1_pT',
'pTGrandD2': 'matchGen_D0Dau2_pT',
# 'pTerrGrandD1':
# 'pTerrGrandD2':
'EtaGrandD1': 'matchGen_D0Dau1_eta',
'EtaGrandD2': 'matchGen_D0Dau2_eta',
# 'dedxHarmonic2GrandD1':
# 'dedxHarmonic2GrandD2'
    
    
    
    
    
    
}
var2DMC = [
'matchGEN',
'matchGen3DPointingAngle',
'matchGen2DPointingAngle',
'matchGen3DDecayLength',
'matchGen2DDecayLength',
'matchGen_D0pT', #Dstar->D0
'matchGen_D0eta',
'matchGen_D0phi',
'matchGen_D0mass',
'matchGen_D0y',
'matchGen_D0charge',
'matchGen_D0pdgId',
'matchGen_D0Dau1_pT', #d0->daughter(0) (pion or kaon)
'matchGen_D0Dau1_eta',
'matchGen_D0Dau1_phi',
'matchGen_D0Dau1_mass',
'matchGen_D0Dau1_y',
'matchGen_D0Dau1_charge',
'matchGen_D0Dau1_pdgId',
'matchGen_D0Dau2_pT', #d0->daughter(1) (pion or kaon)
'matchGen_D0Dau2_eta',
'matchGen_D0Dau2_phi',
'matchGen_D0Dau2_mass',
'matchGen_D0Dau2_y',
'matchGen_D0Dau2_charge',
'matchGen_D0Dau2_pdgId',
'matchGen_D1pT', #Dstar->pion
'matchGen_D1eta',
'matchGen_D1phi',
'matchGen_D1mass',
'matchGen_D1y',
'matchGen_D1charge',
'matchGen_D1pdgId',
'matchGen_D1decayLength2D_',
'matchGen_D1decayLength3D_',
'matchGen_D1angle2D_',
'matchGen_D1angle3D_',
'matchGen_D1ancestorId_',
'matchGen_D1ancestorFlavor_'
]
var2DData = [
'VtxProb',
'VtxChi2',
'VtxNDF',
'3DCosPointingAngle',
'3DPointingAngle',
'2DCosPointingAngle',
'2DPointingAngle',
'3DDecayLengthSignificance',
'3DDecayLength',
'2DDecayLengthSignificance',
'2DDecayLength',
'massDaugther1',
'pTD1', #D1 (dStar->D0)
'EtaD1',
'PhiD1',
'VtxProbDaugther1',
'VtxChi2Daugther1',
'VtxNDFDaugther1',
'3DCosPointingAngleDaugther1',
'3DPointingAngleDaugther1',
'2DCosPointingAngleDaugther1',
'2DPointingAngleDaugther1',
'3DDecayLengthSignificanceDaugther1',
'3DDecayLengthDaugther1',
'3DDecayLengthErrorDaugther1',
'2DDecayLengthSignificanceDaugther1',
'zDCASignificanceDaugther2',
'xyDCASignificanceDaugther2',
'NHitD2',
'HighPurityDaugther2',
'pTD2', #D1 (dStar->pion)
'EtaD2',
'PhiD2',
'pTerrD1',
'pTerrD2',
'dedxHarmonic2D2',
'zDCASignificanceGrandDaugther1', #daughter1 (dstar->d0))
'zDCASignificanceGrandDaugther2', 
'xyDCASignificanceGrandDaugther1',
'xyDCASignificanceGrandDaugther2',
'NHitGrandD1',
'NHitGrandD2',
'HighPurityGrandDaugther1',
'HighPurityGrandDaugther2',
'pTGrandD1', #(d0->pion or kaon)
'pTGrandD2',
'pTerrGrandD1',
'pTerrGrandD2',
'EtaGrandD1',
'EtaGrandD2',
'dedxHarmonic2GrandD1',
'dedxHarmonic2GrandD2'
    
    # 'y',
    # 'phi'
    # 'cBin', 'nColl', 'hiHF', 'hiHFPlus', 'hiHFMinus', 'hiNtrk', 'hiNpixelTrk', 'hiEB', 'hiEE',
    # 'global_muon', 'matching', 'ParticleFlow', 'norm_chi2', 'norm_chi2_inner',
    # 'norm_chi2_bestTraker', 'local_chi2', 'trkKink', 'segment_comp', 'n_Valid_hits', 
    # 'n_Valid_hits_inner', 'n_Valid_hits_bestTraker', 'n_MatchedStations', 'n_MatchedChamber', 'dxy', 'dz', 'dz_new', 'Valid_pixel', 'pixel_layers', 'tracker_layers', 'pt', 'eta',
    # 'validFraction', 'Medium_muon', 'Tight_muon', 'Tight_muon_newdef', 'nVtx', 'glbTrkProb', 
    # 'glbKink', 'localDist', 'chi2LocalPos', 'inner_station_badHits', 'inner_station_Hits', 
    # 'caloCompatibility', 'trk_lambda', 'trk_lambda_err', 'trk_inner_missing_hits', 
    # 'trk_outer_missing_hits', 'trk_num_loss_hits', 'qoverp', 'qoverp_err', 'beta', 'beta_err', 
    # 'jetPtRatio', 'jetPtRel', 'ecalIso', 'hcalIso', 'pfSize', 'pfAvg_corrected_hcalE', 
    # 'pfAvg_corrected_ecalE', 'pfAvg_raw_hcalE', 'pfAvg_raw_ecalE', 'pfVar_corrected_hcalE', 
    # 'pfVar_corrected_ecalE', 'pfVar_raw_hcalE', 'pfVar_raw_ecalE',
    # 'dDxDz_sta1_DT', 
    # 'dDyDz_sta1_DT', 'dX_sta1_DT', 'dY_sta1_DT', 'pullDxDz_sta1_DT', 'pullDyDz_sta1_DT', 
    # 'pullX_sta1_DT', 'pullY_sta1_DT', 'segmentDxDz_sta1_DT', 'segmentDxDzErr_sta1_DT', 
    # 'segmentDyDz_sta1_DT', 'segmentDyDzErr_sta1_DT', 'segmentX_sta1_DT', 'segmentXErr_sta1_DT', 
    # 'segmentY_sta1_DT', 'segmentYErr_sta1_DT', 'trackDist_sta1_DT', 'trackDistErr_sta1_DT', 
    # 'trackDxDz_sta1_DT', 'trackDxDzErr_sta1_DT', 'trackDyDz_sta1_DT', 'trackDyDzErr_sta1_DT', 
    # 'trackEdgeX_sta1_DT', 'trackEdgeY_sta1_DT', 'trackX_sta1_DT', 'trackXErr_sta1_DT', 
    # 'trackY_sta1_DT', 'trackYErr_sta1_DT', 'dDxDz_sta2_DT', 'dDyDz_sta2_DT', 'dX_sta2_DT', 
    # 'dY_sta2_DT', 'pullDxDz_sta2_DT', 'pullDyDz_sta2_DT', 'pullX_sta2_DT', 'pullY_sta2_DT', 
    # 'segmentDxDz_sta2_DT', 'segmentDxDzErr_sta2_DT', 'segmentDyDz_sta2_DT', 
    # 'segmentDyDzErr_sta2_DT', 'segmentX_sta2_DT', 'segmentXErr_sta2_DT', 'segmentY_sta2_DT', 
    # 'segmentYErr_sta2_DT', 'trackDist_sta2_DT', 'trackDistErr_sta2_DT', 'trackDxDz_sta2_DT', 
    # 'trackDxDzErr_sta2_DT', 'trackDyDz_sta2_DT', 'trackDyDzErr_sta2_DT', 'trackEdgeX_sta2_DT', 
    # 'trackEdgeY_sta2_DT', 'trackX_sta2_DT', 'trackXErr_sta2_DT', 'trackY_sta2_DT', 
    # 'trackYErr_sta2_DT', 'dDxDz_sta1_CSC', 'dDyDz_sta1_CSC', 'dX_sta1_CSC', 'dY_sta1_CSC', 
    # 'pullDxDz_sta1_CSC', 'pullDyDz_sta1_CSC', 'pullX_sta1_CSC', 'pullY_sta1_CSC', 
    # 'segmentDxDz_sta1_CSC', 'segmentDxDzErr_sta1_CSC', 'segmentDyDz_sta1_CSC', 
    # 'segmentDyDzErr_sta1_CSC', 'segmentX_sta1_CSC', 'segmentXErr_sta1_CSC', 'segmentY_sta1_CSC', 
    # 'segmentYErr_sta1_CSC', 'trackDist_sta1_CSC', 'trackDistErr_sta1_CSC', 'trackDxDz_sta1_CSC', 
    # 'trackDxDzErr_sta1_CSC', 'trackDyDz_sta1_CSC', 'trackDyDzErr_sta1_CSC', 
    # 'trackEdgeX_sta1_CSC', 'trackEdgeY_sta1_CSC', 'trackX_sta1_CSC', 'trackXErr_sta1_CSC', 
    # 'trackY_sta1_CSC', 'trackYErr_sta1_CSC', 'dDxDz_sta2_CSC', 'dDyDz_sta2_CSC', 'dX_sta2_CSC', 
    # 'dY_sta2_CSC', 'pullDxDz_sta2_CSC', 'pullDyDz_sta2_CSC', 'pullX_sta2_CSC', 'pullY_sta2_CSC', 
    # 'segmentDxDz_sta2_CSC', 'segmentDxDzErr_sta2_CSC', 'segmentDyDz_sta2_CSC', 
    # 'segmentDyDzErr_sta2_CSC', 'segmentX_sta2_CSC', 'segmentXErr_sta2_CSC', 'segmentY_sta2_CSC', 
    # 'segmentYErr_sta2_CSC', 'trackDist_sta2_CSC', 'trackDistErr_sta2_CSC', 'trackDxDz_sta2_CSC', 
    # 'trackDxDzErr_sta2_CSC', 'trackDyDz_sta2_CSC', 'trackDyDzErr_sta2_CSC', 
    # 'trackEdgeX_sta2_CSC', 'trackEdgeY_sta2_CSC', 'trackX_sta2_CSC', 'trackXErr_sta2_CSC', 
    # 'trackY_sta2_CSC', 'trackYErr_sta2_CSC', 'dDxDz_sta1_RPC', 'dDyDz_sta1_RPC', 'dX_sta1_RPC', 
    # 'dY_sta1_RPC', 'pullDxDz_sta1_RPC', 'pullDyDz_sta1_RPC', 'pullX_sta1_RPC', 'pullY_sta1_RPC', 
    # 'segmentDxDz_sta1_RPC', 'segmentDxDzErr_sta1_RPC', 'segmentDyDz_sta1_RPC', 
    # 'segmentDyDzErr_sta1_RPC', 'segmentX_sta1_RPC', 'segmentXErr_sta1_RPC', 'segmentY_sta1_RPC', 
    # 'segmentYErr_sta1_RPC', 'trackDist_sta1_RPC', 'trackDistErr_sta1_RPC', 'trackDxDz_sta1_RPC', 
    # 'trackDxDzErr_sta1_RPC', 'trackDyDz_sta1_RPC', 'trackDyDzErr_sta1_RPC', 
    # 'trackEdgeX_sta1_RPC', 'trackEdgeY_sta1_RPC', 'trackX_sta1_RPC', 'trackXErr_sta1_RPC', 
    # 'trackY_sta1_RPC', 'trackYErr_sta1_RPC', 'dDxDz_sta2_RPC', 'dDyDz_sta2_RPC', 'dX_sta2_RPC', 
    # 'dY_sta2_RPC', 'pullDxDz_sta2_RPC', 'pullDyDz_sta2_RPC', 'pullX_sta2_RPC', 'pullY_sta2_RPC', 
    # 'segmentDxDz_sta2_RPC', 'segmentDxDzErr_sta2_RPC', 'segmentDyDz_sta2_RPC', 
    # 'segmentDyDzErr_sta2_RPC', 'segmentX_sta2_RPC', 'segmentXErr_sta2_RPC', 'segmentY_sta2_RPC', 
    # 'segmentYErr_sta2_RPC', 'trackDist_sta2_RPC', 'trackDistErr_sta2_RPC', 'trackDxDz_sta2_RPC', 
    # 'trackDxDzErr_sta2_RPC', 'trackDyDz_sta2_RPC', 'trackDyDzErr_sta2_RPC', 
    # 'trackEdgeX_sta2_RPC', 'trackEdgeY_sta2_RPC', 'trackX_sta2_RPC', 'trackXErr_sta2_RPC', 
    # 'trackY_sta2_RPC', 'trackYErr_sta2_RPC'
]
'''
variables cbin to matching is only for event info/ weight / labelling purpose
cBin 0
nColl 1 (weight)
hiHF 2
hiHFPlus 3
hiHFMinus 4
hiNtrk 5
hiNpixelTrk 6
hiEB 7 
hiEE 8
matching 9 (class label)
'''
varDataTraining = [
    'cBin', 'nColl', 'hiHF', 'hiHFPlus', 'hiHFMinus', 'hiNtrk', 'hiNpixelTrk', 'hiEB', 'hiEE', 'matching', 
    'global_muon', 'ParticleFlow', 'norm_chi2', 'norm_chi2_inner', 'norm_chi2_bestTraker', 'local_chi2', 'trkKink', 'segment_comp', 'n_Valid_hits', 'n_Valid_hits_inner',
    'n_Valid_hits_bestTraker', 'n_MatchedStations', 'n_MatchedChamber', 'dxy', 'dz', 'Valid_pixel', 'pixel_layers', 'tracker_layers', 'pt', 'eta', 
    'validFraction', 'glbTrkProb', 'glbKink', 'localDist', 'chi2LocalPos', 'inner_station_badHits', 
    'inner_station_Hits', 'caloCompatibility', 'trk_lambda', 'trk_lambda_err', 'trk_inner_missing_hits', 
    'trk_outer_missing_hits', 'trk_num_loss_hits', 'qoverp', 'qoverp_err', 'ecalIso', 'hcalIso',
    'dDxDz_sta1_DT', 
    'dDyDz_sta1_DT', 'dX_sta1_DT', 'dY_sta1_DT', 'pullDxDz_sta1_DT', 'pullDyDz_sta1_DT', 
    'pullX_sta1_DT', 'pullY_sta1_DT', 'segmentDxDz_sta1_DT', 'segmentDxDzErr_sta1_DT', 
    'segmentDyDz_sta1_DT', 'segmentDyDzErr_sta1_DT', 'segmentX_sta1_DT', 'segmentXErr_sta1_DT', 
    'segmentY_sta1_DT', 'segmentYErr_sta1_DT', 'trackDist_sta1_DT', 'trackDistErr_sta1_DT', 
    'trackDxDz_sta1_DT', 'trackDxDzErr_sta1_DT', 'trackDyDz_sta1_DT', 'trackDyDzErr_sta1_DT', 
    'trackEdgeX_sta1_DT', 'trackEdgeY_sta1_DT', 'trackX_sta1_DT', 'trackXErr_sta1_DT', 
    'trackY_sta1_DT', 'trackYErr_sta1_DT', 'dDxDz_sta2_DT', 'dDyDz_sta2_DT', 'dX_sta2_DT', 
    'dY_sta2_DT', 'pullDxDz_sta2_DT', 'pullDyDz_sta2_DT', 'pullX_sta2_DT', 'pullY_sta2_DT', 
    'segmentDxDz_sta2_DT', 'segmentDxDzErr_sta2_DT', 'segmentDyDz_sta2_DT', 
    'segmentDyDzErr_sta2_DT', 'segmentX_sta2_DT', 'segmentXErr_sta2_DT', 'segmentY_sta2_DT', 
    'segmentYErr_sta2_DT', 'trackDist_sta2_DT', 'trackDistErr_sta2_DT', 'trackDxDz_sta2_DT', 
    'trackDxDzErr_sta2_DT', 'trackDyDz_sta2_DT', 'trackDyDzErr_sta2_DT', 'trackEdgeX_sta2_DT', 
    'trackEdgeY_sta2_DT', 'trackX_sta2_DT', 'trackXErr_sta2_DT', 'trackY_sta2_DT', 
    'trackYErr_sta2_DT', 'dDxDz_sta1_CSC', 'dDyDz_sta1_CSC', 'dX_sta1_CSC', 'dY_sta1_CSC', 
    'pullDxDz_sta1_CSC', 'pullDyDz_sta1_CSC', 'pullX_sta1_CSC', 'pullY_sta1_CSC', 
    'segmentDxDz_sta1_CSC', 'segmentDxDzErr_sta1_CSC', 'segmentDyDz_sta1_CSC', 
    'segmentDyDzErr_sta1_CSC', 'segmentX_sta1_CSC', 'segmentXErr_sta1_CSC', 'segmentY_sta1_CSC', 
    'segmentYErr_sta1_CSC', 'trackDist_sta1_CSC', 'trackDistErr_sta1_CSC', 'trackDxDz_sta1_CSC', 
    'trackDxDzErr_sta1_CSC', 'trackDyDz_sta1_CSC', 'trackDyDzErr_sta1_CSC', 
    'trackEdgeX_sta1_CSC', 'trackEdgeY_sta1_CSC', 'trackX_sta1_CSC', 'trackXErr_sta1_CSC', 
    'trackY_sta1_CSC', 'trackYErr_sta1_CSC', 'dDxDz_sta2_CSC', 'dDyDz_sta2_CSC', 'dX_sta2_CSC', 
    'dY_sta2_CSC', 'pullDxDz_sta2_CSC', 'pullDyDz_sta2_CSC', 'pullX_sta2_CSC', 'pullY_sta2_CSC', 
    'segmentDxDz_sta2_CSC', 'segmentDxDzErr_sta2_CSC', 'segmentDyDz_sta2_CSC', 
    'segmentDyDzErr_sta2_CSC', 'segmentX_sta2_CSC', 'segmentXErr_sta2_CSC', 'segmentY_sta2_CSC', 
    'segmentYErr_sta2_CSC', 'trackDist_sta2_CSC', 'trackDistErr_sta2_CSC', 'trackDxDz_sta2_CSC', 
    'trackDxDzErr_sta2_CSC', 'trackDyDz_sta2_CSC', 'trackDyDzErr_sta2_CSC', 
    'trackEdgeX_sta2_CSC', 'trackEdgeY_sta2_CSC', 'trackX_sta2_CSC', 'trackXErr_sta2_CSC', 
    'trackY_sta2_CSC', 'trackYErr_sta2_CSC', 
    'trackDist_sta1_RPC', 'trackDistErr_sta1_RPC', 'trackDxDz_sta1_RPC', 'trackDxDzErr_sta1_RPC',
    'trackDyDz_sta1_RPC', 'trackDyDzErr_sta1_RPC', 'trackEdgeX_sta1_RPC', 'trackEdgeY_sta1_RPC', 
    'trackX_sta1_RPC', 'trackXErr_sta1_RPC', 'trackY_sta1_RPC', 'trackYErr_sta1_RPC', 
    'trackDist_sta2_RPC', 'trackDistErr_sta2_RPC', 'trackDxDz_sta2_RPC', 'trackDxDzErr_sta2_RPC', 
    'trackDyDz_sta2_RPC', 'trackDyDzErr_sta2_RPC', 'trackEdgeX_sta2_RPC', 'trackEdgeY_sta2_RPC', 
    'trackX_sta2_RPC', 'trackXErr_sta2_RPC', 'trackY_sta2_RPC', 'trackYErr_sta2_RPC'
]
varDataTrainingRanked = [
    'cBin', 'nColl', 'hiHF', 'hiHFPlus', 'hiHFMinus', 'hiNtrk', 'hiNpixelTrk', 'hiEB', 'hiEE', 'matching', 
    'pt', 'eta', 'n_MatchedStations', 'hcalIso', 'trackDyDzErr_sta1_RPC', 'trackXErr_sta1_RPC', 'ecalIso', 'segmentY_sta1_CSC', 'validFraction', 'qoverp', 'qoverp_err','trackYErr_sta1_RPC', 'dxy', 'trkKink', 'n_MatchedChamber', 'trk_outer_missing_hits', 'trackXErr_sta2_CSC', 'pullDxDz_sta1_CSC', 'segment_comp', 'dz', 'trackYErr_sta1_DT', 'caloCompatibility','pullY_sta1_CSC', 'trackDxDzErr_sta2_DT', 'dDyDz_sta1_CSC', 'trackEdgeY_sta1_RPC','dDxDz_sta1_CSC', 'trackDist_sta1_CSC', 'glbKink', 'trackDyDzErr_sta1_CSC', 'norm_chi2_bestTraker', 'segmentDxDz_sta1_CSC', 'segmentYErr_sta1_CSC', 'dX_sta1_CSC', 'trackDyDz_sta1_RPC','trackDistErr_sta2_RPC', 'trackEdgeY_sta1_CSC','pullDxDz_sta2_CSC', 'segmentY_sta2_CSC', 'localDist' ]
rangeMC = {
    'matchGen3DCosPointingAngle' : [[-1.5,1.5],100],
    'matchGen3DPointingAngle' : [[-1.5,1.5],100],
    'matchGen2DCosPointingAngle' : [[1.5,1.5],100],
    'matchGen2DPointingAngle' : [[-1.5,1.5],100],
    
}
rangeData = {
    'pT' : [[0., 30.], 30 ],
    'eta' : [[-3., 3.], 24 ],
    'phi' : [[-1*np.pi, np.pi], 100 ],
    'mass' : [[1.8,2.2], 100],
    'y' : [[-3,3], 24],
    'VtxProb' : [[0,1],100],
    'VtxChi2' : [[0,1],100],
    'VtxNDF' : [[0,10],100],
    '3DCosPointingAngle' : [[-1.5,1.5],100],
    '3DPointingAngle' : [[-1.5,1.5],100],
    '2DCosPointingAngle' : [[-1.5,1.5],100],
    '2DPointingAngle' : [[-1.5,1.5],100],
    '3DDecayLengthSignificance' : [[0,5],100],
    '3DDecayLength' : [[0,2],100],
    '2DDecayLengthSignificance' : [[0,3],100],
    '2DDecayLength' : [[0,3],100],
    'massDaugther1' : [[1.5,3],100],
    'pTD1' : [[0,3],100],
    'EtaD1' : [[-3,3],24],
    'PhiD1' : [[-1*np.pi,np.pi],100],
    'VtxProbDaugther1' : [[0,1],100],
    'VtxChi2Daugther1' : [[0,8],100],
    'VtxNDFDaugther1' : [[0,0.5],100],
    '3DCosPointingAngleDaugther1' : [[-1*np.pi,np.pi],100],
    '3DPointingAngleDaugther1' : [[-1*np.pi,np.pi],100],
    '2DCosPointingAngleDaugther1' : [[-1*np.pi,np.pi],100],
    '2DPointingAngleDaugther1' : [[-1*np.pi,np.pi],100],
    '3DDecayLengthSignificanceDaugther1' : [[0,100],100],
    '3DDecayLengthDaugther1' : [[-10,10],100],
    '3DDecayLengthErrorDaugther1' : [[-10,10],100],
    '2DDecayLengthSignificanceDaugther1' : [[0,10],100],
    'zDCASignificanceDaugther2' : [[0,5],100],
    'xyDCASignificanceDaugther2' : [[0,5],100],
    'NHitD2' : [[0,40],40],
    'HighPurityDaugther2' : [[0,100],100],
    'pTD2' : [[0,2],100],
    'EtaD2' : [[-3,3],100],
    'PhiD2' : [[-np.pi,np.pi],100],
    'pTerrD1' : [[0,0.05],100],
    'pTerrD2' : [[0,0.05],100],
    'dedxHarmonic2D2' : [[0,0.2],100],
    'zDCASignificanceGrandDaugther1' : [[0,10],100],
    'zDCASignificanceGrandDaugther2' : [[0,10],100],
    'xyDCASignificanceGrandDaugther1' : [[0,10],100],
    'xyDCASignificanceGrandDaugther2' : [[0,10],100],
    'NHitGrandD1' : [[0,40],40],
    'NHitGrandD2' : [[0,40],40],
    'HighPurityGrandDaugther1' : [[0,5],100],
    'HighPurityGrandDaugther2' : [[0,5],100],
    'pTGrandD1' : [[0,10],100],
    'pTGrandD2' : [[0,10],100],
    'pTerrGrandD1' : [[0,0.2],100],
    'pTerrGrandD2' : [[0,0.2],100],
    'EtaGrandD1' : [[-3,3],100],
    'EtaGrandD2' : [[-3,3],100],
    'dedxHarmonic2GrandD1' : [[-0.1,0.1],100],
    'dedxHarmonic2GrandD2' : [[-0.1,0.1],100]
    #'cBin' : [[0,200], 200],
    #'nColl' : [[0, 450], 450],
    #'hiHF' : [[0,10000], 500],
    #'hiHFPlus' : [[0,10000], 500],
    #'hiHFMinus' : [[0,10000], 500],
    #'hiNtrk' : [[0,10000], 1000],
    #'hiNpixelTrk' : [[0,50000], 1000],
    #'hiEB' : [[0, 3000], 300],
    #'hiEE' : [[0, 3000], 300],
    #'global_muon' : [[0, 5], 3],
    #'matching' : [[-12, 12], 24],
    #'ParticleFlow' : [[0, 5], 5 ],
    #'norm_chi2' : [[0., 30], 30 ],
    #'norm_chi2_inner' : [[0., 20], 20 ],
    #'norm_chi2_bestTraker' : [[0., 20.], 20 ],
    #'local_chi2' : [[-1.0, 500], 10 ],
    #'trkKink' : [[0.0, 1500.0], 150 ],
    #'segment_comp' : [[0.0, 1.4], 70 ],
    #'n_Valid_hits' : [[0, 15], 15 ],
    #'n_Valid_hits_inner' : [[0, 35], 35 ],
    #'n_Valid_hits_bestTraker' : [[0, 30], 30 ],
    #'n_MatchedStations' : [[0, 10], 10 ],
    #'n_MatchedChamber' : [[0, 10], 10 ],
    #'dxy' : [[0, 12], 12 ],
    #'dz' : [[0, 30], 180 ],
    #'dz_new' : [[0, 30], 180 ],
    #'Valid_pixel' : [[0, 14], 14 ],
    #'tracker_layers' : [[0, 25], 25 ],
    #'pixel_layers' : [[0, 25], 25 ],
    #'validFraction' : [[0., 1.2], 60 ],
    #'Medium_muon' : [[0, 5], 5 ],
    #'Tight_muon' : [[0, 5], 5 ],
    #'Tight_muon_newdef' : [[0, 5], 5 ],
    #'nVtx' : [[0, 7], 7 ],
    #'glbTrkProb' : [[0.0, 1.], 1000 ],
    #'glbKink' : [[0.0, 5e+2], 25 ],
    #'localDist' : [[-1.0, 500], 100 ],
    #'chi2LocalPos' : [[-1.0, 2942.1863], 100 ],
    #'inner_station_badHits' : [[0, 8], 8 ],
    #'inner_station_Hits' : [[0, 8], 8 ],
    #'caloCompatibility' : [[0, 1], 100 ],
    #'trk_lambda' : [[-1.5, 1.5], 30 ],
    #'trk_lambda_err' : [[0, 0.2], 40 ],
    #'trk_inner_missing_hits' : [[0, 12], 12 ],
    #'trk_outer_missing_hits' : [[0, 18], 18 ],
    #'trk_num_loss_hits' : [[0, 10], 10 ],
    #'qoverp' : [[-0.45, 0.45], 36],
    #'qoverp_err' : [[0., 1.], 100 ],
    #'beta' : [[0.0, 0.0], 2],
    #'beta_err' : [[0., 0.], 2 ],
    #'jetPtRatio' : [[0.0, 0.0], 2 ],
    #'jetPtRel' : [[0.0, 0.], 2 ],
    #'ecalIso' : [[0.0, 110], 55 ],
    #'hcalIso' : [[0.0, 100], 50 ],
    #'pfSize' : [[2, 2], 1],
    #'pfAvg_corrected_hcalE' : [[0.,0.], 1],
    #'pfAvg_corrected_ecalE' : [[0.,0.], 1],
    #'pfAvg_raw_hcalE' : [[0.,0.], 1],
    #'pfAvg_raw_ecalE' : [[0.,0.], 1],
    #'pfVar_corrected_hcalE' : [[0.,0.], 1],
    #'pfVar_corrected_ecalE' : [[0.,0.], 1],
    #'pfVar_raw_hcalE' : [[0.,0.], 1],
    #'pfVar_raw_ecalE' : [[0.,0.], 1],
    #'dDxDz_sta1_DT' : [[-10,10], 200],
    #'dDyDz_sta1_DT' : [[-10,10], 200],
    #'dX_sta1_DT' : [[-10,10], 200],
    #'dY_sta1_DT' : [[-10,10], 200],
    #'pullDxDz_sta1_DT' : [[-10,10], 200],
    #'pullDyDz_sta1_DT' : [[-10,10], 200],
    #'pullX_sta1_DT' : [[-10,10], 200],
    #'pullY_sta1_DT' : [[-10,10], 200],
    #'segmentDxDz_sta1_DT' : [[-10,10], 200],
    #'segmentDxDzErr_sta1_DT' : [[-10,10], 200],
    #'segmentDyDz_sta1_DT' : [[-10,10], 200],
    #'segmentDyDzErr_sta1_DT' : [[-10,10], 200],
    #'segmentX_sta1_DT' : [[-10,10], 200],
    #'segmentXErr_sta1_DT' : [[-10,10], 200],
    #'segmentY_sta1_DT' : [[-10,10], 200],
    #'segmentYErr_sta1_DT' : [[-10,10], 200],
    #'trackDist_sta1_DT' : [[-10,10], 200],
    #'trackDistErr_sta1_DT' : [[-10,10], 200],
    #'trackDxDz_sta1_DT' : [[-10,10], 200],
    #'trackDxDzErr_sta1_DT' : [[-10,10], 200],
    #'trackDyDz_sta1_DT' : [[-10,10], 200],
    #'trackDyDzErr_sta1_DT' : [[-10,10], 200],
    #'trackEdgeX_sta1_DT' : [[-10,10], 200],
    #'trackEdgeY_sta1_DT' : [[-10,10], 200],
    #'trackX_sta1_DT' : [[-10,10], 200],
    #'trackXErr_sta1_DT' : [[-10,10], 200],
    #'trackY_sta1_DT' : [[-10,10], 200],
    #'trackYErr_sta1_DT' : [[-10,10], 200],
    #'dDxDz_sta2_DT' : [[-10,10], 200],
    #'dDyDz_sta2_DT' : [[-10,10], 200],
    #'dX_sta2_DT' : [[-10,10], 200],
    #'dY_sta2_DT' : [[-10,10], 200],
    #'pullDxDz_sta2_DT' : [[-10,10], 200],
    #'pullDyDz_sta2_DT' : [[-10,10], 200],
    #'pullX_sta2_DT' : [[-10,10], 200],
    #'pullY_sta2_DT' : [[-10,10], 200],
    #'segmentDxDz_sta2_DT' : [[-10,10], 200],
    #'segmentDxDzErr_sta2_DT' : [[-10,10], 200],
    #'segmentDyDz_sta2_DT' : [[-10,10], 200],
    #'segmentDyDzErr_sta2_DT' : [[-10,10], 200],
    #'segmentX_sta2_DT' : [[-10,10], 200],
    #'segmentXErr_sta2_DT' : [[-10,10], 200],
    #'segmentY_sta2_DT' : [[-10,10], 200],
    #'segmentYErr_sta2_DT' : [[-10,10], 200],
    #'trackDist_sta2_DT' : [[-10,10], 200],
    #'trackDistErr_sta2_DT' : [[-10,10], 200],
    #'trackDxDz_sta2_DT' : [[-10,10], 200],
    #'trackDxDzErr_sta2_DT' : [[-10,10], 200],
    #'trackDyDz_sta2_DT' : [[-10,10], 200],
    #'trackDyDzErr_sta2_DT' : [[-10,10], 200],
    #'trackEdgeX_sta2_DT' : [[-10,10], 200],
    #'trackEdgeY_sta2_DT' : [[-10,10], 200],
    #'trackX_sta2_DT' : [[-10,10], 200],
    #'trackXErr_sta2_DT' : [[-10,10], 200],
    #'trackY_sta2_DT' : [[-10,10], 200],
    #'trackYErr_sta2_DT' : [[-10,10], 200],
    #'dDxDz_sta1_CSC' : [[-10,10], 200],
    #'dDyDz_sta1_CSC' : [[-10,10], 200],
    #'dX_sta1_CSC' : [[-10,10], 200],
    #'dY_sta1_CSC' : [[-10,10], 200],
    #'pullDxDz_sta1_CSC' : [[-10,10], 200],
    #'pullDyDz_sta1_CSC' : [[-10,10], 200],
    #'pullX_sta1_CSC' : [[-10,10], 200],
    #'pullY_sta1_CSC' : [[-10,10], 200],
    #'segmentDxDz_sta1_CSC' : [[-10,10], 200],
    #'segmentDxDzErr_sta1_CSC' : [[-10,10], 200],
    #'segmentDyDz_sta1_CSC' : [[-10,10], 200],
    #'segmentDyDzErr_sta1_CSC' : [[-10,10], 200],
    #'segmentX_sta1_CSC' : [[-10,10], 200],
    #'segmentXErr_sta1_CSC' : [[-10,10], 200],
    #'segmentY_sta1_CSC' : [[-10,10], 200],
    #'segmentYErr_sta1_CSC' : [[-10,10], 200],
    #'trackDist_sta1_CSC' : [[-10,10], 200],
    #'trackDistErr_sta1_CSC' : [[-10,10], 200],
    #'trackDxDz_sta1_CSC' : [[-10,10], 200],
    #'trackDxDzErr_sta1_CSC' : [[-10,10], 200],
    #'trackDyDz_sta1_CSC' : [[-10,10], 200],
    #'trackDyDzErr_sta1_CSC' : [[-10,10], 200],
    #'trackEdgeX_sta1_CSC' : [[-10,10], 200],
    #'trackEdgeY_sta1_CSC' : [[-10,10], 200],
    #'trackX_sta1_CSC' : [[-10,10], 200],
    #'trackXErr_sta1_CSC' : [[-10,10], 200],
    #'trackY_sta1_CSC' : [[-10,10], 200],
    #'trackYErr_sta1_CSC' : [[-10,10], 200],
    #'dDxDz_sta2_CSC' : [[-10,10], 200],
    #'dDyDz_sta2_CSC' : [[-10,10], 200],
    #'dX_sta2_CSC' : [[-10,10], 200],
    #'dY_sta2_CSC' : [[-10,10], 200],
    #'pullDxDz_sta2_CSC' : [[-10,10], 200],
    #'pullDyDz_sta2_CSC' : [[-10,10], 200],
    #'pullX_sta2_CSC' : [[-10,10], 200],
    #'pullY_sta2_CSC' : [[-10,10], 200],
    #'segmentDxDz_sta2_CSC' : [[-10,10], 200],
    #'segmentDxDzErr_sta2_CSC' : [[-10,10], 200],
    #'segmentDyDz_sta2_CSC' : [[-10,10], 200],
    #'segmentDyDzErr_sta2_CSC' : [[-10,10], 200],
    #'segmentX_sta2_CSC' : [[-10,10], 200],
    #'segmentXErr_sta2_CSC' : [[-10,10], 200],
    #'segmentY_sta2_CSC' : [[-10,10], 200],
    #'segmentYErr_sta2_CSC' : [[-10,10], 200],
    #'trackDist_sta2_CSC' : [[-10,10], 200],
    #'trackDistErr_sta2_CSC' : [[-10,10], 200],
    #'trackDxDz_sta2_CSC' : [[-10,10], 200],
    #'trackDxDzErr_sta2_CSC' : [[-10,10], 200],
    #'trackDyDz_sta2_CSC' : [[-10,10], 200],
    #'trackDyDzErr_sta2_CSC' : [[-10,10], 200],
    #'trackEdgeX_sta2_CSC' : [[-10,10], 200],
    #'trackEdgeY_sta2_CSC' : [[-10,10], 200],
    #'trackX_sta2_CSC' : [[-10,10], 200],
    #'trackXErr_sta2_CSC' : [[-10,10], 200],
    #'trackY_sta2_CSC' : [[-10,10], 200],
    #'trackYErr_sta2_CSC' : [[-10,10], 200],
    #'dDxDz_sta1_RPC' : [[-10,10], 200],
    #'dDyDz_sta1_RPC' : [[-10,10], 200],
    #'dX_sta1_RPC' : [[-10,10], 200],
    #'dY_sta1_RPC' : [[-10,10], 200],
    #'pullDxDz_sta1_RPC' : [[-10,10], 200],
    #'pullDyDz_sta1_RPC' : [[-10,10], 200],
    #'pullX_sta1_RPC' : [[-10,10], 200],
    #'pullY_sta1_RPC' : [[-10,10], 200],
    #'segmentDxDz_sta1_RPC' : [[-10,10], 200],
    #'segmentDxDzErr_sta1_RPC' : [[-10,10], 200],
    #'segmentDyDz_sta1_RPC' : [[-10,10], 200],
    #'segmentDyDzErr_sta1_RPC' : [[-10,10], 200],
    #'segmentX_sta1_RPC' : [[-10,10], 200],
    #'segmentXErr_sta1_RPC' : [[-10,10], 200],
    #'segmentY_sta1_RPC' : [[-10,10], 200],
    #'segmentYErr_sta1_RPC' : [[-10,10], 200],
    #'trackDist_sta1_RPC' : [[-10,10], 200],
    #'trackDistErr_sta1_RPC' : [[-10,10], 200],
    #'trackDxDz_sta1_RPC' : [[-10,10], 200],
    #'trackDxDzErr_sta1_RPC' : [[-10,10], 200],
    #'trackDyDz_sta1_RPC' : [[-10,10], 200],
    #'trackDyDzErr_sta1_RPC' : [[-10,10], 200],
    #'trackEdgeX_sta1_RPC' : [[-10,10], 200],
    #'trackEdgeY_sta1_RPC' : [[-10,10], 200],
    #'trackX_sta1_RPC' : [[-10,10], 200],
    #'trackXErr_sta1_RPC' : [[-10,10], 200],
    #'trackY_sta1_RPC' : [[-10,10], 200],
    #'trackYErr_sta1_RPC' : [[-10,10], 200],
    #'dDxDz_sta2_RPC' : [[-10,10], 200],
    #'dDyDz_sta2_RPC' : [[-10,10], 200],
    #'dX_sta2_RPC' : [[-10,10], 200],
    #'dY_sta2_RPC' : [[-10,10], 200],
    #'pullDxDz_sta2_RPC' : [[-10,10], 200],
    #'pullDyDz_sta2_RPC' : [[-10,10], 200],
    #'pullX_sta2_RPC' : [[-10,10], 200],
    #'pullY_sta2_RPC' : [[-10,10], 200],
    #'segmentDxDz_sta2_RPC' : [[-10,10], 200],
    #'segmentDxDzErr_sta2_RPC' : [[-10,10], 200],
    #'segmentDyDz_sta2_RPC' : [[-10,10], 200],
    #'segmentDyDzErr_sta2_RPC' : [[-10,10], 200],
    #'segmentX_sta2_RPC' : [[-10,10], 200],
    #'segmentXErr_sta2_RPC' : [[-10,10], 200],
    #'segmentY_sta2_RPC' : [[-10,10], 200],
    #'segmentYErr_sta2_RPC' : [[-10,10], 200],
    #'trackDist_sta2_RPC' : [[-10,10], 200],
    #'trackDistErr_sta2_RPC' : [[-10,10], 200],
    #'trackDxDz_sta2_RPC' : [[-10,10], 200],
    #'trackDxDzErr_sta2_RPC' : [[-10,10], 200],
    #'trackDyDz_sta2_RPC' : [[-10,10], 200],
    #'trackDyDzErr_sta2_RPC' : [[-10,10], 200],
    #'trackEdgeX_sta2_RPC' : [[-10,10], 200],
    #'trackEdgeY_sta2_RPC' : [[-10,10], 200],
    #'trackX_sta2_RPC' : [[-10,10], 200],
    #'trackXErr_sta2_RPC' : [[-10,10], 200],
    #'trackY_sta2_RPC' : [[-10,10], 200],
    #'trackYErr_sta2_RPC' : [[-10,10], 200]
}



def getFile(_data_, _res_):
    catalogue = json.load(open('../data/input.json'))
    return catalogue[0][_data_][_res_]


def getNormalizationMethod(varName, **kwargs):
    return kwargs[varName]
class ROOTDataset(torch.utils.data.IterableDataset):
    def __init__(self, file_path, tree_name, variables,batch_size):
        self.file_path = file_path
        self.tree_name = tree_name
        self.variables = variables
        self.batch_size = batch_size
    def __iter__(self):
        for data_chunk in uproot.iterate(f"{self.file_path}:{self.tree_name}", self.variables, library="np"):
            flattened_data = {var: np.concatenate(data_chunk[var]) for var in self.variables}
            # flattened_data = np.array(list(flattened_data.values())).T
            data_chunk_tensor = None 
            yield data_chunk_tensor,flattened_data
class ROOTDatasetForXGB(torch.utils.data.IterableDataset):
     def __init__(self, file_path, tree_name, variables, batch_size):
         self.file_path = file_path
         self.tree_name = tree_name
         self.variables = variables
         self.batch_size = batch_size
         self.scaler = None

     def __iter__(self):
         for data_chunk in uproot.iterate(f"{self.file_path}:{self.tree_name}", self.variables, library="np", step_size=self.batch_size):
             data_original = data_chunk
             data_chunk = np.array(list(data_chunk.values()) ).T
             data_chunk_tensor = None
             if self.scaler is not None :
                 data_chunk_s = self.scaler.fit_transform(data_chunk)
                 data_chunk_tensor = torch.tensor(data_chunk_s, dtype=torch.float32)
             yield data_chunk_tensor, data_original