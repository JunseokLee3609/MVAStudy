import numpy as np

def ClipAndNorm(min, max):
    return  lambda arg : np.divide(np.clip(arg, min, max), (max-min))

def DivideWithOffset(val, offset):
    return lambda arg : np.divide(arg+offset, val)

def Pass():
    return lambda arg : arg

def NormSet():
    normalizationMap = {
        "pt" : ClipAndNorm(0,50),
        "eta" : DivideWithOffset(4.8, 2.4),
        'global_muon' : Pass(),
        'matching' : Pass(),
        'ParticleFlow' : Pass(),
        'norm_chi2' : Pass(),
        'norm_chi2_inner' : Pass(),
        'norm_chi2_bestTraker' : Pass(),
        'local_chi2' : Pass(),
        'trkKink' : Pass(),
        'segment_comp' : Pass(),
        'n_Valid_hits' : Pass(),
        'n_Valid_hits_inner' : Pass(),
        'n_Valid_hits_bestTraker' : Pass(),
        'n_MatchedStations' : Pass(),
        'dxy' : Pass(),
        'dz' : Pass(),
        'dz_new' : Pass(),
        'Valid_pixel' : Pass(),
        'tracker_layers' : Pass(),
        'validFraction' : Pass(),
        'Medium_muon' : Pass(),
        'Tight_muon' : Pass(),
        'Tight_muon_newdef' : Pass(),
        'nVtx' : Pass(),
        'glbTrkProb' : Pass(),
        'glbKink' : Pass(),
        'localDist' : Pass(),
        'chi2LocalPos' : Pass(),
        'inner_station_badHits' : Pass(),
        'inner_station_Hits' : Pass(),
        'caloCompatibility' : Pass(),
        'trk_lambda' : Pass(),
        'trk_lambda_err' : Pass(),
        'trk_inner_missing_hits' : Pass(),
        'trk_outer_missing_hits' : Pass(),
        'trk_num_loss_hits' : Pass(),
        'qoverp' : Pass(),
        'qoverp_err' : Pass(),
        'beta' : Pass(),
        'beta_err' : Pass(),
        'jetPtRatio' : Pass(),
        'jetPtRel' : Pass(),
        'ecalIso' : Pass(),
        'hcalIso' : Pass(),
        'pfSize' : Pass(),
        'pfAvg_corrected_hcalE' : Pass(),
        'pfAvg_corrected_ecalE' : Pass(),
        'pfAvg_raw_hcalE' : Pass(),
        'pfAvg_raw_ecalE' : Pass(),
        'pfVar_corrected_hcalE' : Pass(),
        'pfVar_corrected_ecalE' : Pass(),
        'pfVar_raw_hcalE' : Pass(),
        'pfVar_raw_ecalE' : Pass(),
    }
    return normalizationMap