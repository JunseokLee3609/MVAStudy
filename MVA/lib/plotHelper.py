import matplotlib.pyplot as plt
import mplhep as hep

def DrawVarHist(histArr):
    return lambda var : hep.histplot(histArr[var][0], histArr[var][1])
