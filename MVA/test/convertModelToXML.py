import xgboost as xgb
from xgb2tmva import *
import sys
sys.path.insert(0, '../lib')
from treeIO import varData, varDataTraining


model = xgb.XGBClassifier()
model.load_model("xgb_model.json")
bst = model.get_booster()

model = bst.get_dump()

print(len(varDataTraining))
var = [('f' + str(i), 'F') for i in range(173)]
print (var)
convert_model(model, input_variables=var, output_xml ='xgboost.xml')
