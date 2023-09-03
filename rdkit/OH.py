#compare a range of OH containing molecules with the FTIR wavenumber shift range of 3700-3200 cm-1

#import the modules
from rdkit.Chem import Descriptors
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import tensorflow as tf
import pandas as pd

#OH containing molecules

water =Chem.MolFromSmiles("O")
MPG=Chem.MolFromSmiles("CC(CO)O")
BG=Chem.MolFromSmiles("CCCCOCCO")
glycerine=Chem.MolFromSmiles("C(C(CO)O)O")
benzylOH=Chem.MolFromSmiles("C1=CC=C(C=C1)CO")
MEG=Chem.MolFromSmiles("C(CO)O")
DEG=Chem.MolFromSmiles("C(COCCO)O")
methanol=Chem.MolFromSmiles("CO")

#takes all the descriptors for all the molecules

def getMolDescriptors(mol, missingVal=None):
    ''' calculate the full list of descriptors for a molecule

        missingVal is used if the descriptor cannot be calculated
    '''
    res = {}
    for nm, fn in Descriptors._descList:
        # some of the descriptor fucntions can throw errors if they fail, catch those here:
        try:
            val = fn(mol)
        except:
            # print the error message:
            import traceback
            traceback.print_exc()
            # and set the descriptor value to whatever missingVal is
            val = missingVal
        res[nm] = val
    return res


water=getMolDescriptors(water)
MPG=getMolDescriptors(MPG)
BG=getMolDescriptors(BG)
glycerine=getMolDescriptors(glycerine)
benzylOH=getMolDescriptors(benzylOH)
MEG=getMolDescriptors(MEG)
DEG=getMolDescriptors(DEG)
methanol=getMolDescriptors(methanol)

#create a pandas dataframe of water MPG BG glycerine benzylOH MEG DEG methanol and name the rows water MPG BG glycerine benzylOH MEG DEG methanol
df = pd.DataFrame([water,MPG,BG,glycerine,benzylOH,MEG,DEG,methanol], index=['water','MPG','BG','glycerine','benzylOH','MEG','DEG','methanol'])
print(df)

#transpose the dataframe
dfT=df.transpose()
print(dfT)