from rdkit.Chem import Descriptors

from rdkit import Chem
from rdkit.Chem import Descriptors


cyrene="C1CC(=O)C2OCC1O2"
NMP="CN1CCCC1=O"


#make a list of the molecules
molecules=[cyrene,NMP]
print(molecules)


#make a list of the names of the molecules
molecule_names=["cyrene","NMP"]
print(molecule_names)

#make a list of the molecular weights of the molecules
molecular_weights=[Descriptors.ExactMolWt(Chem.MolFromSmiles(molecule)) for molecule in molecules]
print(molecular_weights)

#make a list of the logP values of the molecules
logP=[Descriptors.MolLogP(Chem.MolFromSmiles(molecule)) for molecule in molecules]
print(logP)

#make a list of the number of hydrogen bond donors of the molecules
num_hbd=[Descriptors.NumHDonors(Chem.MolFromSmiles(molecule)) for molecule in molecules]
print(num_hbd)

#make a list of the number of hydrogen bond acceptors of the molecules
num_hba=[Descriptors.NumHAcceptors(Chem.MolFromSmiles(molecule)) for molecule in molecules]
print(num_hba)


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


a = getMolDescriptors(cyrene)
print(a)



