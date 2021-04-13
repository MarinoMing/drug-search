#!/usr/bin/python
# -*- coding:utf-8 -*-



"""
这里算对应文件的MACCS指纹

from rdkit import Chem
# from rdkit.Chem import AllChem

from rdkit.Chem import MACCSkeys

molecule=Chem.MolFromSmiles("SC[C@H]1CC(=O)N2N([C@@H](CCC2)C(=O)O)C1=O") #此处输入smile文件
fingerPrints=MACCSkeys.GenMACCSKeys(molecule)
print(len(fingerPrints))
print(fingerPrints.ToBitString())

"""


"""
输出ECF6指纹的方法



from rdkit import DataStructs
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

mols=[]
m=Chem.MolFromSmiles('CCC(=O)CC')
# fps=Chem.RDKFingerprint(m)
# fps= AllChem.GetMorganFingerprint(m,3,2048)
fps=AllChem.GetMorganFingerprintAsBitVect(m,3,2048)
print(len(fps))
print(fps.ToBitString())
# a=fps.ToBitString()
# print(len(a))


# smo1=DataStructs.FingerprintSimilarity(fps[0],fps[1])
# print(smo1)

"""


