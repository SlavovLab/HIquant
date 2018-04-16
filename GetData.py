import numpy as np
import pandas as pd
import os
import sys

os.getcwd()

print( 'Importing data from the directory below ...')
# Define function to get the input and parse it.
def GetData ( path ):
    if len(sys.argv) < 1:
        print (' You have to input the absolute Path of your file! ')
    else:
        Path=path
        print ('The file path you inputed is : \n', Path)
        dat=pd.read_csv(Path,sep='\t')
        return dat

# /Users/chentianchi/Desktop/test/current/data/hiquant_getdata_test/HIquant_test_alex/alex_kea_hiquant/paralogs_extract.txt
test_path='/Users/chentianchi/Desktop/test/current/data/hiquant_getdata_test/HIquant_test_alex/IFN/all_proteoform/paralogs_extract.txt'
a=GetData(test_path)
