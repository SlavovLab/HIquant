Sets = ['SET_2/TMT2_MQ_Phospho/', 'SET_1/TMT6_4AA/', 'TMT8/all_MQ/', 'TMT8/Fraction_1_20/phospho_mascot_FDR2/', 'TMT8/Fraction_21_40/phospho_mascot_FDR2/' 'TMT8/Fraction_1_20/phospho_mascot_FDR2/Isolation_Inteference/', 'TMT8/Fraction_21_40/phospho_mascot_FDR2/Isolation_Inteference/', 'SET_1/TMT6_4AA_MQ/'  'SET_2/TMT2_PD_Methyl/' 'SET_2/TMT2_PD/' 'SET_2/TMT2_PD_semitryptic/']
file_name = 'paralogs_extract.txt'
location1 = '/Volumes/IN/research/'
location2 = '/Users/chentianchi/Desktop/test/current/data/'


keys = [Sets[i].split('/')[1] for i in range(len(Sets))]
values = [location2 + Sets[i] + file_name for i in range(len(Sets))]
test_set = dict(zip(keys, values))

# a working one with alex Data
#test_path='/Users/chentianchi/Desktop/test/current/data/hiquant_getdata_test/HIquant_test_alex/IFN/all_proteoform/paralogs_extract.txt'
