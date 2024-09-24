#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#this module compares the models from the HHblits
#requires a directory with files in txt output format (any output format can be used as long as compatible with the script
#and all of the files have the same extension)
#chooses models with an e-value lower than 1e-8 and probability higher than 97.9%
#if the direct output from HHsuite is used then the first 8 lines need to be removed before running this script (the header)
#in addition to the compare module it compares if the specific pHMM is in the RepLysin66 list or in the student t-test list 

import pandas as pd 
import os, sys

#for every file in the folder map the protein names and swap them 
def Compare_hhblits(folder_path, nodefile, edgefile, replysinfile, tstudentfile):
    os.chdir(folder_path)
    files = [file_path for file_path in os.listdir(folder_path) if file_path.endswith('.txt')]
    data = pd.DataFrame([])
    replysin_df = pd.read_csv(replysinfile, header = None, sep ='\t')
    replysin_df.columns = ['pHMM_66']
    replysin = replysin_df['pHMM_66'].tolist()
    tstudent_df = pd.read_csv(tstudentfile, header = None, sep ='\t')
    tstudent_df.columns = ['pHMM_137']
    tstudent = tstudent_df['pHMM_137'].tolist()
    with open(nodefile, mode='w') as handle: 
        for filename in files:
            df = pd.read_csv(filename, sep = '\t') 
            df[['E-value', 'P-value', 'Score']] = df[['E-value', 'P-value', 'Score']].astype(float)
            relevant_e_value = df['E-value'] < 0.00000001
            relevant_probability = df['Prob'] > 97.9
            df_relevant = df[relevant_e_value]
            df_relevant = df[relevant_probability]
            #the filename is the pHMM to which all the hits are shown, so for the edgefile it becomes the Query column 
            qstr = str(filename).split('.')[0]
            querystring = qstr.replace("_", ":", 1)
            df_relevant['Query'] = querystring
            df_relevant['Hit'] = df_relevant['Hit'].str.replace("_", ":", 1)
            #the color column enables different coloring in Cytoscape
            df_relevant['color'] = 'nocolor'
            for idx, row in df_relevant.iterrows():
                if df_relevant['Hit'] in replysin:
                    df_relevant.at[idx, 'color'] = 'pos66'
                elif df_relevant['Hit'] in tstudent:
                    df_relevant.at[idx, 'color'] = 'pos137'
            hmmer_tbl = df_relevant[['Query', 'Hit', 'Prob', 'color']]
            data = data.append(hmmer_tbl, ignore_index = True)
            handle.write(querystring+'\n')
        data.to_csv(edgefile, index= None, sep = '\t', mode = 'a+')
    handle.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a folder with txt HHsuite output files')
    parser.add_argumet("folder_path", type=str, help="The folder path where the the HHsuite output files are kept.")
    parser.add_argument("nodefile", type=str, help="The node file with a list pHMMs used as an input in Cytoscape.")
    parser.add_argument("edgefile", type=str, help="The edge file with the hit and query columns and the probability to create the edges in Cytoscape.")
    parser.add_argument("replysinfile", type=str, help="The file with the pHMMs in the RepLysin66 set")
    parser.add_argument("tstudentfile", type=str, help="The file with the pHMMs from the student t-test")
    args = parser.parse_args()
    print(f"Arguments: {args}")
    print(f"Received directory to parse: {args.folder_path}")
    Compare_hhblits(args.folder_path, args.nodefile, args.edgefile, args.replysinfile, args.tstudentfile)    

