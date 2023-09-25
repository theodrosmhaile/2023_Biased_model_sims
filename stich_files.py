#!/usr/bin/env python
# coding: utf-8

# Process pickle files

import pandas as pd 
import numpy as np
import glob
study = 'strategy_model/'
loc = './sims/str/' 
files =  glob.glob(loc + 'STR*')#'STR*'
files.sort()





def stc(files):
	
	temp_list = pd.DataFrame( columns=['set3_learn','set6_learn', 'set3_test', 'set6_test','se', 'alpha', 'egs', 'mas', 'ans' ,'strtg' ])

	for filename in files:
		temp_df =  pd.read_pickle(filename) 

		temp_list = pd.concat([temp_list, temp_df], ignore_index=True)
	
	return temp_list

helldat=stc(files)
# def concat(files, locs):

# 	if len(files) == 1:
# 		return pd.read_pickle(loc + files[0])

# 	else:
# 		return pd.DataFrame.append(pd.read_pickle(loc + files[0]), concat(files[len(files)-1], loc))
