#!/usr/bin/env python
# coding: utf-8

# ### Python interface for integrated model
## ============================================================== ;;;
## INTEGRATED DECLARATIVE/PROCEDURAL MODEL INTERFACE
## ============================================================== ;;;
## - This interface presents stimuli and feedback to the model and collects
##   responses.
## - It performs analysis of the simulated data and compiles results
##   across simulations in to the variable sim_data for export.
## - This version differes from integrated_model_interface.py in that
##   it utilizes a parameter for explicitly specifying
##   a strategy(Reinforcement learning OR WM/LTM) for each trial.
##
##
## ============================================================== ;;;
## Change log:
##
##
## 12-2020 - Fixed a bug that was preventing means of all simulations
##           from being saved for test data.
## 09-2023 - Updated model_loop() to include the new schedule_event()
##           functionality to schedule all stimulus presentations.
##           This replaces the relative schedules in present_feedback()
##           function.
##         - Added call to the spacing effects module.
##         - Added the new stimulus information about block identitiy to
##           present_stim() function.
##         - Updated parameters to work with the MAS (replacing imag) and
##           se, rate of forgetting, which replaces BLL.
##         - Updated code so that the random distribution of strategies,
##           RL or LTM is shuffled with in a simulaiton.
##
##
## ============================================================== ;;;


## ============================================================== ;;;
##   To execute in python terminal enter:
##      1) "run integrated_model_interface.py"
##      2) "run_simulation(mas, alpha, egs, se, ans, nSims)" with
##         parameters specificed.
##
## ============================================================== ;;;


import random as rnd
import numpy as np
import os
import sys
import string
import actr
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
import itertools


show_output = False

#Load model
curr_dir = os.path.dirname(os.path.realpath(__file__))
actr.load_act_r_model(os.path.join(curr_dir, "spacing-effect.lisp")) #### TMH 09-2023
actr.load_act_r_model(os.path.join(curr_dir, "strategy-integrated-model_2023.lisp")) #integrated-model.lisp

## ==============================================================
## Daisy chained python functions to present stimuli, get response
## and  present feedback.
## ==============================================================

def present_stim():
    global chunks
    global stims
    global i
    global show_output
    global current_strategy

    if i < nTrials:

       #### For this model, a strategy parameter is used.
        #print(current_strategy)


        chunks = actr.define_chunks(['isa', 'stimulus',
            'picture', stims[i],
            'do-strategy', str(current_strategy[i]),
            'block_ID', block_ID[i]]) ## TMH 09-2023

        actr.set_buffer_chunk('visual', chunks[0])
        if(show_output):
            print('Presented: ', stims[i])
            print('correct response: ', cor_resps[i])


def get_response(model, key):
    global current_response
    global i

    actr.schedule_event_relative(0, 'present_feedback')

    current_response[i] = key

    #return current_response


def present_feedback():
    global i
    global current_response
    global accuracy

    if i > lastLearnTrial:
    # this tests whether the current trial is a test phase. This portion presents meaningless feedback and checks accuracy
        if current_response[i] == cor_resps[i]:
            accuracy[i] = 1

        feedback = "x"
        chunks = actr.define_chunks(['isa', 'feedback', 'feedback',feedback])
        actr.set_buffer_chunk('visual', chunks[0])

       # actr.schedule_event_relative(1, 'present_stim') taken over by schedule event #### TMH 09-2023

        if (show_output):
            print("Feedback given: X, test phase" )
            print(accuracy)


    else:
    # This runs for learning phase. This portion presents feedback
        feedback = 'no'

    # check if response matches the appropriate key for the current stimulus in cue
        if current_response[i] == cor_resps[i]:
            feedback = 'yes'
            accuracy[i] = 1
    # present feedback
        chunks = actr.define_chunks(['isa', 'feedback', 'feedback',feedback])
        actr.set_buffer_chunk('visual', chunks[0])

        if (show_output):
            print("Feedback given: ", feedback )
            print(accuracy)

        #if i == lastLearnTrial:                               ## TMH 09-2023
            #print("BREAK HERE")                               ##
         #   actr.schedule_event_relative(600, 'present_stim') ##
        #else:                                                 ##
         #   actr.schedule_event_relative(1, 'present_stim')   ## Entire block replaced by schedule event below
#increase index for next stimulus
    i = i + 1



##### This function builds ACT-R representations of the python functions

def model_loop():

    global win
    global accuracy
    global nTrials
    global t #TMH 09-2023
    global i

    accuracy = np.repeat(0, nTrials).tolist()



    #initial goal dm
    actr.define_chunks(['make-response','isa', 'goal', 'fproc','yes'])

    actr.goal_focus('make-response')

    #open window for interaction
    win = actr.open_exp_window("test", visible = False)
    actr.install_device(win)
    #actr.schedule_event_relative(0, 'present_stim' )

   ########## This is the new absolute time for stimulus presentaitons! TMH 09-2023

    for t in range(nTrials):


        if t >= lastLearnTrial:

            actr.schedule_event(((2*t) + 1680), 'present_stim')

        else:

            actr.schedule_event(2 * t, 'present_stim')


    actr.run(2000) #2000


## ==============================================================
## Set up experiment
## ==============================================================


actr.add_command('present_stim', present_stim, 'presents stimulus')
actr.add_command('present_feedback', present_feedback, 'presents feedback')
actr.add_command('get_response', get_response, 'gets response')
actr.monitor_command("output-key", 'get_response')

## execute model and simulate data

#Stimuli to be used and exp parameters
stims_3 = ['cup','bowl','plate']
stims_6 = ['hat','gloves','shoes', 'shirt', 'jacket', 'jeans']
test_stims = ["bowl", "shirt", "jeans", "plate", "cup", "jacket"] #Each stimulus was presented 4 times during test
nPresentations = 12 #learning phase, items were presented 12-14 times during learning
nTestPresentations = 4
nTrials = (nPresentations * 9) + (nTestPresentations * np.size(test_stims))  #3 #for sets size three experiment/block
accuracy =  accuracy = np.repeat(0, nTrials).tolist()

#associated responses (matches Collins' patterns of response-stim associations)
stims_3_resps = ['j', 'j', 'l']
stims_6_resps = ['k','k', 'j', 'j', 'l', 'l']
test_resps    = ["j", "j", "l", "l", "j", "l"]

#generate stimult to present to model **Edit as needed **

#this shuffles both lists, stimuli and associated correct responses, in the same order

# 3 set block
stims_temp3 = list( zip(np.repeat(stims_3, 12).tolist(),
         np.repeat(stims_3_resps,12).tolist()
        ))

rnd.shuffle(stims_temp3)

stims3, cor_resps3 = zip(*stims_temp3)

# 6 set block
stims_temp6 = list( zip(np.repeat(stims_6, 12).tolist(),
    np.repeat(stims_6_resps, 12).tolist()
    ))
rnd.shuffle(stims_temp6)
stims6, cor_resps6 = (zip(*stims_temp6))


# test phase
test_temp = list(zip(np.repeat(test_stims, 4).tolist(),
np.repeat(test_resps, 4).tolist()
  ))

rnd.shuffle(test_temp)
teststims, cor_testresps = (zip(*test_temp))

# concat all stimuli and responses together to present

stims = stims3 + stims6 + teststims
cor_resps = cor_resps3 + cor_resps6 + cor_testresps

#variables needed
chunks = None
current_response  = np.repeat('x', nTrials * 2).tolist() #multiply by 2 for number of blocks
lastLearnTrial = np.size(stims3 + stims6) -1

# --------TMH - 09-2023 --- New information about blocks to present to model. This should increase fan effect for set-size 6

block_ID_temp = np.append(np.repeat('set_3_block', len(stims3)),np.repeat('set_6_block', len(stims6)))
block_ID = np.append(block_ID_temp, np.repeat('test', len(teststims)))


## ==============================================================
## set up model parameters
## ==============================================================


#parameter ranges for simulation

#se_param = [0.24,0.25, 0.26, 0.27, 0.28,.29, 0.3,.31, 0.32,.33, 0.34,.35, 0.36, .37, .38] #spacing effect parameter for rate of forgetting TMH 09-2023
se_param = [.22, .23] #.20,.21, 
mas_param = [1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2] #MAS parameter TMH 09-2023
alpha_param = [0.05, 0.1, 0.15, 0.2, 0.25] # learning rate of the RL utility selection 0.2 rec val
egs_param   = [0.1, 0.2, 0.3, 0.4, 0.5] # amount of noise added to the RL utility selection
ans_param   = [0.1, 0.2, 0.3, 0.4, 0.5] #parameter for noise in dec. memory activation. Range recommended by ACTR manual.
strtg_param   = ['RL20', 'RL40', 'RL60', 'RL80'] # this is the strategy parameter - proportion of decl/proced to use.

### Replaced parameters TMH - 09-2023:
#bll_param   = [0.3, 0.4, 0.5, 0.6, 0.7]  ## replaced by spacing effects, se_param which is alpha, rate of forgetting.
#imag_param  = [0.1, 0.2, 0.3 , 0.4, 0.5] #  replaced by MAS parameter above

#combine all params for a loop
params = [mas_param, alpha_param, egs_param, se_param, ans_param, strtg_param]
param_combs = list(itertools.product(*params))



 ###########initialize variables to concat all outputs from simulations

sim_data3 = [] #saves mean curves and parameters
sim_data6 = []
sim_data  = []
I_data = []
sim_std=[]
current_strategy = [];






## ==============================================================
## simulator and data analysis
## ==============================================================

  ## function inputs updated to reflect new parameters - THM - 09-2023
def simulation(mas, alpha, egs, se, ans, strtg, nSims):

    global i
    global sim_data3
    global sim_data
    global sim_data6
    global accuracy
    global current_strategy
    global sim_std
    global RL20
    global RL40
    global RL60
    global RL80
    global ex_ct


## This was moved down here
## ==============================================================
## set up strategy distributions
## ==============================================================

    RL20 = np.random.permutation(
        np.concatenate(
        [np.repeat(1,round(132*0.20)) ,
        np.repeat(2, round(132 * 0.8))]))

    RL40 = np.random.permutation(
        np.concatenate(
        [np.repeat(1,round(132*0.4)) ,
        np.repeat(2, round(132 * 0.6))]))

    RL60 = np.random.permutation(
        np.concatenate(
        [np.repeat(1,round(132*0.6)) ,
        np.repeat(2, round(132 * 0.4))]))

    RL80 = np.random.permutation(
        np.concatenate(
        [np.repeat(1,round(132*0.8)) ,
        np.repeat(2, round(132 * 0.20))]))


    current_strategy = eval(strtg)
    #print(current_strategy)

    temp3 = []
    temp6 = []
    temp_test3 = []
    temp_test6 = []
    #accuracy = np.repeat(0, nTrials).tolist()
    nsimulations = np.arange(nSims) #set the number of simulations "subjects"
    for n in nsimulations:
        
            
        actr.reset()
        #actr.hide_output()
        
        actr.set_parameter_value(":mas", mas)
        actr.set_parameter_value(":alpha", alpha)
        actr.set_parameter_value(":egs", egs)
        actr.set_parameter_value(":se-intercept", se)#formerly imaginal activation
        actr.set_parameter_value(":ans", ans)

        i = 0
        t = 0
        win = None
        model_loop()


       ### Analyze generated data: LEARNING
            ##set 3 analysis

        stims_array = np.asarray(stims[0:lastLearnTrial + 1])
        acc_array   = np.asarray(accuracy[0:lastLearnTrial + 1])

        cup_presented   = np.where(stims_array == 'cup')
        bowl_presented  = np.where(stims_array == 'bowl')
        plate_presented = np.where(stims_array == 'plate')

        acc3 = np.mean([acc_array[cup_presented], acc_array[plate_presented], acc_array[bowl_presented]],0)

               ##set 6 analysis


        hat_presented    = np.where(stims_array == 'hat')
        gloves_presented = np.where(stims_array == 'gloves')
        shoes_presented  = np.where(stims_array == 'shoes')
        shirt_presented  = np.where(stims_array == 'shirt')
        jacket_presented = np.where(stims_array == 'jacket')
        jeans_presented  = np.where(stims_array == 'jeans')

        acc6 = np.mean([acc_array[hat_presented],
            acc_array[gloves_presented],
            acc_array[shoes_presented],
            acc_array[shirt_presented],
            acc_array[jacket_presented],
            acc_array[jeans_presented]], 0)

        temp3.append(acc3)
        temp6.append(acc6)

            ### Analyze generated data:  TESTING PHASE

        test_array = np.asarray(stims[lastLearnTrial+1 : np.size(stims)])
        test_acc_array   = np.asarray(accuracy[lastLearnTrial+1 : np.size(stims)])


        cup_presented_t   = np.where(test_array == 'cup')
        bowl_presented_t  = np.where(test_array == 'bowl')
        plate_presented_t = np.where(test_array == 'plate')

        hat_presented_t    = np.where(test_array == 'hat')
        gloves_presented_t = np.where(test_array == 'gloves')
        shoes_presented_t  = np.where(test_array == 'shoes')
        shirt_presented_t  = np.where(test_array == 'shirt')
        jacket_presented_t = np.where(test_array == 'jacket')
        jeans_presented_t  = np.where(test_array == 'jeans')

        test_3 = np.mean([test_acc_array[cup_presented_t], test_acc_array[plate_presented_t], test_acc_array[bowl_presented_t]],0)

        test_6 = np.mean([
            test_acc_array[shirt_presented_t],
            test_acc_array[jacket_presented_t],
            test_acc_array[jeans_presented_t]], 0)
       # print(temp3)

        #aggregate accuracies across simulations
        temp_test3.append(test_3)
        temp_test6.append(test_6)

       # print('accuracy ', np.mean(accuracy))
            #pyplot.figure(dpi=120)
            #sns.barplot(x=["set 3", "set 6"], y=[np.mean(test_3),np.mean(test_6)])

        if False:
            #save data to files
            f3 = open("set3.csv", 'a')
            f6 = open("set6.csv", 'a')
            acc6.transpose().to_csv(f6, mode='a', header = False)
            acc3.transpose().to_csv(f3, mode='a', header = False)
            f3.close()
            f6.close()
        I_data.append(i)



#                   save averaged resluts from simulations along with parameters

        #sim_data.append([temp3, temp6, np.mean(test_3), np.mean(test_6), mas, alpha, egs, se, ans ])
        #del temp3, temp6
    #changelog: saving all instances of the simulation by moving the sim_data insidr the simulator loop
    sim_data.append([np.mean(temp3,0), np.mean(temp6,0), np.mean(np.mean(temp_test3,1)), np.mean(np.mean(temp_test6, 1)),
     mas, alpha, egs, se, ans, strtg])
    #grab stds for distribution
    #sim_std.append([np.std(temp3,0), np.std(temp6,0), np.std(np.mean(temp_test3,1)), np.std(np.mean(temp_test6, 1))])
    #sim_data3 = temp_test3

   # return sim_data
#sum(np.array(pd.DataFrame(I_data)<132))
def execute_sim(n,fromI,toI, frac):
    global ex_ct
    ex_ct = 1;
    for i in range(fromI, toI):
        print(ex_ct)
        simulation(param_combs[i][0], param_combs[i][1],param_combs[i][2], param_combs[i][3], param_combs[i][4],param_combs[i][5], n)
        ex_ct +=1
    sim = pd.DataFrame(sim_data, columns=['set3_learn','set6_learn', 'set3_test', 'set6_test','mas', 'alpha', 'egs', 'se', 'ans','strtg' ])
    #sim_st = pd.DataFrame(sim_std, columns=['set3_learn','set6_learn', 'set3_test', 'set6_test'])

    #sim_st.to_pickle('./sims/STR_std_data_' + 'frac_' +np.str(frac) +'_'+ np.str(fromI) + '_to_' + np.str(toI))
    sim.to_pickle('./sims/STR_sim_data_' + 'frac_' +str(frac) +'_'+ str(fromI) + '_to_' + str(toI))
    #sim.to_json('./sims/STR_sim_data_' + 'frac_' +np.str(frac) +'_'+ np.str(fromI) + '_to_' + np.str(toI) + '.json', orient='table')
