'''
  File:            summs_utils.py

  About
  ------------------
  utility functions for data processing and related tasks for summary Markov & related models

'''

import numpy as np
import math
import random
import os
import ast
import pickle
from itertools import product
import SuMM

#from xml.dom import minidom


### DATA PROCESSING


# convert time-stamped event stream(s) to event sequence(s) (multiple streams)
def stream_to_seq(D_data):

    if type(D_data) == dict:
        D_seq_data = {}
        for id in D_data:
            D = D_data[id]
            D_seq_data[id] = [event[1] for event in D]
    else:
        D_seq_data = {}

    return D_seq_data


# function recovers sorted unique set of event labels from event sequence data (multiple streams)
def get_unique_labels(D_seq_data):
    '''
    :param D_seq_data: event sequence dataset: dict of lists (l_i)
    :return: L: label set
    '''

    L = []
    for id in D_seq_data:
        D = D_seq_data[id]
        if D:
            L += sorted(list(set(D)))
    L = list(set(L))

    return L


# function restricts a multiple event sequence dataset to a set of specified event labels
def restrict_data_event_seq(D_dict, label_list):
  '''
  Inputs
  ------
  D_dict: multiple event sequence dataset, dict of lists
  label_list: list of labels of interest
  '''

  restricted_D_dict = {}

  for id in sorted(D_dict):
    D = D_dict[id]

    this_restricted_D = []
    for event in D:
      if event in label_list:
        this_restricted_D.append(event)

    restricted_D_dict[id] = this_restricted_D

  return restricted_D_dict


# function processes a multiple event sequence dataset such that labels are not repeated in a sequence
def remove_repeat_labels_event_seq(D_dict):

  new_D_dict = D_dict.copy()
  for id in sorted(D_dict):
    D = D_dict[id]
    new_D = list(set(D))
    if new_D:
      new_D_dict[id] = new_D
    else:
      del new_D_dict[id]

  return new_D_dict


# convert SPMF data into an event sequence for each line
def process_SPMF_data_to_event_seq(file_string):

  D_dict = {}
  seq_id = 0

  with open(file_string, 'r') as f:
    for line in f:
      # split line into event labels and remove item separator -1 and line separator -2
      seq = [x for x in line.split() if x != '-2' and x != '-1']
      # put this sequence in the dict
      D_dict[seq_id] = seq
      seq_id += 1

  return D_dict


# convert SPMF data to event sequences
# this function is from the schmidt repo
def process_SPMF_data_to_list(file_string):

  word_list = []  # list of all words in the data, in order of appearance
  unique_words = []  # list of unique words (i.e. event categories)

  with open(file_string, 'r') as f:
    for line in f:
      # split line into words and remove item separator -1 and line separator -2
      words = [x for x in line.split() if x != '-2' and x != '-1']

      # append this list to the total list of words
      word_list = word_list + words

      # update unique list of words
      unique_words = list(set().union(unique_words, words))

  return (word_list, unique_words)


# function splits a dict of event sequences by keys of the dict into a train/test pair
def split_dict_of_event_seq(D_dict, training_fraction=0.5):

  num_train = math.ceil(len(D_dict) * training_fraction) # number of entries for train set
  random_tuples = random.sample(D_dict.items(), num_train)
  train = dict(random_tuples)
  test = {k: D_dict[k] for k in set(D_dict) - set(train)}

  return train, test


# function does a threeway split into train/test/dev, for multiple event sequence data
# full data is divided into f1 (train), (1-f1)*f2 (dev), remaining (test)
def threeway_split(D_dict, f1, f2, common_labels_bool):
  '''
  :param D_dict:
  :param f1: fraction of first split
  :param f2: fraction of second split
  :param common_labels_bool: whether event labels need to be restricted to only retain common labels among the three splits
  :return:
  '''

  D_train, D_other = split_dict_of_event_seq(D_dict, f1)
  D_dev, D_test = split_dict_of_event_seq(D_other, f2)

  L_train = get_unique_labels(D_train)
  L_dev = get_unique_labels(D_dev)
  L_test = get_unique_labels(D_test)

  # print checks
  #print('Len of D_train:', len(D_train), 'Len of L_train:', len(L_train))
  #print('Len of D_dev:', len(D_dev), 'Len of L_dev:', len(L_dev))
  #print('Len of D_test:', len(D_test), 'Len of L_test:', len(L_test))
  print('Len of L_train:', len(L_train))
  print('Len of L_dev:', len(L_dev))
  print('Len of L_test:', len(L_test))

  # restrict all datasets to common labels if required
  if common_labels_bool:
    common_labels = list(set(L_train).intersection(set(L_dev)).intersection(L_test))
    print('Len of common_labels:', len(common_labels))

    D_train = restrict_data_event_seq(D_train, common_labels)
    D_dev = restrict_data_event_seq(D_dev, common_labels)
    D_test = restrict_data_event_seq(D_test, common_labels)

  return D_train, D_dev, D_test


### UTILITY FUNCTIONS FOR SuMMs

# increment counts in a dictionary
def increment_count_dict(this_dict, this_key):
    if this_key in this_dict:
        this_dict[this_key] += 1
    else:
        this_dict[this_key] = 1

    return this_dict


# function that counts number of occurrences of each label in a dict
def counts_dict(D_dict):

  counts_dict = {}
  for key in D_dict:
    D = D_dict[key]
    for lab in D:
      counts_dict = increment_count_dict(counts_dict, lab)

  return counts_dict


# function finds the set of labels that appear in the same sentence as a single event label X
# this is used to potentially restrict parent search to these labels
def find_labels_in_same_seqs(X, D_dict):

  labs_in_same_seqs = []
  for key in D_dict:
    D = D_dict[key]
    if X in D:
      labs_in_same_seqs += D
      labs_in_same_seqs = list(set(labs_in_same_seqs))

  return labs_in_same_seqs


# function that computes the number of order instantiations, given the number of parents
def num_order_instantiations(num_parents):

  num_order_instantiations = 0
  for i in range(0, num_parents+1):
    num_order_instantiations += (math.factorial(num_parents)/ math.factorial(i))

  return num_order_instantiations


### ------------------------------------------------------- ###

### EVALUATION RELATED


### OLDER VERSION
# compute evaluation metric for a single probabilistic prediction for binary variable (whether label X happened or not)
# IMP! this function returns the NEGATIVE of log loss and brier score
# def eval_single_prob_pred(prob, pos_event_bool, eval_case, error=0.001):
#     '''
#     :param prob: probability of the binary variable occurrence that was observed w.r.t a given label X; i.e.
#     the probability of observing that label if label is observed
#     or the probability of not observing that label if not observed (in a sequence at a particular point)
#     :param pos_event_bool: boolean for whether the positive event label (X) occurred or not
#     :param eval_case: either 'logloss' or 'brier'
#     :param error: this is an error term assumed when prob. is 0, which is not allowed (default is 0.001)
#     :return: negative of log loss or brier score, so that higher is better
#     '''
#
#     if eval_case == 'logloss':
#         if prob == 0:
#             #print('Prob. cannot be 0 for the log loss metric!')
#             this_eval_metric = np.log(error)
#             #return
#         else:
#             this_eval_metric = np.log(prob)
#
#     elif eval_case == 'brier': # we return negative brier score so that higher is better!
#         if pos_event_bool:
#             this_eval_metric = -1 * ((prob - 1) ** 2)
#         else:
#             this_eval_metric = -1 * ((1-prob) ** 2) # this is because 1-prob is the probability of positive event x
#
#     else:
#         print('This eval case is not covered!')
#         return
#
#     return this_eval_metric


def eval_single_prob_pred(prob, eval_case, error=0.001):
  '''
  :param prob: probability of the binary variable occurrence that was observed w.r.t a given label X; i.e.
  the probability of observing that label if label is observed
  or the probability of not observing that label if not observed (in a sequence at a particular point)
  :param eval_case: either 'logloss' or 'brier'
  :param error: this is an error term assumed when prob. is 0, which is not allowed (default is 0.001)
  :return: log loss or negative of brier score, so that higher is better
  '''

  if eval_case == 'logloss':

    if prob == 0:
      # print('Prob. cannot be 0 for the log loss metric!')
      this_eval_metric = np.log(error)
      # return
    else:
      this_eval_metric = np.log(prob)

  elif eval_case == 'brier':  # we return negative brier score so that higher is better!
      this_eval_metric = -1 * ((1 - prob) ** 2)

  else:
    print('This eval case is not covered!')
    return

  return this_eval_metric


# evaluate baseline - LR and HMM
def eval_baseline(baseline, name, param):

    # load and process baseline results & other files
    #output_dict, path, filename = process_baseline_data(baseline, name, param)
    output_dict = process_baseline_data(baseline, name, param)

    # load ground truth data from test set
    #with open(os.path.join(path, filename), 'rb') as f:
    #    D_dict = pickle.load(f)

    if name == 'timelineKB':

      labs_of_interest = [
        'armed conflict', 'arrest', 'aviation incident', 'bomb attack', 'disease outbreak',
        'economy', 'election', 'explosion', 'military raid', 'pathogen spread',
        'protest', 'quarantine', 'riot', 'vaccination', 'war']
      #labs_of_interest = output_dict.keys()

    elif name == 'timelineOH':
      labs_of_interest = [
        'arrest_Q1403016', 'battle_Q178561', 'biological process_Q2996394', 'injury_Q193078',
        'kidnapping_Q318296', 'military casualty classification_Q16861376', 'military operation_Q645883',
        'military tactics_Q207645', 'protest_Q273120', 'rebellion_Q124734']

    else:
      labs_of_interest = output_dict.keys()

    print('Printing keys!')
    for key in sorted(labs_of_interest):
      print(key)
    print('\n')

    eval_case = 'logloss'
    eval_metric_list = []
    for X in sorted(labs_of_interest):
        this_eval_metric = eval_single_lab(X, output_dict, eval_case)
        eval_metric_list.append(this_eval_metric)

    print(eval_metric_list)
    print('\n')
    print('Mean:', np.mean(eval_metric_list))

    return eval_metric_list


# function loads and processes baseline outputs for LR and HMM and dataset
def process_baseline_data(baseline, name, param):

    if baseline == 'LR':

      lookback = param
      # load baseline output data
      if name == 'stack_overflow':
        input_filename = 'output_stack_lgr_nopenalty_' + str(lookback) + '.txt'
      else:
        input_filename = 'output_' + name + '_lgr_nopenalty_' + str(lookback) + '.txt'

      output_path = os.path.join('..', '..', 'data', 'seq_data', 'lr_output')
      with open(os.path.join(output_path, input_filename)) as file:
          output_dict = ast.literal_eval(file.read())
      #print('Printing keys!')
      #for key in sorted(output_dict):
      #    print(key)
      #print('\n')

    elif baseline == 'HMM':

      states = param
      input_filename = 'output_' + name + '_' + str(states) + 'states_v2.txt'
      output_path = os.path.join('..', '..', 'data', 'seq_data', 'hmm_output')
      with open(os.path.join(output_path, input_filename)) as file:
          output_dict = ast.literal_eval(file.read())
      #print('Printing keys!')
      #for key in sorted(output_dict):
      #    print(key)
      #print('\n')

    else:
      print('This baseline is not covered!')
      return

    # # process path and filename
    # if name == 'mimic':
    #     path = os.path.join('..', '..', 'data', 'seq_data', 'mimic_data', 'threeway_split', 'train70')
    #     filename = name + '_seq_test'
    # elif name == 'diabetes':
    #     path = os.path.join('..', '..', 'data', 'seq_data', 'diabetes_data', 'threeway_split', 'train70')
    #     filename = 'new_diabetes_K_3_seq_test'
    # elif name == 'stack_overflow':
    #     path = os.path.join('..', '..', 'data', 'seq_data', 'so_data', 'threeway_split', 'train70')
    #     filename = name + '_subset1000_seq_test'
    # elif name == 'linkedin':
    #     path = os.path.join('..', '..', 'data', 'seq_data', 'linkedin', 'threeway_split', 'train70')
    #     filename = name + '_offset_seq_test'
    # elif name == 'beigebooks':
    #     path = os.path.join('..', '..', 'data', 'seq_data', 'beigebooks_data', 'threeway_split', 'train70')
    #     filename = name + '_seq_test'
    # elif name == 'narr_ied':
    #     path = os.path.join('..', '..', 'data', 'seq_data', 'narratives', 'ied')
    #     filename = name + '_seq_test'
    # elif name == 'narr_outbreaks':
    #     path = os.path.join('..', '..', 'data', 'seq_data', 'narratives', 'outbreaks')
    #     filename = name + '_seq_test'
    # elif name == 'narr_covid':
    #     path = os.path.join('..', '..', 'data', 'seq_data', 'narratives', 'covid')
    #     filename = name + '_seq_test'
    # else:
    #     print('This name is not covered!')
    #     return

    #return output_dict, path, filename
    return output_dict


# function evaluates for a single label on output dict of a baseline
#def eval_single_lab(X, output_dict, D_dict, eval_case):
def eval_single_lab(X, output_dict, eval_case):

    output_X = output_dict[X]
    eval_metric = 0

    for seq_id in output_X:

        this_output = output_X[seq_id]
        #D = D_dict[seq_id]
        #counter = 0
        #for lab in D:
        for prob_this_lab in this_output:
            # output is always probability of what is observed (whether X is observed or not)
            #prob_this_lab = this_output[counter]
            #if lab == X:
            #    # true_lab = X
            #    pos_event_bool = True
            #else:
            #    # true_lab = 'none'
            #    pos_event_bool = False
            # print('lab, prob:', true_lab, prob_this_lab)
            #eval_metric += eval_single_prob_pred(prob_this_lab, pos_event_bool, eval_case)

            eval_metric += eval_single_prob_pred(prob_this_lab, eval_case)

            #counter += 1
        # print('\n')

    return eval_metric


### ANALYZING RESULTS OF SUMMS


# function to find amplifying and inhibiting influencers for bsumm
def find_amplifiers_and_inhibitors(lab, influencers, thetas, ratio_threshold):
  '''
  :param lab: label under consideration
  :param influencers: list of influencers/parents of the SuMM
  :param thetas: probability parameters of the SuMM
  :param ratio_threshold (>1): a threshold (like 2), e.g. strong amplifiers average > 200% increase in prob. and
  strong inhibitors average < 50% increase in prob. as averaged over other instantiations
  :return: lists of amplifiers and inhibitors
  '''

  simplified_thetas = {}
  for key in thetas:
    if key[0] == lab:
      simplified_thetas[key[1]] = thetas[key]

  amplifiers = []
  inhibitors = []
  strong_amplifiers = []
  strong_inhibitors = []

  index = 0
  for influencer in influencers:

    ratio_dict = find_ratio_dict(index, simplified_thetas)
    if ratio_dict:
      mean_ratio = np.mean(list(ratio_dict.values()))
      if mean_ratio > 1:
        amplifiers.append(influencer)
      if mean_ratio < 1:
        inhibitors.append(influencer)
      if mean_ratio > ratio_threshold:
        strong_amplifiers.append(influencer)
      if mean_ratio < 1 / ratio_threshold:
        strong_inhibitors.append(influencer)

    index += 1

  return amplifiers, inhibitors, strong_amplifiers, strong_inhibitors


# function returns a ratio dictionary for each index in the thetas of a SUMM,
# i.e. for each influencer index, it finds the ratio of probabilities with and without the influencer
# for all other instantiations
def find_ratio_dict(index, simplified_thetas):
  '''
  :param index: index of the influencer in the list
  :param simplified_thetas: probability parameters in the SuMM in a form where
  key is tuple of 0s and 1s only and value is prob.
  :return:
  '''

  ratio_dict = {}
  for theta in simplified_thetas:
    bool_for_influencer = theta[index]
    # identify the corresponding tuples where this index is 0 and 1
    if bool_for_influencer == 0:
      tup_with_0 = theta
      tup_with_1_list = list(theta).copy()
      tup_with_1_list[index] = 1
      tup_with_1 = tuple(tup_with_1_list)
    else:
      tup_with_1 = theta
      tup_with_0_list = list(theta).copy()
      tup_with_0_list[index] = 0
      tup_with_0 = tuple(tup_with_0_list)

    # only find ratio when both probabilities are available
    if tup_with_0 in simplified_thetas and tup_with_1 in simplified_thetas:
      if tup_with_0 not in ratio_dict:
        ratio_dict[tup_with_0] = simplified_thetas[tup_with_1] / simplified_thetas[tup_with_0]
        # if tup_with_0 == 0:
        #    ratio_dict[tup_with_0] = thetas[tup_with_1]/min_theta
        # else:
        #    ratio_dict[tup_with_0] = thetas[tup_with_1]/thetas[tup_with_0]

  return ratio_dict


### SYNTHETIC DATA GENERATION

# function that provides probability for any of 5 events (A, B, C, D, E) to occur in a sequence
# A and B depend on B and C's prior occurrences; the others do not depend on prior occurrences
# prob of A and B combined is 0.4; B is an amplifier for A and C is an amplifier for B
def wins_and_probs_example():

  windows_for_A_and_B = [2,2]

  p_A_dict = {
    (0,0): 0.6,
    (0,1): 0.2,
    (1,0): 0.7,
    (1,1): 0.4
  }

  p_B_dict = {
    (0,0): 0.2,
    (0,1): 0.6,
    (1,0): 0.1,
    (1,1): 0.4
  }

  p_C = 0.15
  p_D = 0.025
  p_E = 0.025

  prob_dict = {
    'A': p_A_dict,
    'B': p_B_dict,
    'C': p_C,
    'D': p_D,
    'E': p_E
  }

  return prob_dict, windows_for_A_and_B # p_A_dict, p_B_dict, p_C, p_D, p_E


# function generates multiple sequences for the example
def gen_mult_seq_example(num_seqs, num_events_per_seq):

  L = ['A', 'B', 'C', 'D', 'E']
  prob_dict, windows = wins_and_probs_example()

  D_dict = {}
  for k in range(1, num_seqs + 1):
    D = gen_single_seq_example(num_events_per_seq, L, prob_dict, windows)
    D_dict[k] = D

  return D_dict


# function generates a single sequence for the example
def gen_single_seq_example(num_events_per_seq, L, prob_dict, windows):

  summ = SuMM.SuMM()

  (p_A_dict, p_B_dict, p_C, p_D, p_E) = \
    (prob_dict['A'], prob_dict['B'], prob_dict['C'], prob_dict['D'], prob_dict['E'])
  U = ['B', 'C'] # influencers of A and B
  model_case = 'bsumm'

  # maintain a dictionary of indices of each parent in the sorted list
  par_indices = {par_lab: U.index(par_lab) for par_lab in U}
  # dictionary that keeps updating the most recent time/stage for any event
  t_latest = {lab: -1 for lab in L}

  D = []
  for t in range(1, num_events_per_seq + 1):

    # determine the instantiation s (u or o) based on the history and window
    s_instantiation = summ.find_s_instantiation(U, t, par_indices, t_latest, windows, model_case)

    # find the exact probabilities of A and B
    prob_list = [p_A_dict[s_instantiation], p_B_dict[s_instantiation], p_C, p_D, p_E]

    # generate a label and append
    this_lab = np.random.choice(L, p=prob_list)
    D.append(this_lab)

    # update the latest time for the event observed and the sequence index/time
    t_latest[this_lab] = t

  return D


# function learns and evaluates
def run_synth_example(num_iters, num_seqs, num_events_per_seq):

  summ = SuMM.SuMM()
  L = ['A', 'B', 'C', 'D', 'E']
  model_case = 'bsumm'
  alpha = 0.1
  score_type = 'BIC'
  window = 3
  windows_info = {par: window for par in L}
  penalty_weight = 1
  X = ['A']
  true_parents = ['B', 'C']

  # generate data, learn model, evaluate
  f1_vec = []
  for iter_num in range(0, num_iters):
    D_dict = gen_mult_seq_example(num_seqs, num_events_per_seq)
    N = sum([len(D_dict[key]) for key in D_dict])
    LL, _ = summ.fit(X, L, D_dict, N, alpha, score_type, penalty_weight, windows_info, model_case)
    estimated_parents = summ.parents
    _, _, f1 = compare_parents(true_parents, estimated_parents)
    f1_vec.append(f1)

  return np.mean(f1_vec)


# function generates multiple sequences using the function "gen_synth_data_from_topology"
def gen_mult_seq_from_topology_example(num_seqs, num_events_per_seq, mult_factor):
  '''
  :param num_seqs: number of sequences generated
  :param num_events_per_seq: number of events per sequence
  :param mult_factor: multiplicative factor
  :return:
  '''

  L = ['A', 'B', 'C', 'D', 'E']



  # GRAPH 2
  G = {
    'A': [],
    'B': ['A'],
    'C': ['A', 'B'],
    'D': ['A'],
    'E': ['A','B']
  }


   # GRAPH 1
#   G = {
#     'A': [],
#     'B': ['A'],
#     'C': ['A', 'B'],
#     'D': ['E'],
#     'E': ['D']
#   }

  D_dict = {}
  for k in range(1, num_seqs + 1):
    D = gen_synth_data_from_topology(L, G, num_events_per_seq, mult_factor)
    D_dict[k] = D

  return D_dict


# function for topological sequence generator
# here, the unnormalized probability of a label occurrence depends on counts of prior
# occurrences in the history of its parent labels according to some topology
def gen_synth_data_from_topology(L, G, num_events_per_seq, mult_factor):
  '''
  :param L: label set
  :param G: underlying topology/graph that guides sequence generation
  :param num_events_per_seq: number of events per sequence
  :param mult_factor: multiplicative factor
  :return:
  '''

  hist_info_dict = {} # dict that stores counts of all event labels

  D = []
  for t in range(1, num_events_per_seq + 1):
    unnormalized_prob_vec = []
    for lab in sorted(L):
      unnormalized_prob = 1 # this is the default
      # count the number of occurrences of
      counts = 0
      for par in G[lab]:
        if par in hist_info_dict:
          counts += 1
      if counts != 0:
        unnormalized_prob = mult_factor * counts
      unnormalized_prob_vec.append(unnormalized_prob)

    prob_vec = np.array(unnormalized_prob_vec) / sum(np.array(unnormalized_prob_vec))
    # generate a label and append
    this_lab = np.random.choice(L, p=prob_vec)
    D.append(this_lab)
    # update the historical information
    if this_lab in hist_info_dict:
      hist_info_dict[this_lab] += 1
    else:
      hist_info_dict[this_lab] = 1

  return D


### EVALUATION RELATED


# function compares parents; each graph is a list of IDs of its parents
def compare_parents(true_parents, estimated_parents):

  '''
  Inputs
  ------
  true_parents: true parents
  estimated_parents: estimated parents
  '''

  num_true_pos = 0
  num_false_pos = 0
  num_false_neg = 0

  # check all true parents, mark true positives and false negatives
  for par in true_parents:
    if par in estimated_parents:
      num_true_pos += 1
    else:
      num_false_neg += 1
  # check all estimated parents, mark false positives
  for par in estimated_parents:
    if par not in true_parents:
      num_false_pos += 1

  if num_true_pos + num_false_pos == 0:
      prec = 1 # here tp = 0; fp = 0 - there are no spurious results
  else:
    prec = num_true_pos / (num_true_pos + num_false_pos)

  if num_true_pos + num_false_neg == 0:
    recall = 1 # here tp = 0; fn = 0 - all the true positives are discovered
  else:
    recall = num_true_pos / (num_true_pos + num_false_neg)

  if prec + recall == 0:
    f1 = 0
  else:
    f1 = (2 * prec * recall) / (prec + recall)

  #num_true_neg = (num_nodes ** 2) - (num_true_pos + num_false_pos + num_false_neg)
  #acc = (num_true_pos + num_true_neg) / (num_nodes ** 2)

  return prec, recall, f1


### PROCESS TEXT

# def xml_to_text(INPUT_FILE):
#
#     # parse an xml file by name
#     mydoc = minidom.parse(INPUT_FILE)
#
#     # get sentences
#     sents = mydoc.getElementsByTagName('ORIGINAL_TEXT')
#
#     text = ''
#     for elem in sents:
#         sent = elem.firstChild.data
#
#         #if sent[-1] != '.':
#         #    sent += '.'
#         # print(sent)
#
#         # only include sentences with a period at the end
#         if sent[-1] == '.':
#             text += sent
#
#     return text