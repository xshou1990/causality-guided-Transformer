'''

  About
  ------------------
  Implementation of the summary Markov model (SuMM) - for binary (BSuMM) and ordinal (OSuMM)
  We will try to use the following notation for data:
    - D_dict refers to event sequences in the form of a dict with a separate sequence for each key

  All models need to implement the following functions:
  .save -- save the model (pickled) to the given path
  .load -- load the model from the given path

'''

import pickle
import numpy as np
#import itertools
#import os
#import random
from collections import Counter

import summs_utils_v2


class SuMM:
  '''
  A summary Markov model, either binary or ordinal
  '''


  def __init__(self):
    '''
    This is the initializer for the model which consists of parents, windows, thetas

    Class parameters
    ----------------

    parents: a sorted list of parent IDs
    windows: a list of windows (in order of sorted parents)
    thetas: dict where key is tuple of the form (x; s) -
    x is state of the r.v. X associated with a subset of labels and s is either binary vector of parent instantiations
    (in order of sorted parents), or order of any subset of parents. the value is the theta parameter (probability).
    note that for consistency, we always use a tuple for instantiation, i.e. also for no parents (tuple([]))
    or 1 parent (ex: tuple([0]))
    '''

    self.parents = []
    self.windows = []
    self.thetas = {}


  def save(self, file_name):
    '''
    This should pickle the necessary bits to save the model.

    Data
    -----------
    model: python object
      object to be saved.

    file_name: string
      The file name to save myself.
    '''

    with open(file_name, 'wb') as f:
      pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)


  def load(self, file_name):
    '''
    This should load the necessary bits to instantiate an instance of this model.

    Data
    -----------
    file_name: string
      The name to load from.
    '''

    with open(file_name, 'rb') as f:
      tmp_dict = pickle.load(f)
    self.__dict__.update(tmp_dict)


  # function fits either a BSuMM or OSuMM and returns log likelihood and score (BIC)
  # windows info is a dict for BSuMM but a number for OSuMM
  def fit(self, X, L, D_dict, N, alpha, score_type, penalty_weight, windows_info, model_case, restriction_bool = False):
    '''
    :param X: event labels under consideration (list of event labels)
    :param L: label set
    :param D_dict: data dict - multiple event sequences
    :param N: total number of events
    :param alpha: parameter for prior on thetas; we assume all alpha_x|s = alpha
    :param score_type: currently only BIC is enabled
    :param penalty_weight: weight on complexity term, 0 < gamma <= 1
    :param windows_info: information about windows
    :param model_case: type of model - either 'bsumm' or 'osumm'
    :param restriction_bool: True if there is a restriction on parent set (pre-determined for now); o/w False
    :return: model and its LL and score on the data
    '''

    # determine set of feasible parents to search over
    if restriction_bool:
      feasible_pars = []
      for lab in X:
        feasible_pars += summs_utils.find_labels_in_same_seqs(lab, D_dict)
        feasible_pars = list(set(feasible_pars))
    else:
      feasible_pars = L
    #print(feasible_pars)

    # forward search
    SuMM_dict = self.forward_search(X, feasible_pars, L, D_dict, N, alpha, score_type, penalty_weight, windows_info, model_case)
    # backward search
    SuMM_dict = \
      self.backward_search(SuMM_dict, X, L, D_dict, N, alpha, score_type, penalty_weight, windows_info, model_case)

    self.parents = SuMM_dict['parents']
    self.windows = SuMM_dict['windows']
    self.thetas = SuMM_dict['thetas']

    return SuMM_dict['LL'], SuMM_dict['score']


  # function sums both types of counts (obtained from find_counts) for multiple sequences
  # also returns a list of counts for each sequence, to aid with the LL computation
  def tot_summary_stats(self, X, parents, windows, L, D_dict, model_case):
    '''
    :param X: event labels under consideration (list of event labels)
    :param parents: parents of X (list of event labels)
    :param windows: windows for parents of X (list of integers, in the same order as parents)
    :param L: label set - note that each seq. can have a different set of labels, so L is input here to cover a broader set of labels
    :param D_dict: data dict - multiple event sequences
    :param model_case: the type of model - either 'bsumm' or 'osumm'
    :return: returns the sum of both counts over sequences, and also the list of both counts for each stream
    '''

    counts = Counter({})
    summ_counts = Counter({})

    counts_list = []
    summ_counts_list = []

    for id in sorted(D_dict):

      D = D_dict[id]

      # check for empty dataset
      if D:
        this_counts, this_summ_counts = self.find_counts(X, parents, windows, L, D, model_case)
      else:
        this_counts, this_summ_counts = {}, {}

      counts = counts + Counter(this_counts)
      summ_counts = summ_counts + Counter(this_summ_counts)

      counts_list.append(this_counts)
      summ_counts_list.append(this_summ_counts)

    return dict(counts), dict(summ_counts), counts_list, summ_counts_list


  # function finds the counts for label set X, for given parents and window, for a single event stream
  # windows are left closed, right open: [t-w, t)
  # works for both binary and ordinal counting
  def find_counts(self, X, parents, windows, L, D, model_case):
    '''
    Inputs
    ------
    X: event labels under consideration (list of event labels)
    parents: parents of X (list of event labels)
    windows:
    - for bsumm: a window for each parent of X (list of real-valued numbers, in the same order as parents)
    - for osumm: a list with a single window
    L: label set (list of event labels)
    D: single event sequence dataset (list of event labels) assumed ordered by time
    model_case: the type of model - either 'bsumm' or 'osumm'

    Outputs
    ------
    counts: N[x; s] for instantiations x of X and summary instantiations (either u or o) of parents U
    summ_counts: N[s] for summary instantiations (either u or o) of parents U
    '''

    # check for empty event stream
    if not D:
      return {}, {}

    # check for valid windows
    for window in windows:
      if window < 0:
        print("There is a negative window!")
        return

    # check for consistent length between parents and windows, depending on the model case
    if model_case == 'bsumm':
      if len(parents) != len(windows):
        print("The number of parents and number of windows should be the same for bsumm!")
        return
    elif model_case == 'osumm':
      if len(windows) != 1:
        print("For osumm, windows should be a list with a single window!")
        return
    else:
      print('This model case is not covered!')
      return

    U = sorted(parents)  # parent set sorted by ID
    counts = {}
    summ_counts = {}

    # maintain a dictionary of indices of each parent in the sorted list
    par_indices = {par_lab: U.index(par_lab) for par_lab in U}
    # dictionary that keeps updating the most recent time/stage for any event
    t_latest = {lab: -1 for lab in L}

    t = 1  # sequence index/time
    for this_lab in D:

      # determine the instantiation s (u or o) based on the history and window
      s_instantiation = self.find_s_instantiation(U, t, par_indices, t_latest, windows, model_case)

      # update N(s)
      summ_counts = summs_utils.increment_count_dict(summ_counts, s_instantiation)

      # determine the state x based on the current label
      if this_lab in X:
        this_state = this_lab
      else:
        this_state = 'none'  # we reserve a special label for when the observed label is not in Z

      # update N(x; s)
      counts = summs_utils.increment_count_dict(counts, (this_state, s_instantiation))

      # update the latest time for the event observed and the sequence index/time
      t_latest[this_lab] = t
      t += 1

    return counts, summ_counts


  # functions finds the s_instantiation in any position, given the necessary information
  def find_s_instantiation(self, U, t, par_indices, t_latest, windows, model_case):

    if model_case == 'bsumm':
      u_vec = [0] * len(U)
      for parent_lab in U:
        this_index = par_indices[parent_lab]
        window = windows[this_index]  # window for this parent
        # set the binary element in the vector to 1 if it has occurred within the window
        if t_latest[parent_lab] != -1 and (t - t_latest[parent_lab]) <= window:
          u_vec[this_index] = 1
      # print(u_vec)
      s_instantiation = tuple(u_vec)
    elif model_case == 'osumm':
      window = windows[0]  # single window in osumm
      order_tup_list = []
      for parent_lab in U:
        # add tuples of (label, latest time) if they are present
        if t_latest[parent_lab] != -1 and (t - t_latest[parent_lab]) <= window:
          order_tup_list.append((parent_lab, t_latest[parent_lab]))
      if order_tup_list:  # sort the labels in order of the latest time they arrived ("last" masking function)
        order_tup_list.sort(key=lambda x: x[1])
        o_vec = [tup[0] for tup in order_tup_list]
      else:
        o_vec = []
      # print(o_vec)
      s_instantiation = tuple(o_vec)
    else:
      print('This model case is not covered!')
      return

    return s_instantiation


  # function finds the optimal thetas (probabilities) given the summary statistics
  # currently we take a Bayesian approach to prevent zero probabilities
  # this function only stores a theta when (x;s) is observed in the counts data - the others are accounted
  # for in the LL computation directly
  def opt_thetas(self, X, L, alpha, counts, summ_counts):
    '''
    :param X: labels that form the random variable of interest
    :param L: label set
    :param alpha: parameter for prior on thetas; we assume all alpha_x|s = alpha
    :param counts: N(x;s)
    :param summ_counts: N(s)
    :return: thetas/probabilities (theta_x|s)
    '''

    if len(X) == len(L):
      X_states = X
    else:
      X_states = X + ['none']
    num_states_X = len(X_states)
    alpha_s = alpha * num_states_X # alpha_s is the sum of alpha_x|s over all states x

    thetas = {}

    # OLD METHOD
    # for key in counts:
    #   u_instantiation = key[1]
    #   numer = alpha + counts[key] # alpha_x|u + N(x;u)
    #   denom = alpha_s + summ_counts[u_instantiation]  # alpha_s + N(u)
    #   thetas[key] = numer / denom

    # NEW METHOD
    # we account for all cases EXCEPT when N(s) = 0, in which case, theta = alpha_x|s / alpha_s
    # these thetas are not explicitly assigned here, for efficiency
    for s_instantiation in summ_counts:
      for x_state in X_states:
        this_key = (x_state, s_instantiation)
        if this_key in counts:
          this_N_xs = counts[this_key]
        else:
          this_N_xs = 0
        numer = alpha + this_N_xs  # alpha_x|s + N(x;s)
        denom = alpha_s + summ_counts[s_instantiation]  # alpha_s + N(s)
        thetas[this_key] = numer / denom

    return thetas


  # compute log likelihood using thetas and counts from each sequence
  # note that we only need individual sequence counts N^k(x;s) !!!
  # we will infer any missing thetas that are needed using only the alphas
  def log_likelihood_from_thetas(self, num_states_X, alpha, thetas, counts_list):

    alpha_s = alpha * num_states_X  # alpha_s is the sum of alpha_x|s over all states x

    LL = 0
    # summing over event sequences
    for seq_index in range(0, len(counts_list)):
      #print('seq_index:', seq_index)
      this_counts = counts_list[seq_index]

      # summing over (x;s) combinations
      for key in this_counts:
        #print('key:', key)
        if key in thetas:
          this_theta = thetas[key]
        else: # if the corresponding theta is missing, then compute it and incorporate
          this_theta = alpha / alpha_s
        if this_theta == 0:
          print('Something is wrong - theta should not be 0!')

        this_LL = this_counts[key] * np.log(this_theta)
        #print('this_LL:', this_LL)
        LL += this_LL

    return LL

  # compute log likelihood on new data using a model
  # we assume that the label set L is the same !!!
  def log_likelihood_new_data(self, X, L, D_dict_new, alpha, model_case):

    if len(X) < len(L):
      num_states_X = len(X) + 1
    else:
      num_states_X = len(X)

    # compute summary stats on new data
    _, _, counts_list, _ = self.tot_summary_stats(X, self.parents, self.windows, L, D_dict_new, model_case)
    # LL
    LL = self.log_likelihood_from_thetas(num_states_X, alpha, self.thetas, counts_list)

    return LL


  # function computes BIC score from log likelihood for node and parent list and other parameters
  def find_score(self, num_states_X, parents, LL, N, score_type, penalty_weight, model_case):

    # need to multiply the number of states of r.v. X minus 1 with the number of instantiations
    if model_case == 'bsumm':
      num_params = (num_states_X-1) * (2**len(parents))
    elif model_case == 'osumm':
      num_params = (num_states_X-1) * summs_utils.num_order_instantiations(len(parents))
    else:
      print('This model case is not covered!')
      return

    # currently only BIC score is enabled
    if score_type == 'BIC':
      if N == 0:
        penalty = 0
      else:
        penalty = num_params * (np.log(N)/2)

      eff_penalty = (penalty_weight * penalty)
      score = LL - eff_penalty

    else:
      print('This score type is not recognized!')
      return

    return score, LL, eff_penalty


  # function finds the optimal model given windows
  def opt_model_given_windows(self, X, parents, windows, L, D_dict, alpha, model_case):

    if len(X) < len(L):
      num_states_X = len(X) + 1
    else:
      num_states_X = len(X)
    # summary statistics
    counts, summ_counts, counts_list, _ = self.tot_summary_stats(X, parents, windows, L, D_dict, model_case)
    # optimal thetas
    thetas = self.opt_thetas(X, L, alpha, counts, summ_counts)
    # LL
    LL = self.log_likelihood_from_thetas(num_states_X, alpha, thetas, counts_list)

    return thetas, LL

  # function formats windows and finds the thetas, LL and score given windows info dict
  def opt_model_and_score_from_windows_info(self, X, parents, L, D_dict, N, alpha, score_type,
                                            penalty_weight, windows_info, model_case):

    # format windows appropriately for the parents
    if model_case == 'bsumm':
      windows = []
      for par in parents:
        windows.append(windows_info[par])
    elif model_case == 'osumm':
      windows = [windows_info]
    else:
      print('This model case is not covered!')
      return

    # find optimal model given windows: thetas, LL
    thetas, LL = self.opt_model_given_windows(X, parents, windows, L, D_dict, alpha, model_case)

    # find score
    if len(X) < len(L):
      num_states_X = len(X) + 1
    else:
      num_states_X = len(X)
    score, _, _ = self.find_score(num_states_X, parents, LL, N, score_type, penalty_weight, model_case)

    return thetas, windows, LL, score


  # function performs a forward search for parents
  # we allow for a future approach to also obtain windows - for now, they are assumed known
  def forward_search(self, X, feasible_pars, L, D_dict, N, alpha, score_type, penalty_weight, windows_info, model_case):

    U = []
    global_opt_score = -np.inf
    global_opt_LL = None
    global_opt_windows = []
    global_opt_thetas = []
    score_increase_indic = 1

    while score_increase_indic == 1 and len(U) <= len(feasible_pars):

      local_opt_score = -np.inf
      local_opt_Z = None
      local_opt_LL = None
      local_opt_windows = []
      local_opt_thetas = {}

      for Z in feasible_pars:
        if Z not in U:
          local_U = sorted(U + [Z])

          # format windows, find optimal model (thetas, LL) and compute score
          thetas, windows, LL, score = \
            self.opt_model_and_score_from_windows_info(X, local_U, L, D_dict, N, alpha, score_type,
                                                       penalty_weight, windows_info, model_case)

          # update if this score beats the local optimal
          if score > local_opt_score:
            local_opt_score = score
            local_opt_Z = Z
            local_opt_LL = LL
            local_opt_windows = windows
            local_opt_thetas = thetas

      if local_opt_score > global_opt_score:
        global_opt_score = local_opt_score
        global_opt_LL = local_opt_LL
        global_opt_windows = local_opt_windows
        global_opt_thetas = local_opt_thetas
        U = sorted(U + [local_opt_Z])

        #print('Latest parent set from forward search:', U)

      else:
        score_increase_indic = 0

    SuMM_dict = {
      'parents': U,
      'windows': global_opt_windows,
      'thetas': global_opt_thetas,
      'score': global_opt_score,
      'LL': global_opt_LL
    }

    return SuMM_dict


  # function performs a backward search for parents
  # we allow for a future approach to also obtain windows - for now, they are assumed known
  def backward_search(self, SuMM_dict, X, L, D_dict, N, alpha, score_type, penalty_weight, windows_info, model_case):

    U = SuMM_dict['parents']
    global_opt_score = SuMM_dict['score']
    global_opt_LL = SuMM_dict['LL']
    global_opt_windows = SuMM_dict['windows']
    global_opt_thetas = SuMM_dict['thetas']
    score_increase_indic = 1

    while score_increase_indic == 1 and len(U) > 0:

      local_opt_score = -np.inf
      local_opt_LL = None
      local_opt_Z = None
      local_opt_windows = []
      local_opt_thetas = {}

      for Z in U:
        # remove Z from U
        local_U = U[:]
        local_U.remove(Z)

        # format windows, find optimal model (thetas, LL) and compute score
        thetas, windows, LL, score = \
          self.opt_model_and_score_from_windows_info(X, local_U, L, D_dict, N, alpha, score_type,
                                                     penalty_weight, windows_info, model_case)

        # update if this score beats the local optimal
        if score > local_opt_score:
          local_opt_score = score
          local_opt_Z = Z
          local_opt_LL = LL
          local_opt_windows = windows
          local_opt_thetas = thetas

      if local_opt_score > global_opt_score:
        global_opt_score = local_opt_score
        global_opt_LL = local_opt_LL
        global_opt_windows = local_opt_windows
        global_opt_thetas = local_opt_thetas
        U.remove(local_opt_Z)

        #print('\n')
        #print('Latest parent set from backward search:', U)

      else:
        score_increase_indic = 0

    SuMM_dict = {
      'parents': U,
      'windows': global_opt_windows,
      'thetas': global_opt_thetas,
      'score': global_opt_score,
      'LL': global_opt_LL
    }

    return SuMM_dict


  # function evaluates probabilistic prediction (for a single variable X) on data using a model
  # eval_case is either 'logloss' or 'brier'
  def eval_predict(self, X, L, D_dict, alpha, eval_case, model_case):

    if len(X) != 1:
      print('X must be a single variable!')
      return
    if len(X) < len(L):
      num_states_X = len(X) + 1
    else:
      num_states_X = len(X)
    alpha_s = alpha * num_states_X  # alpha_s is the sum of alpha_x|s over all states x

    U = self.parents
    windows = self.windows

    # maintain a dictionary of indices of each parent in the sorted list
    par_indices = {par_lab: U.index(par_lab) for par_lab in U}

    eval_metric = 0
    # summing over sequences
    for id in D_dict:

      D = D_dict[id]
      # dictionary that keeps updating the most recent time/stage for any event
      t_latest = {lab: -1 for lab in L}

      t = 1
      for this_lab in D:

        # determine the instantiation s (u or o) based on the history and window
        s_instantiation = self.find_s_instantiation(U, t, par_indices, t_latest, windows, model_case)

        # determine the state x based on the current label
        # IMP! recall that this currently only works for a singleton X
        if this_lab in X:
          x_state = this_lab
        else:
          x_state = 'none'  # we reserve a special label for when the observed label is not in X

        # find theta for this position
        key = (x_state, s_instantiation)
        if key in self.thetas:
          this_theta = self.thetas[key]
        else: # need to compute thetas if missing - the ones missing can be computed from only the alphas
          this_theta = alpha / alpha_s

        # evaluate prediction for this position
        pos_event_bool = (x_state in X)
        this_eval_metric = summs_utils.eval_single_prob_pred(this_theta, pos_event_bool, eval_case)
        eval_metric += this_eval_metric

        #print('s:', s_instantiation)
        #print('x state:', x_state)
        #print('eval:', this_eval_metric)

        # update the latest time for the event observed and the sequence index/time
        t_latest[this_lab] = t
        t += 1

    return eval_metric


  # function to store counts information associated with thetas in a model
  # the key is stored as a string here for easy conversion to json
  def store_counts_info(self, X, L, D_dict, parents, windows_info, thetas, model_case):

    if model_case == 'bsumm':
      windows = []
      for par in parents:
        windows.append(windows_info[par])
    else:
      windows = [windows_info]

    # find counts and summ_counts
    counts, summ_counts, _, _ = self.tot_summary_stats(X, parents, windows, L, D_dict, model_case)

    # format output
    counts_dict = {}
    for key in thetas:
      if key[0] == X:
        str_key = ''
        counter = 0
        # prepare string
        for num in key[1]:
          if counter < len(key[1]) - 1:
            str_key += str(num) + ','
          else:
            str_key += str(num)
          counter += 1
        # track counts of the form N(x;s)
        if key in counts:
          this_counts = counts[key]
        else:
          this_counts = 0
        # track counts of the form N(s)
        if key[1] in summ_counts:
          this_summ_counts = summ_counts[key[1]]
        else:
          this_summ_counts = 0
        counts_dict[str_key] = (this_counts, this_summ_counts)

    return counts_dict


