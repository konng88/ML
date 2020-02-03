import numpy as np

from util import accuracy
from hmm import HMM

# TODO:
def model_training(train_data, tags):
	"""
	Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
	"""
	model = None
	###################################################
	# Edit hereN = len(train_data)
	obs_dict = {}
	state_dict = {}
	obs_code = 0
	state_code = 0
	for line in train_data:
		for i in range(len(line.words)):
			line.words[i] = line.words[i].lower()
	for line in train_data:
		for word in line.words:
			if word not in obs_dict.keys():
				obs_dict[word] = obs_code
				obs_code += 1
	for tag in tags:
		if tag not in state_dict.keys():
			state_dict[tag] = state_code
			state_code += 1
	S = len(tags)
	L = len(obs_dict)
	N = len(train_data)
	pi = np.zeros((S,))
	A = np.zeros((S,S))
	B = np.zeros((S,L))
	start_count = dict(zip(state_dict.values(),np.zeros(S)))

	for line in train_data:
		s1 = state_dict[line.tags[0]]
		start_count[s1] += 1
		l = len(line.words)

		for i in range(l-1):
			A[state_dict[line.tags[i]]][state_dict[line.tags[i+1]]] += 1

		for i in range(l):
			B[state_dict[line.tags[i]]][obs_dict[line.words[i]]] += 1

	A = (A.T / np.sum(A,axis=1)).T
	B = (B.T / np.sum(B,axis=1)).T

	for tag in tags:
		pi[state_dict[tag]] = start_count[state_dict[tag]] / N

	model = HMM(pi, A, B, obs_dict, state_dict)

	###################################################
	return model

# TODO:
def sentence_tagging(test_data, model, tags):
	"""
	Inputs:
	- test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- model: an object of HMM class

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
	"""
	tagging = []
	###################################################
	# Edit here
	S , K = model.B.shape

	for line in test_data:
		for i in range(len(line.words)):
			word = line.words[i].lower()
			line.words[i] = word
			if word not in model.obs_dict.keys():
				model.obs_dict[word] = len(model.obs_dict)-1
				e = np.ones((S,1)) * 10**(-6)
				model.B = np.hstack((model.B,e))
				K += 1


	for line in test_data:
		p_old = -1
		p_new = -2

		while p_new > p_old:
			p_old = p_new
			p_new = model.sequence_prob(line.words)
			gamma = model.posterior_prob(line.words)
			ksi = model.likelihood_prob(line.words)

			model.pi = gamma[:,0]
			for s1 in range(S):
				for s2 in range(S):
					model.A[s1][s2] = np.sum(gamma[s1][s2]) / np.sum(gamma[s1])
				for k in range(K):
					tmp = []
					for word in line.words:
						tmp.append(int(model.obs_dict[word]==k))
					model.B[s1][k] = np.sum(tmp*gamma[s1,:]) / np.sum(gamma[s1,:])

		tagging.append(model.viterbi(line.words))

	###################################################
	return tagging
