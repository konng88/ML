from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        # Edit here
        for s in range(S):
            alpha[s][0] = self.pi[s] * self.B[s][self.obs_dict[Osequence[0]]]

        for t in range(1,L):
            for s in range(S):
                tmp = np.dot(alpha[:,t-1].reshape((1,S)),self.A[:,s].reshape((S,1)))
                alpha[s][t] = self.B[s][self.obs_dict[Osequence[t]]] * float(tmp)
        ###################################################

        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        # Edit here
        ###################################################
        for s in range(S):
            beta[s][L-1] = 1
        for t in reversed(range(0,L-1)):
            for s in range(S):
                a = self.A[s,:]
                b = self.B[:,self.obs_dict[Osequence[t+1]]]
                c = beta[:,t+1]
                beta[s][t] = np.sum(a*b*c)
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        # Edit here
        T = len(Osequence)
        alpha = self.forward(Osequence)
        prob = np.sum(alpha[:,T-1])
        ###################################################
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        # Edit here
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        prior = np.sum(alpha[:,L-1])
        for s in range(S):
            for t in range(L):
                prob[s][t] = alpha[s][t] * beta[s][t] / prior
        ###################################################
        return prob
    #TODO:
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        # Edit here
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        prior = np.sum(alpha[:,L-1])
        for t in range(L-1):
            for s1 in range(S):
                for s2 in range(S):
                    prob[s1][s2][t] = alpha[s1][t] * self.A[s1][s2] * self.B[s2][self.obs_dict[Osequence[t+1]]] * beta[s2][t+1] / prior
        ###################################################
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        # Q3.3 Edit here
        S = len(self.pi)
        L = len(Osequence)
        state_dict_rev = dict(zip(self.state_dict.values(),self.state_dict.keys()))
        detla = np.zeros([S,L])
        DETLA = np.zeros([S,L])
        for s in range(S):
            detla[s][0] = self.pi[s] * self.B[s][self.obs_dict[Osequence[0]]]
        for t in range(1,L):
            for s in range(S):
                detla[s][t] = self.B[s][self.obs_dict[Osequence[t]]] * max(self.A[:,s] * detla[:,t-1])
                DETLA[s][t] = np.argmax(self.A[:,s] * detla[:,t-1])

        path = ['None'] * L
        path[L-1] = np.argmax([detla[:,L-1]])
        for t in reversed(range(1,L)):
            path[t-1] = int(DETLA[path[t]][t])
        for t in range(L):
            path[t] = state_dict_rev[path[t]]


        # for t in range(L):
        #     s = np.argmax(detla[:,t])
        #     path.append(state_dict_rev[s])

        ###################################################
        return path
