#!/usr/bin/python
import numpy as np
import random

class SBP:        
    """
        Simulates a stick-breaking process where the kth weight
        w_k = beta_k' * \prod_{i=1}^{k-1}{(1-beta_i')}, and beta_k' is drawn
        from a Beta distribution with parameters alpha and beta
    """
    def __init__(self,alpha,beta):
        """
            alpha: first shape parameter for the Beta distribution
            beta:  second shape parameter for the Beta distribution
        """
        self.alpha=alpha
        self.beta=beta
        self.weights = []
        self.cummulative = []  #the kth entry in this list holds the sum of the first k weights
        self.betaDraws = []
        self.products = [] #the kth entry holds the cummulative product \prod_{i=1}^{k}{(1-beta_i')}

    def __getitem__(self,k):
        """
            Returns the kth weight. It may be necessary to first
            obtain this weight by sampling from a Beta distribution
        """
        while len(self.weights)-1 < k:
            self.breakStick()
        return self.weights[k]
        
    def sampleWeight(self):
        """
            Returns one of the weights. The weight is sampled with
            probability w_k/T, where w_k is the kth weight, and T is the
            sum of all the weights.
        """
        x=random.random()
        i = 0
        while True:
            w = self.__getitem__(i)
            if x <= self.cummulative[i]:
                return w
            i += 1

    def sampleIndex(self):
        """
            Returns the index of one of the weights. The index is sampled with
            probability w_k/T, where w_k is the kth weight, and T is the
            sum of all the weights.
        """
        x=random.random()
        i = 0
        while True:
            w = self.__getitem__(i)
            if x <= self.cummulative[i]:
                return i
            i += 1
            
    def breakStick(self):
        """
            Simulate the stick-breaking process. Sample a new element from a beta 
            distribution, and use it to compute a new weight. Then, update the auxiliary 
            lists.
        """
        betaSample = np.random.beta(self.alpha,self.beta)
        self.betaDraws.append(betaSample)
        if self.products:
            w = betaSample*self.products[-1]
            self.weights.append(w)
            self.products.append((1-betaSample)*self.products[-1])
            self.cummulative.append(self.cummulative[-1]+w)
        else: #this gets executed only for the first weight sampled (i.e. k = 0)
            self.weights.append(betaSample)
            self.products.append((1-betaSample))
            self.cummulative.append(betaSample)


class HDP_Sample(SBP):
    """
        Represents a sample from a Hierarchical Dirichlet process. The kth weight in this 
        sampled is obtained by following a stick-breaking process with variable parameters 
        for sampling from the Beta distribution.
        Each weight pi_k = pi_k' * \prod_{i=1}^{k-1}{(1-pi_i')}, and pi_k' is drawn from 
        a Beta distribution with parameters alpha_0*beta_k, and alpha_0*(1-\sum_{i=1}^k{beta_i}),
        where alpha_0 is the inverse variance parameter of a Hierarchical Dirichlet Process 
        and the weights beta_k are the global weights sampled from that process following 
        a stick-breaking process
    """
    def breakStick(self):
        """
            Overwrite the stick-breaking process to use variable parameters to sample from 
            the Beta Distribution. 
        """
        k = len(self.weights)
        betaSample = np.random.beta(self.alpha*self.beta[k],
                                    self.alpha*(1-self.beta.cummulative[k]))
        self.betaDraws.append(betaSample)
        if self.products:
            w = betaSample*self.products[-1]
            self.weights.append(w)
            self.products.append((1-betaSample)*self.products[-1])
            self.cummulative.append(self.cummulative[-1]+w)
        else:
            self.weights.append(betaSample)
            self.products.append((1-betaSample))
            self.cummulative.append(betaSample)
    
class HDP:
    """
        Represents a Hierarchical Dirichlet Process. Samples are generated following 
        the stick-breaking construction in Section 4.1 of 
        Y. W. Teh, M. I. Jordan, M. J. Beal, and D. M. Blei. 
        Hierarchical dirichlet processes. 
        Journal of the American Statistical Association, 101(476), 2006.
    """
    def __init__(self,gamma,alpha_0):
        """
            gamma:   concentration parameter for the Dirichlet process from where the global 
                     random probability measure G_0 is drawn.
            alpha_0: concentration parameter for the Dirichlet process from where the 
                     random measures G_j are drawn.
        """
        self.sb_process = SBP(1,gamma) #Stick breaking process to sample the weights of G_0
        self.gamma = gamma
        self.alpha_0 = alpha_0
        self.samples = []

    def newSample(self):
        """
            Draws a new random measure from Dirichlet process and returns it 
        """
        self.samples.append(HDP_Sample(self.alpha_0, self.sb_process))
        return self.samples[-1]
