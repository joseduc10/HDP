#!/usr/bin/python

import hdp
import random
import numpy  as np
from math import log

class Network:
    """
        Simulates a social network environment. Contains basic methods
        commonly found in REST API's of social networks, such as getting
        all entities (comments, users, etc), or a subset of the entities
        filtered by some criteria
    """
    submissions = []
    users = []

    def __init__(self, users=[]):
        """
            users: A list of users that will participate in the network
        """
        self.users = users

    def addSubmission(self, submission):
        """
            Adds a submission and returns the ID of the submission in the network
        """
        self.submissions.append(submission)
        return len(self.submissions)-1

    def addUser(self, user):
        """
            Adds a user and returns the ID of the user in the network
        """
        self.users.append(user)
        return len(self.users)-1
    
    def getSubmission(self, ind):
        """
            Returns the submission with index "ind" or None if
            such submission does not exist
        """
        if ind >= 0 and ind < len(self.submissions):
            return self.submissions[ind]
        return None

    def getUser(self, ind):
        """
            Returns the user with index "ind" or None if
            such user does not exist
        """
        if ind >= 0 and ind < len(self.users):
            return self.users[ind]
        return None
    
    def getSubmissionsByUser(self, i):
        """
            Returns a list of tuples (ind,sub) for each submission sent by user i.
            ind is the ID of the submission in the network, and sub is the actual
            Submission instance
        """
        return [(ind,sub) for ind, sub in enumerate(self.submissions) if sub.authorId == i]

    def getSubmissionsByTopic(self, i):
        """
            Returns a list of tuples (ind,sub) for each submission whose topic is i.
            ind is the ID of the submission in the network, and sub is the actual
            Submission instance
        """
        return [(ind,sub) for ind, sub in enumerate(self.submissions) if sub.topicId == i]

    def getAllSubmissions(self):
        """
            Returns a list of tuples (ind,sub) for every submission.
            ind is the ID of the submission in the network, and sub is the actual
            Submission instance
        """
        return [(ind, sub) for ind, sub in enumerate(self.submissions)]

    def getAllUsers(self):
        """
            Returns a list of tuples (ind,user) for every user.
            ind is the ID of the user in the network, and user is the actual
            User instance
        """
        return [(ind, user) for ind, user in enumerate(self.users)]

class Submission:
    """
        Represents a submission in a social network. A submission has the following attributes:
        - authorId:  the id of the author of the submission
        - topicId:   the topic that the submission belongs to
        - content:   a string of text with the content of the submission
        - comments:  a list of Submission instances representing comments made on this submission
        - parentId:  the id of a parent submission of this submission or -1 if there is no parent
    """
    def __init__(self, uid, topicid, content, parentId=-1):
        self.authorId = uid
        self.topicId = topicid
        self.content = content
        self.parentId = parentId
        self.comments = []

    def addComment(self, comment):
        """
            Adds a comment (Submission instance) and returns the ID of the comment relative to
            this submission
        """
        comment.topicId = self.topicId
        self.comments.append(comment)
        return len(self.comments)-1

    def getComment(self, ind):
        """
            Returns the submission with index "ind" or None if
            such submission does not exist
        """
        if ind >= 0 and ind < len(self.comments):
            return self.comments[ind]
        return None

    def getAllComments(self):
        """
            Returns a list of tuples (ind,sub) for every comment.
            ind is the ID of the submission in the network, and sub is the actual
            Submission instance
        """
        return [(ind, comment) for ind, comment in enumerate(self.comments)]

    def printSubmissionTree(self, padding=""):
        """
            Prints the author id and content of this Submission, and then recursively
            prints the same information for each of its children.
            padding is a string of text to be appended at the beginning of each line
        """
        print padding + "Author: %s" %self.authorId
        print padding + "Content: %s" %self.content

        if self.comments:
            print padding + "Children:"        
            for c in self.comments:
                c.printSubmissionTree(padding+"  ")

class User:
    """
        Represents a user in a social network. A user has a prior distribution over topics, drawn
        from a Hierarchical Dirichlet Process, and a learned distribution that gets updated (based
        on the reward function R) every time the user makes a submission to a topic.
    """
    
    SUBMISSION = "submission"
    COMMENT = "comment"
    LINEAR_REWARD = "linear_reward"
    LOG_REWARD = "log_reward"
    
    def __init__(self, eta, B, Delta, G_j, epsilon=0, phi=0, R=LINEAR_REWARD):
        """
            eta:      strength of prior belief
            B:        reward for making a new submission or comment in a social network
            Delta:    reward for receiving a reply to a submission
            G_j:      prior distribution over topics
            epsilon:  exploration factor for updating the learned distribution
            phi:      forgetting factor for updating the learned distribution
            R:        type of reward. Currently only 2 types are supported: linear and logarithmic rewards
        """
        self.propensity = Propensity(eta,G_j,epsilon,phi)
        self.priorTopicDistr = G_j
        self.baseReward = B
        self.replyReward = Delta
        self.rewardFunctions = {self.LINEAR_REWARD: self.linearReward, self.LOG_REWARD: self.logReward}
        self.R = R  
        
    def pickTopicFromPrior(self):
        """
            Samples a topic from the user's prior topic distribution
        """
        return self.priorTopicDistr.sampleIndex()

    def pickTopicFromLearned(self):
        """
            Samples a topic from the user's learned topic distribution
        """
        return self.propensity.sampleIndex()
    
    def updatePropensity(self, k, rewardType=SUBMISSION, commentCount = 0):
        """
            Updates the weight of a topic
            k:             the index of the topic to update
            rewardType:    the type of reward received, either SUBMISSION (for new submission) or
                           COMMENT (for getting a reply to an existing submission)
            commentCount:  the current number of replies received in a submission
        """
        reward = self.rewardFunctions[self.R](rewardType, commentCount)
        #print "Reward: %s" %reward
        #print "Propensity before updating: %s" %(self.propensity.weights[k])
        self.propensity.update(k, reward)
        #print "Propensity after updating: %s" %(self.propensity.weights[k])
        
    def linearReward(self, rewardType=SUBMISSION, commentCount = 0):
        """
            Reward is given by R(n) = B + Delta*n, where n is the number of comments in a
            a submission. In other words, when using this function, the user gets a reward
            of B for posting a submission, and an additional reward of Delta every time someone
            else writes a reply to that submission.
        """
        if rewardType == self.SUBMISSION:
            return self.baseReward
        elif rewardType == self.COMMENT:
            return self.replyReward
        else:
            raise Exception("Invalid argument for Reward Type: %s" %rewardType)

    def logReward(self, rewardType=SUBMISSION, commentCount = 0):
        """
            Reward is given by R(n) = B + Delta*ln(n+1), where n is the number of comments in a
            submission, and ln is the natural log function. The user gets a reward of B for
            posting a submission. When a reply is received, the previous reward ln((n-1)+1) gets
            replaced by ln(n+1)
        """
        if rewardType == self.SUBMISSION:
            return self.baseReward
        elif rewardType == self.COMMENT:
            if not isinstance(commentCount, int) or commentCount <= 0:
                raise Exception("Invalid argument for commentCount: %s" %commentCount)
            return self.replyReward*(log(commentCount+1) - log(commentCount))
        else:
            raise Exception("Invalid argument for Reward Type: %s" %rewardType)
        
class Propensity:
    """
        Represents the probabilistic weights for the set of actions (topics) available to a
        user. There are theoretically an infinite number of topics that the user can choose
        from, and, therefore, the weights must be evaluated lazily.

        Each action k has an initial weight of eta*pi_k, where eta is the strength of the
        prior belief and pi_k is the prior weight of action k.
    """
    sampledActions = []
    
    def __init__(self, eta, G_j, epsilon=0, phi=0):
        """
            eta:      strength of prior belief
            G_j:      prior distribution over topics
            epsilon:  exploration factor for updating the learned distribution
            phi:      forgetting factor for updating the learned distribution
        """
        self.priorBelief = eta
        self.priorTopicDistr = G_j
        self.explorationFactor = epsilon
        self.forgettingFactor = phi
        self.weights = []
        self.explorationFund = 0

    def __getitem__(self,k):
        """
            Returns the weight of the kth action. It may be necessary to first
            obtain this weight from the prior distribution over topics
        """
        while len(self.weights)-1 < k:
            ind = len(self.weights)
            self.weights.append(self.priorBelief*self.priorTopicDistr[ind])
            #print "Prior weight: %s" %self.priorTopicDistr[ind]
        return self.weights[k]
        
    def sampleWeight(self):
        """
            Returns the weights of one of the actions. The weight is sampled with
            probability w_k/T, where w_k is the weight of the kth action, and T is the
            sum of the weights of all the actions.
        """
        x=random.random()
        i = 0
        n = len(self.weights)-1
        cummulativeWeight = 0
        #Distribute the exploration weight evenly among all the actions that have been
        #taken up to this point in time by any of the users
        if len(self.sampledActions) == 0:
            explorationWeight = 0
        else:
            explorationWeight = self.explorationFund / len(self.sampledActions)
        #Compute the normalization factor. If no action has been sampled by this user yet,
        #then each action k has weight eta*pi_k, where pi_k is the weight of k in the
        #prior distribution. Then, the normalization factor is the sum(eta*pi_k) for all k,
        #which is equal to eta*sum(pi_k), which is just eta, since the sum of the previous
        #weights has to add up to 1.
        #If one or more actions have been already sampled, the normalization factor is the
        #sum of 1) the weights already in self.weights, 2) the exploration fund, and 3) the
        #weights of the actions that are not yet in self.weights. Each one of these actions
        #has weight eta*pi_k (because it hasn't been sampled yet), so the total weight of the
        #mass of actions not yet in self.weights is eta*(1-sum(pi_l)), where the sum is over all
        #the weights already in self.weights
        if n < 0:
            normalizationFactor = self.priorBelief
        else:
            normalizationFactor = sum(self.weights) + self.explorationFund + \
                                  self.priorBelief*(1-self.priorTopicDistr.cummulative[n])
        #Keep getting the next weight until the combined mass of the weights is less than the
        #random number x
        while True:
            w = self.__getitem__(i)
            if i in self.sampledActions:
                w += explorationWeight
            cummulativeWeight += w
            if x <= cummulativeWeight/normalizationFactor:
                if i not in self.sampledActions:
                    self.sampledActions.append(i)
                return w
            i += 1

    def sampleIndex(self):
        """
            Returns the index of one of the actions. The index is sampled with
            probability w_k/T, where w_k is the weight of the kth action, and T is the
            sum of the weights of all the actions.
        """
        x=random.random()
        i = 0
        n = len(self.weights)-1
        cummulativeWeight = 0
        if len(self.sampledActions) == 0:
            explorationWeight = 0
        else:
            explorationWeight = self.explorationFund / len(self.sampledActions)
        if n < 0:
            normalizationFactor = self.priorBelief
        else:
            normalizationFactor = sum(self.weights) + self.explorationFund + \
                                  self.priorBelief*(1-self.priorTopicDistr.cummulative[n])
        while True:
            w = self.__getitem__(i)
            if i in self.sampledActions:
                w += explorationWeight
            cummulativeWeight += w
            if x <= cummulativeWeight/normalizationFactor:
                if i not in self.sampledActions:
                    self.sampledActions.append(i)
                return i
            i += 1

    def update(self, k, reward):
        """
            Updates the weights q by applying the following formula:
            q_k = (1-forgettingFactor) * q_k + (1-explorationFactor)*reward,
            q_j = (1-forgettingFactor) * q_j if j != k and j has been previously sampled by any of the users

            Finally, the quantity (self.explorationFactor*reward) gets added to the exploration fund
        """
        #the index to be updated should already
        #have been sampled
        assert (len(self.weights)-1 >=  k) and (k in self.sampledActions)

        for i in self.sampledActions:
            self.__getitem__(i)
            self.weights[i] = (1-self.forgettingFactor)*self.weights[i]
        self.weights[k] += (1-self.explorationFactor)*reward
        self.explorationFund = (1-self.forgettingFactor)*self.explorationFund + (self.explorationFactor*reward)


def testPropensity():
    gamma = np.random.gamma(5,1)
    alpha_0 = np.random.gamma(1,1)
    eta = np.random.gamma(5,2)
    B = np.random.gamma(2.5,1)
    Delta = np.random.gamma(2.5,1)
    newSubmissionPrior = np.random.gamma(5,2)
    epsilon = np.random.beta(2,8)
    phi = np.random.beta(1,9)

    print "Creating HDP with parameters gamma=%s and alpha_0=%s" %(gamma,alpha_0)
    hdProcess = hdp.HDP(gamma, alpha_0)
    G_j1 = hdProcess.newSample()
    G_j2 = hdProcess.newSample()

    print "Creating propensity instances"
    prop1 = Propensity(eta, G_j1, epsilon, phi)
    prop2 = Propensity(eta, G_j2, epsilon, phi)

    print "Sampled topics should be empty"
    if prop1.sampledActions:
        print "Error. sampledActions is not empty. Exiting"
        sys.exit()
    print "Sampling topic from prop1"
    x1 = prop1.sampleIndex()
    print "Topic sampled: %s" %x1
    print "weight: %s" %prop1.weights[x1]
    print "weight of prior: %s" %prop1.priorTopicDistr.weights[x1]
    print "sampledActions should be the same for prop1 and prop2"
    if prop1.sampledActions != prop2.sampledActions or x1 not in prop1.sampledActions:
        print "Error. Invalid contents in sampledActions."
        print prop1.sampledActions
        print "Exiting"
        sys.exit()
    print prop1.sampledActions
    print "Sampling topic from prop2"
    x2 = prop2.sampleIndex()
    print "Topic sampled: %s" %x2
    print "weight: %s" %prop2.weights[x2]
    print "weight of prior: %s" %prop2.priorTopicDistr.weights[x2]
    print "sampledActions should be the same for prop1 and prop2"
    if prop1.sampledActions != prop2.sampledActions or x2 not in prop2.sampledActions:
        print "Error. Invalid contents in sampledActions."
        print prop2.sampledActions
        print "Exiting"
        sys.exit()
    print prop1.sampledActions

    print "Extending prop1 and prop2 to have at least 5 elements each"
    prop1[4]
    prop2[4]
    print "Weights of the first 5 actions"
    print prop1.weights[:5]
    print prop2.weights[:5]

    print "Updating weight 0 with a reward of 2"
    prop1.update(0,2)
    prop2.update(0,2)
    print "prop1 weights and exploration fund after updating"
    print prop1.weights[0:5]
    print prop1.explorationFund
    print "prop2 weights and exploration fund after updating"
    print prop2.weights[0:5]
    print prop2.explorationFund

    print "Sampling again"
    print "Sampling topic from prop1"
    x1 = prop1.sampleIndex()
    print "Topic sampled: %s" %x1
    print "weight: %s" %prop1.weights[x1]
    print "weight of prior: %s" %prop1.priorTopicDistr.weights[x1]
    print "sampledActions should be the same for prop1 and prop2"
    if prop1.sampledActions != prop2.sampledActions or x1 not in prop1.sampledActions:
        print "Error. Invalid contents in sampledActions."
        print prop1.sampledActions
        print "Exiting"
        sys.exit()
    print prop1.sampledActions
    print "Sampling topic from prop2"
    x2 = prop2.sampleIndex()
    print "Topic sampled: %s" %x2
    print "weight: %s" %prop2.weights[x2]
    print "weight of prior: %s" %prop2.priorTopicDistr.weights[x2]
    print "sampledActions should be the same for prop1 and prop2"
    if prop1.sampledActions != prop2.sampledActions or x2 not in prop2.sampledActions:
        print "Error. Invalid contents in sampledActions."
        print prop2.sampledActions
        print "Exiting"
        sys.exit()
    print prop1.sampledActions

    print "Done Testing"

def testUser():
    gamma = np.random.gamma(5,1)
    alpha_0 = np.random.gamma(1,1)
    eta = np.random.gamma(5,2)
    B = np.random.gamma(2.5,1)
    Delta = np.random.gamma(2.5,1)
    newSubmissionPrior = np.random.gamma(5,2)
    epsilon = np.random.beta(2,8)
    phi = np.random.beta(1,9)

    print "Creating HDP with parameters gamma=%s and alpha_0=%s" %(gamma,alpha_0)
    hdProcess = hdp.HDP(gamma, alpha_0)
    G_j1 = hdProcess.newSample()

    print "Creating user instance with parameters eta=%s, B=%s, Delta=%s, epsilon=%s, phi=%s" \
           %(eta,B,Delta,epsilon, phi)
    u = User(eta, B, Delta, G_j1, epsilon, phi)

    print "Picking topic from prior distribution"
    pTopic = u.pickTopicFromPrior()
    print "Topic selected: %s" %pTopic
    print "Picking topic from learned distribution"
    pTopic = u.pickTopicFromLearned()
    print "Topic selected: %s" %pTopic

    print "These are the propensities of the user so far:"
    print u.propensity.weights

    print "Updating propensity after a submission in topic %s" %pTopic
    u.updatePropensity(pTopic)
    print "Propensities after updating:"
    print u.propensity.weights
    pTopic = u.pickTopicFromLearned()
    print "Updating propensity after getting a comment in topic %s" %pTopic
    u.updatePropensity(pTopic, User.COMMENT)
    print "Propensities after updating:"
    print u.propensity.weights

    print "Done Testing"
    
if __name__=="__main__":
    testUser()
    testPropensity()
