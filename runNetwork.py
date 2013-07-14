#!/usr/bin/python

import argparse
import numpy as np
import hdp
import network
import random
import sys

def loadText(fname):
    """
        Load text for posts from a text file
    """
    text = []
    fh = open(fname)
    quote = ""
    for line in fh:
        line = line.strip()
        if line:
            quote += line + " "
        else:
            text.append(quote)
            quote = ""
    fh.close()
    return text

def getSubmissionDistribution(submissions, newSubmissionPrior):
    """
        Dummy function that just assigns probability of newSubmissionPrior to
        the "New Submission" action and evenly distributes the rest of the
        probability mass (1-newSubmissionPrior) among the existing submissions
    """
    if submissions:
        probMass = 1-newSubmissionPrior
        prob = probMass / len(submissions)
        distribution = [(i,prob) for i in submissions] + [((-1,None),newSubmissionPrior)]
        return distribution
        #prob = 1 / float(len(submissions))
        #distribution = [(i,prob) for i in submissions]
        #return distribution
    return [((-1,None),1)]

def pickSubmission(submissionDistr):
    x = random.random()
    cummulative = 0
    i=0
    while True:
        w = submissionDistr[i][1]
        cummulative += w
        if x <= cummulative:
            return submissionDistr[i][0]
        i += 1

def pickAParentSubmission(subm):
    """
        Given a Submission instance, pick a comment within the instance.
        Flip a fair coin. If the flip is 0, just pick the current submission,
        else, recursively and randomly pick one of the children of the current
        submission
    """
    rand = random.randint(0,1)
    if rand:
        children = subm[1].getAllComments()
        #if the submission has not comments, just return the submission
        if children:
            return pickAParentSubmission(random.choice(children))
        return subm
    return subm
        

def main():
    print "Starting"

    print "Loading text for posts"
    posts = loadText("quotes.txt")
    
    U = 5
    T = 20
    
    print "Number of users: %s" %U
    print "Number of rounds: %s" %T

    #model parameters
    gamma = np.random.gamma(5,1)
    alpha_0 = np.random.gamma(1,1)
    eta = np.random.gamma(5,2)
    B = np.random.gamma(2.5,1)
    Delta = np.random.gamma(2.5,1)
    newSubmissionPrior = np.random.gamma(5,2)
    epsilon = np.random.beta(2,8)
    phi = np.random.beta(1,9)

    #HDP
    print "Creating HDP with parameters gamma=%s and alpha_0=%s" %(gamma,alpha_0)
    hdProcess = hdp.HDP(gamma, alpha_0)
    #list of users
    users = []
    
    #sample topic distributions from the HDP
    #and assign them to users
    print "Creating user instances with parameters eta=%s, B=%s, Delta=%s, epsilon=%s, phi=%s" \
           %(eta,B,Delta,epsilon, phi)
    for i in range(U):
        topicDistr = hdProcess.newSample()
        #u = network.User(eta,B,Delta,topicDistr, epsilon, phi)
        u = network.User(eta,B,Delta,topicDistr, epsilon, phi, network.User.LOG_REWARD)
        users.append(u)

    #Start "Social Network" environment
    print "Creating social network environment"
    reddit = network.Network(users)
    
    for i in range(T):
        print "Round %s" %(i+1)
        #pick a user and a topic for that user
        userIndex = random.randint(0,len(users)-1)
        u = reddit.getUser(userIndex)
        topicId = u.pickTopicFromLearned()
        print "User %s will make a submission in topic %s" %(userIndex, topicId)
        
        #get all submissions in that topic
        #create a distribution over submissions (including the "New Submission" option)
        #and pick submission from the distribution
        submissions = reddit.getSubmissionsByTopic(topicId)
        print "There are %s submissions in this topic" %len(submissions)
        candidateSubmissions = [s for s in submissions
                                if not (s[1].authorId == userIndex and len(s[1].comments) == 0)]
        submissionDistr = getSubmissionDistribution(candidateSubmissions, newSubmissionPrior)
        subm = pickSubmission(submissionDistr)
        if subm[0] < 0:
            #new submission
            print "Making new submission"
            newSubmission = network.Submission(userIndex, topicId, random.choice(posts))
            reddit.addSubmission(newSubmission)
        else:
            #pick a random comment within the submission to reply to
            #and update propensity of the author of that comment
            print "Commenting in existing submission"
            parent = pickAParentSubmission(subm)
            #prevent user from replying to himself
            while parent[1].authorId == userIndex:
                parent = pickAParentSubmission(subm)
            newComment = network.Submission(userIndex, topicId, random.choice(posts),parent[0])
            parent[1].addComment(newComment)
            parentAuthor = reddit.users[parent[1].authorId]
            parentAuthor.updatePropensity(topicId, network.User.COMMENT, len(parent[1].comments))
        #update poster's propensity
        u.updatePropensity(topicId)

    print "End of simulation. Printing all comments in all topics and all user propensities"
    print "Number of submissions: %s" %len(reddit.submissions)
    for i in sorted(network.Propensity.sampledActions):
        print "---------------------------------"
        print "Topic %s" %i
        submissions = reddit.getSubmissionsByTopic(i)
        for s in submissions:
            s[1].printSubmissionTree()

    for i in range(len(users)):
        print "--------------------------"
        print "user %s" %i
        print "Propensity:"
        print users[i].propensity.weights
        
    print "Done"
    
if __name__=="__main__": main()
