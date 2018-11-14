
	# CSE411 - Simulation and Modeling
"""
Instructions:

This assignment is to "simulate" Google's pagerank as discussed in class.
This is a skeleton program where you need to add your code in places.

The program first reads web.txt to load the page network. Each row refers to a page
and 0/1 indicates which other pages it visits to (1 mean visit, else no).

It then populates transition probability matrix, P. Recall that a page visits to
the pages it has links to with uniform probability, and with some residual probability
it visits to any other page (may be to itself) uniformly at random. The parameter Alpha defines this split.

Given P, the program then analytically finds ranks of pages (i.e., pi's of underlying Markov chain
of pages). It also "simulates" a navigation process to compute the same.

The program then computes the difference between the two measurements and show them in a plot.

Add your codes at designated places.

Answer the following question at the end of your program

Q. Change the seed (currently 100) to different values. Do you see changes in results?
Can you explain why? Can you measure how much variation you see?

WORK INDEPEDENTLY. CODE SHARING IS STRICTLY PROHIBITED, AND IF HAPPENS WILL LEAD TO PENALTY.
"""

import numpy as np
import random
import matplotlib.pyplot as plt

def printP(P):
    for row in range(len(P)):
        for col in range(len(P[row])):
            print '%5.3f' % P[row][col],
        print


# Compute transition probability matrix
def populateP(file_name):
    alpha = 0.85
    P = None
    with open(file_name) as fhandle:
        alpho = 0.85
        total_pages = int(fhandle.readline())

        # Create Blank NxN Matrix
        P = [[0 for x in range(total_pages)] for y in range(total_pages)]

        row,col = 0,0
        for line in fhandle:
            line = list(line.replace(' ','').replace('\n',''))
            link_count = line.count('1')

            # http://en.wikipedia.org/wiki/PageRank#Damping_factor
            # Stanford pagerank algorithm :
            # At First, ((1-alpha)/total_pages) is Distributed to all page.
            # Thus, Creating Linked among all the pages on the web. This is known as teleportation link.
            # Then all page that are connected , (alpha/connected_link) are added to them.

            for word in line:
                if word == '1':
                    linked = (alpha/link_count) + ((1-alpha)/total_pages)
                    P[row][col] = linked
                else:
                    non_linked = (1-alpha)/total_pages
                    P[row][col] = non_linked

                col += 1
            row,col = row+1,0

    printP(P)
    return P

def computeDiff(pi, simpi):
    if len(pi) != len(simpi):
        raise Error('Pi dimension does not match!!')

    sumdiff = 0
    for i in range(len(pi)):
        sumdiff += abs(pi[i] - simpi[i])
    return sumdiff


# Compute rank analytically

def solveForPi(P):
    A = np.array(P)
    A = A.transpose()
    B = None

    total_pages = len(P)
    B = [0 for x in range(total_pages)]
    B[total_pages-1] = 1    # 0,0,0,0,0,0,0,0,0,1
    B = np.array(B,float)

    for x in range(total_pages):
        A[x][x] = A[x][x]-1
    for x in range(total_pages):
        A[total_pages-1][x] = 1

    pi = np.linalg.solve(A,B)
    return pi

# Compute rank by simulation
# Visit pages for 'iterations' number of times

def computeRankBySimulation(P, iterations):

    total_pages = len(P)
    simPi = [0 for i in range(total_pages)]
    page_no = 0
    count = 0

    for i in range(iterations):
        page_no=choosePage(P[page_no])
        if page_no>=0 and page_no<total_pages:
            simPi[page_no] += 1
            count +=1

    for i in range(len(P)):
       simPi[i]= simPi[i]/float(count)

    simPi = np.array(simPi)
    simPi= simPi/sum(simPi)
    return simPi

"""
    Sample X as defined by distribution "prob"
    prob is a list of probability values, such as [0.2, 0.5, 0.3].
    These are the values index positions take, that is, O happens with prob 0.2, 1 with 0.5 and so on.
"""
def choosePage(prob):
    U = random.random()
    P = 0
    for i in range(len(prob)):
        P = P + prob[i]
        if U < P:
            return i
    return len(prob)

def plotResult(P,analyticalPi):
    X,Y,Y1,Y2 = [],[],[],[]
    print "\n@ SEED = 100 :\n"
    for itr in range(1,11):
        itr = itr * 10000
        simulatedPi = computeRankBySimulation(P, itr)
        diff = computeDiff(analyticalPi, simulatedPi)
        print "%d\t%f" %(itr,diff)
        X.append(itr / 1000)
        Y1.append(diff)

    # ANSWER
    # Cahnging The Seed
    random.seed(10)
    print "\n@ SEED = 10 :\n"
    for itr in range(1,11):
        itr = itr * 10000
        simulatedPi = computeRankBySimulation(P, itr)
        diff = computeDiff(analyticalPi, simulatedPi)
        print "%d\t%f" %(itr,diff)
        Y2.append(diff)

    print "\nVARIATION :\n"
    for x in range(0,10):
    	Y.append(abs(Y1[x]-Y2[x]))
    	itr = (x+1) * 10000
    	print "%d\t%f" %(itr, Y[x])

    # Plotting Seed_10 Vs Seed_100
    seed_100, = plt.plot(X,Y1,'b-',label='SEED_100')
    seed_10, = plt.plot(X,Y2,'r-',label='SEED_10')
    variation, = plt.plot(X,Y,'g-',label='VARIATION')
    plt.legend([seed_10, seed_100, variation], ['@ SEED 10', '@ SEED 100', 'VARIATION'])
    plt.suptitle('Impact of Seed Changes', fontsize=16)

    # Plotting AnalyticalPi Vs SimulatedPi
    newPlot = plt.figure()
    plt2 = newPlot.add_subplot(111)
    plt2.plot(X,Y1)
    plt.suptitle('Differrence between analyticalPi & simulatedPi', fontsize=16)

    plt.xlabel("Iterations (1000's)")
    plt.ylabel("Pi difference")
    plt.show()

# main function
def main():
    P = populateP('web.txt')

    # Compute rank analytically
    analyticalPi = solveForPi(P)
    print "\nAnalyticalPi : \n",analyticalPi

    random.seed(100)
    simulatedPi = computeRankBySimulation(P, 1000)
    print "\nsimulatedPi : \n",simulatedPi

    # PLOTTING
    plotResult(P,analyticalPi)

if __name__ == "__main__":
    main()

    '''
    Your answer for the question goes here.

    ...
    	## We have changed the value of seed and noticed change in the result.
    	## We also calculated the variation and plotted the result.
    	## please see plotResult Functions.
	...
		## Explation

    		Due to change of seed to differet numbers, simPi changes
    		because python random is actually pseudo-random i.e it
    		generates same number sequence for a particular seed.
    		So choosePage always returns their diffrent number
    		sequence for differnt seed and so simuPi varies for
    		differnt seed.
    '''