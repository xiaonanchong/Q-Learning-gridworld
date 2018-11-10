##coding of the four direction
#	0
# 3	       1
#	2
## 0-up, 1-right, 2-down, 3-left

import numpy as np
import random as R

gamma = 0.3
alpha = 0.5

# s1-I[1-2,4-10] a-I[0,3], return: s2-I[1,10]
def takeaction(s1, a):
    #-1 means there is a wall in that direction
    transition_matrix = [[-1, 2, 5, -1], #1
                         [-1, 3, 6, 1],  #2
                         [0, 0, 0, 0], #3 terminal state
                         [-1, -1, 7, 3], #4
                         [1, 6, -1, -1], #5
                         [2, -1, 8, 5],  #6
                         [4, -1, 10, -1],#7
                         [6, 9, -1, -1], #8
                         [-1, 10, 11, 8],#9
                         [7, -1, -1, 9], #10
                         [0, 0, 0, 0]]   #11 terminal state
    if(s1 != 3 and s1 != 11):
        index=s1-1
        s2 = transition_matrix[index][a]
        if (s2==-1):
             return s1
        else: 
             return s2
    else:
        print('warning: you are in terminal state.')

def checkreward(s):
    if(s == 11):
        r = -100
    elif(s==3):
        r=10
    else:
        r=-1
    return r

Q = [ [ int(round(R.random()*10)) for i in range(4) ] for j in range(11) ]#[11.4]
Q[10] = [0 for i in range(4)] #Q(terminal-state, .)=0
Q[2] = [0 for i in range(4)]
print(Q)

for i in range(1000000):
    Q_privious =[ [Q[b][a] for a in range(4)] for b in range(11)]

    start_state = round(R.random()*10)+1 #randomly choose the initial state
    s = int(start_state)
    print('')
    print('episode: '+str(i)+' start with state: '+str(s))

    while (s != 11 and s != 3):

        desired_action = np.argmax(Q[s-1]) ##greedy method

        actions = [0,1,2,3] #corresponding to N, E, S, W - up, right, down, left
        actions.remove(desired_action)
        random_number = R.random()
        if(random_number <= 0.7):
            action_taken = desired_action
        elif(random_number <=0.8):
            action_taken = actions[0]
        elif(random_number <=0.9):
            action_taken = actions[1]
        else:
            action_taken = actions[2]
        
        print('desired action: '+ str(desired_action)+' | action taken: '+ str(action_taken))
           
        next_state = takeaction(s, action_taken)
        reward = checkreward(next_state)
        
        #update Q-value
        m = max(Q[next_state-1])

        Q[s-1][action_taken] = Q[s-1][action_taken] + alpha*(reward + 0.3*m - Q[s-1][action_taken])
        
        print(str(s)+' -> '+str(next_state)+ ' reward= '+str(reward) )  
        print('update Q[' + str(s)+'],[' + str(action_taken) +']' + ' = '+ str(Q[s-1][action_taken]))

        s= next_state

    ##condition for converge
    change = [[Q[j][i]-Q_privious[j][i] for i in range(4)] for j in range(11)]
    print(change)

print('final value function')
print(Q)


