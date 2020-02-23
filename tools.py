import os
import numpy as np
import pandas as pd
from itertools import combinations_with_replacement, permutations
import learner as lr
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle

np.random.seed(42)

### useful variables.
al = [1,0,0]
af = [0,1,0]
ar = [0,0,1]
actions = {'al':al, 'af': af, 'ar':ar}
actions_nums = {'al':0, 'af': 1, 'ar':2}
directions = {0: 'North', 1: 'East', 2: 'South', 3: 'West'}


def print_current_setup(arena_width, state_size, network_architecture, learning_rate, q_learning_rate, gamma, epochs, iterations, epsilon_greedy):
    s = '''
    --------------
    start training
    --------------
    arena size: %d, 
    state_size: %d
    network_architecture: %s
    learning_rate: %f
    q_ learning_rate: %f
    gamma: %f
    epochs: %d
    iterations: %d
    epsilon_greedy: %.1f
    --------------------
    ''' % (arena_width, state_size, network_architecture, learning_rate, q_learning_rate, gamma, epochs, iterations, epsilon_greedy)
    print(s)
    return s


class actor:
    def __init__(self,arena_width,  q_learning_rate, gamma):
        # Initialize q-table values to 0
        self.lr = q_learning_rate
        self.gamma = gamma
        self.Qmat = create_states(arena_width)[['al','af','ar']]

    def q_update(self, state, new_state, action, reward):
        self.Qmat.loc[state, action] = self.Qmat.loc[state, action] + self.lr * (reward + self.gamma * self.Qmat.loc[new_state, ['al', 'af','ar']].max() - self.Qmat.loc[state, action])

    def choose_action(self, state):
        return self.Qmat.loc[state,['al','af','ar']].astype('float').idxmax()


def create_states(arena_width):
    a = np.array(list(combinations_with_replacement(np.arange(arena_width),3)))
    a = a[a.sum(axis=1) < arena_width * 2 - 1, :]
    a = a[a.sum(axis=1) > arena_width - 2, :]
    center = np.floor(arena_width/2)
    b = np.array([[center, center, center]])

    for i in a:
        if len(np.unique(i)) > 1:
            c = list(permutations(i))
            c = np.unique(c, axis=0)
            if 'b' not in locals():
                b = c.copy()
            else:
                b = np.append(b,c, axis = 0)

    b = b[b[:,0] + b[:,2] == arena_width - 1, :]

    s = pd.DataFrame(data=b, columns=['L', 'F', 'R'], dtype='int').astype('str')
    s['state'] = s['L'] + s['F'] + s['R']
    s = s.sort_values('state')
    s = s.reset_index(drop=True)
    s = s.assign(al=0, af=0, ar=0)
    s.index = s['state']
    s[['L', 'F', 'R']] = s[['L', 'F', 'R']].astype('float')
    return s

def take_action(action, curr_dist, arena_width, direction):
    new_directions = {
        2: {0: 1, 1: 2, 2: 3, 3: 0},
        0: {0: 3, 3: 2, 2: 1, 1: 0},
    }

    if action == 0:
        ### turn left
        R = curr_dist[1]
        F = curr_dist[0]
        L = arena_width - 1 - R
    elif action == 1:
        ### step forward
        L = curr_dist[0]
        F = curr_dist[1] - 1
        R = curr_dist[2]
    elif action == 2:
        ### turn right
        L = curr_dist[1]
        F = curr_dist[0]
        R = arena_width - 1 - L
    elif action == 'r':
        L = np.random.randint(arena_width)
        F = np.random.randint(arena_width)
        R = arena_width - 1 - L

    if action == 1:
        nd = direction
    elif action != 'r':
        nd = new_directions[action][direction]
    else:
        nd = np.random.randint(4)

    ### check if the new sate is out of bounds.
    if L < 0 or F < 0 or R < 0:
        [L, F, R] = curr_dist
        nd = direction
    state = '%s%s%s' % (L, F, R)
    return [L, F, R], state, nd


def train_agent(q_learning_rate, gamma, network_architecture, arena_width, iterations, epochs, learning_rate):
    # 1: initialize learner and q matrix. (state)
    agent_actor = actor(arena_width, q_learning_rate, gamma)
    synapses_init = lr.initialize_synapses(network_architecture)  # initialize weights.
    with open("initial_synapses.txt", "wb") as fp:  # Pickling
        pickle.dump(synapses_init, fp)

    ### init learner (size)
    for e in range(epochs):
        df = pd.DataFrame(data=[], columns=['epoch', 'iteration', 'state', 'direction', 'action', 'loss'])
        t1 = time()
        with open("initial_synapses.txt", "rb") as fp:  # Unpickling
            synapses = pickle.load(fp)
        [r, f, l], s, d = take_action('r', 0, arena_width, 0) # choose random place to start.
        for iter in range(iterations):
            ### this loop check the possible outcomes of the next steps.
            xi = [r, f, l]  # current distances
            state = '%s%s%s' % tuple(np.array([r, f, l]).astype(int))  # current state as string.
            for actn, act in actions.items():
                x = np.append(np.array(xi)/ 4, act) # add the current action to test to the distances vector.
                y, new_state, _ = take_action(action=actions_nums[actn], curr_dist=xi, arena_width=arena_width, direction=d) # "take" the action and get the new state
                synapses_, layers_, layers_delta_, layers_error_, loss = lr.learner_ann(x=x, y=np.array(y)/4, synapses=synapses,
                                                                              alpha=learning_rate, num_of_iterations=1, train=False) # try to predict the outcome with the learner (which is an ANN).
                if new_state == state:
                    reward = 0
                else:
                    reward = loss * len(y)# the reward of the possible action.
                agent_actor.q_update(state=state, new_state=new_state, action=actn, reward=reward) # update the Q matrix

            ### Decide which action to take based on the Q matrix and current state.
            action2take = agent_actor.choose_action(state)

            ### Epsilon greedy, 10% will be random choice
            if np.random.randint(10) < epsilon_greedy * 10:
                action2take = list(actions_nums.keys())[np.random.randint(3)]

            ### Advance the state based on the chosen action
            [r, f, l], new_state, new_direction = take_action(action=actions_nums[action2take], curr_dist=xi, arena_width=arena_width,direction=d)
            xf = np.append(np.array([r,f,l]) / 4, actions[action2take])  # add the current action to test to the distances vector.

            ### Train the learner omly on the action that was taken.
            synapses, layers, layers_delta, layers_error, loss = lr.learner_ann(x=x, y=np.array(xf[:3])/4, synapses=synapses,
                                                                          alpha=learning_rate, num_of_iterations=1,
                                                                          train=True)  # try to predict the outcome with the learner (which is an ANN).
            df.loc[iter,:] = [e, iter, state, d, action2take, loss]
            d = new_direction

        del(synapses, layers, layers_delta, layers_error, loss)
        Qc = agent_actor.Qmat.copy()
        Qc['epoch'] = e
        Qc.to_csv(current_experiment + '/Qmats/epoch_%d.csv' % e)

        df[['epoch', 'iteration']] = df[['epoch', 'iteration']].astype(int)
        df['loss'] = df['loss'].astype(float)
        df.to_csv(current_experiment + '/summaries/epoch_%d.csv' %e)

        ### end of epoch
        print('Epoch: %d, time: %.2f' %(e, time()-t1))


### PARAMETERS TO CONTROL
### arena size
arena_width, arena_height = 5, 5 # only odd numbers
### Agent
state_size = arena_width * arena_height
### Learner architecture
input_size = 6
action_size = 3
network_architecture = [input_size, 16, action_size]
learning_rate = 1e-4
q_learning_rate = 1e-4
### Q learning discount factor.
gamma = 0.1
### training params
epochs = 100
iterations = 100
epsilon_greedy = .1

train = True
plot_summary = True

s = print_current_setup(arena_width, state_size, network_architecture, learning_rate, q_learning_rate, gamma, epochs, iterations, epsilon_greedy)

now = datetime.now()
dt_string = now.strftime("%Y%m%d_%H_%M")
current_experiment = 'experiment/%s' % dt_string

os.mkdir(current_experiment)
os.mkdir(current_experiment + '/Qmats')
os.mkdir(current_experiment + '/summaries')
os.mkdir(current_experiment + '/plots')
with open(current_experiment+'/experiment_parameters.txt', 'a') as the_file:
    the_file.write(s)

if train:
    train_agent(q_learning_rate, gamma, network_architecture, arena_width, iterations, epochs, learning_rate)

if plot_summary:
    ### load summaries and combine to one dataframe
    df_comb = pd.DataFrame()
    for df in os.listdir(current_experiment + '/summaries'):
        df = pd.read_csv(current_experiment + '/summaries/'+df, index_col=0)
        df_comb = pd.concat((df_comb, df), axis = 0)
    fig, ax = plt.subplots(1,1)
    sns.lineplot(x='iteration',y='loss', hue='epoch', data=df_comb, alpha=.5, ax=ax)#, legend='full')
    fig.savefig(current_experiment + '/plots/loss_epoch_iteration.jpg')

    ### load Q matriecs and combine to one dataframe
    df_qmats = pd.DataFrame()
    for df in os.listdir(current_experiment + '/Qmats'):
        df = pd.read_csv(current_experiment + '/Qmats/' + df, index_col=0)
        df_qmats = pd.concat((df_qmats, df), axis=0)

    fig, axs = plt.subplots(3,3)
    for i, e in enumerate(np.linspace(0, epochs-1, 9).astype(int)):
        axs[i//3, i%3].matshow(df_qmats.loc[df_qmats.loc[:, 'epoch'] == e, ['al', 'af', 'ar']])
        axs[i // 3, i % 3].set_title('epoch: %d'%e)
    fig.tight_layout()
    fig.savefig(current_experiment + '/plots/qmat.jpg')
plt.show()

