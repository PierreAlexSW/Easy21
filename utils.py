from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from Environment import Easy21

def policy_epsilon_greedy(N0, dealer, player_sum, action_value, number_action_value):
        """Pick action epsilon-greedily"""
        action_value_ij = action_value[dealer-1, player_sum]
        epsilon_t = N0 / (N0 + sum(number_action_value[dealer-1, player_sum,:]) )
        p = np.random.binomial(1,epsilon_t)
        max_index = np.argmax(action_value_ij)
        min_index = np.argmin(action_value_ij)
        if max_index!=min_index:
            if p == 0:
                index_action = max_index

            else:
                index_action = min_index
        else:
             index_action = np.random.binomial(1,0.5)
        return index_action
    
def montecarlo(iterations, N0, discount_factor):
    actions = ["Hit", "Stick"]
    action_value = np.array([[[0.0,0.0] for i in range(0,22)] for j in range(10)])
    number_action_value = np.array([[[0,0] for i in range(0,22)] for j in range(10)])
    deltas_tot = []
    for it in range(iterations):
        deltas = []
        """plays one episode"""
        game = Easy21()
        Gt = 0
        k = 0
        visits = []
        while game.isTerminal == False:
            last_state = game.state
            dealer, player_sum = last_state["dealer"], last_state["player_sum"]
            action_value_ij = action_value[dealer-1, player_sum]


            ##Pick action epsilon-greedily
            index_action = policy_epsilon_greedy(N0, dealer, player_sum, action_value, number_action_value)
            pick_action = actions[index_action]

            state,reward = game.step(pick_action)
            visits.append([last_state, index_action])
            number_action_value[dealer-1, player_sum, index_action]+=1
            Gt+=reward*discount_factor**k
            k+=1

        """episode ended"""
        for step in visits:
            state = step[0]
            action = step[1]
            dealer, player_sum = state["dealer"], state["player_sum"]
            delta_action_value = (Gt - action_value[dealer-1, player_sum, action] ) / number_action_value[dealer-1, player_sum, action]
            deltas.append(delta_action_value)
            action_value[dealer-1, player_sum, action] += delta_action_value
        deltas_tot.append(sum(deltas))
    return action_value, deltas_tot


##plotting
def plot_value_function(action_value, cm, degree = None):
    fig5 = plt.figure(figsize=(20, 10))
    ax = fig5.add_subplot(111, projection='3d')

    _x = np.arange(1,11,1)
    _y = np.arange(1,22,1)
    X,Y = np.meshgrid(_y,_x)
    Z = get_value(action_value)

    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap= cm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Player Sum')
    ax.set_ylabel('Dealer Showing')
    ax.set_zlabel('Value')
    ax.set_title("Value function")
    ax.view_init(ax.elev, degree)
    fig5.colorbar(surf)
    plt.show()

    
def get_value(action_value):
    Value = np.zeros((10,21))
    for i in range(len(Value)):
        for j in range(len(Value[0])):
            Value[i,j] = np.amax(action_value[i,j])
    return Value

def get_policy(action_value):
    Value = np.zeros((10,21))
    for i in range(len(Value)):
        for j in range(len(Value[0])):
            Value[i,j] = np.argmax(action_value[i,j])
    return Value

