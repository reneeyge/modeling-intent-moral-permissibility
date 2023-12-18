
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
from dijkstra import DijkstraSPF, Graph
import numpy as np


def get_utility(norm, k_T, n_T = 1, alpha_bro = 30, reg=True):
    """
    norm (bool): True when loved ones > anonymous people
    k_T (int): -1 when agent wants to kill people on track T, else 1
    n_T (int): # of people on track T (-1 when brother)
    alpha_bro (int): brother is equal to # alpha_bro of anonymous people
    Returns D_T (float): utility of n_T people on track T not being killed
    """
    
    if (not norm or reg) and n_T != -1:
        D_T = n_T * k_T * np.random.exponential()
    else:
        D_T = alpha_bro * k_T * np.random.exponential()
    
    return D_T


def sample_P_DN(a_k = 0.05, a_b = 0.1, a_norm = 0.55, n_M = 1, n_S = 1, reg=True):
    """
    a_k (float): prob that agent wants to kill people on the track, ind for each track
    a_b (float): prob that agent wants to kill as many people as possible
    a_norm (float): prob that agent follows norm
    n_M (int): # of people on main track
    n_S (int): # of people on side track
    Return samples (list[])
    """

    samples = []
    # sample 100 times
    for i in range(100):
        # prob that agent wants to kill as many people as possible
        samp_a_b = np.random.uniform(0,1)
        if samp_a_b < a_b:
            k_T = -1
        else:
            # prob that agent wants to kill people on the track
            samp_a_k = np.random.uniform(0,1)
            if samp_a_k < a_k:
                k_T = -1
            else:
                # agent won't kill anyone
                k_T = 1
        
        # agent values relatives > anonymous people or not
        samp_a_norm = np.random.uniform(0,1)
        if samp_a_norm < a_norm:
            norm = True
        else:
            norm = False

        main_side_utilities = (get_utility(norm, k_T, n_T = n_M, alpha_bro = 30, reg=reg), get_utility(norm, k_T, n_T = n_S, alpha_bro = 30, reg=reg))
        
        if k_T == -1:
            if abs(main_side_utilities[0]) >= abs(main_side_utilities[1]):
                pull_lever = False
            else:
                pull_lever = True
        else:
            if abs(main_side_utilities[0]) >= abs(main_side_utilities[1]):
                pull_lever = True
            else:
                pull_lever = False

        if n_M == -1:
            loc_bro = "M"
            samples.append((pull_lever, k_T, loc_bro, n_M, n_S))
        elif n_S == -1:
            loc_bro = "S"
            samples.append((pull_lever, k_T, loc_bro, n_M, n_S))

    return samples


def sample_P_norm(pull_lever, samples):
    """
    pull_lever (bool)
    samples (list[(tuple)])
    Returns float
    """
    num_norm = 0
    num_samps_of_action = 0
    for samp in samples:
        action, k_T, n_M, n_S = samp[0], samp[1], samp[3], samp[4]
        if action == pull_lever:
            num_samps_of_action += 1

            # get num of lives of each track
            if n_M != -1:
                main_lives = n_M
            else:
                main_lives = 1

            if n_S != -1:
                side_lives = n_S
            else:
                side_lives = 1

            net = main_lives - side_lives

            if action: # did pull lever
                if k_T == -1:
                    if net > 0:
                        num_norm += 1
                if k_T == 1:
                    if net < 0:
                        num_norm += 1
                if net == 0:
                    if k_T == 1:
                        if n_M == -1:
                            num_norm += 1
                    if k_T == -1:
                        if n_S == -1:
                            num_norm += 1
            else: # did not pull lever
                if k_T == -1:
                    if net < 0:
                        num_norm += 1
                if k_T == 1:
                    if net > 0:
                        num_norm += 1
                if net == 0:
                    if k_T == 1:
                        if n_S == -1:
                            num_norm += 1
                    if k_T == -1:
                        if n_M == -1:
                            num_norm += 1

    return num_norm/num_samps_of_action


def sample_P_I(pull_lever, samples):
    """
    action (bool)
    samples (list[(tuple)])
    Returns float
    """
    num_kill = 0
    num_samps_of_action = 0
    for samp in samples:
        action, k_T = samp[0], samp[1]
        
        if action == pull_lever: # action matches sample
            num_samps_of_action += 1

            if k_T == -1:
                num_kill += 1

    return num_kill/num_samps_of_action

def get_max_utility(utilities):
    """
    utilities (list[list[float]]): sublist is a list of utilities from a series of actions
    Returns: list[]
    """
    return max(utilities, key=sum)


def inf_diagram():
    # Create a decision network
    model = gum.InfluenceDiagram()
    # Add a decision node for test
    throw = gum.LabelizedVariable('Throw A','Throw the switch',2)
    throw.changeLabel(0,'Yes')
    throw.changeLabel(1,'No')
    model.addDecisionNode(throw)
    ut1to5 = gum.LabelizedVariable('Util1to5','Utility of 1 to 5',1)
    model.addUtilityNode(ut1to5)
    # Add an utility node 
    ut6 = gum.LabelizedVariable('Util6','Utility of 6',1)
    model.addUtilityNode(ut6)
    # Add connections between nodes
    model.addArc(model.idFromName('Throw A'), model.idFromName('Util1to5'))
    model.addArc(model.idFromName('Throw A'), model.idFromName('Util6'))

    # Add utilities
    model.utility(model.idFromName('Util1to5'))[{'Throw A':'Yes'}]=5
    model.utility(model.idFromName('Util6'))[{'Throw A':'No'}]=1
    model.utility(model.idFromName('Util1to5'))[{'Throw A':'No'}]=0
    model.utility(model.idFromName('Util6'))[{'Throw A':'Yes'}]= 0

    ie = gum.InfluenceDiagramInference(model)
    # Make an inference with default evidence
    ie.makeInference()
    print('--- Inference with default evidence ---')

    print('Maximum Expected Utility (MEU) : {0}'.format(ie.getMEU()))


def inf_diagram_5v1():
    graph = Graph()
    start = "throw_A"
    end1 = "util1to5"
    end2 = "util6"
    choice = "end result"
    graph.add_edge(start, end1, -5)
    graph.add_edge(start, end2, -1)
    graph.add_edge(end2, choice, 0)
    graph.add_edge(end1, choice, 0)
    dijkstra = DijkstraSPF(graph, start)
    print(dijkstra.get_distance(choice)*-1)
    print(dijkstra.get_path(choice))


def inf_diagram_5vB():
    graph = Graph()
    start = "throw_A"
    end1 = "util1to5"
    end2 = "utilB"
    choice = "end result"
    graph.add_edge(start, end1, -5)
    graph.add_edge(start, end2, -10)
    graph.add_edge(end2, choice, 0)
    graph.add_edge(end1, choice, 0)
    dijkstra = DijkstraSPF(graph, start)
    print(dijkstra.get_distance(choice)*-1)
    print(dijkstra.get_path(choice))


def inf_diagram_2v1():
    graph = Graph()
    start = "throw_A"
    end1 = "util1to2"
    end2 = "util3"
    choice = "end result"
    graph.add_edge(start, end1, -2)
    graph.add_edge(start, end2, -1)
    graph.add_edge(end2, choice, 0)
    graph.add_edge(end1, choice, 0)
    dijkstra = DijkstraSPF(graph, start)
    print(dijkstra.get_distance(choice)*-1)
    print(dijkstra.get_path(choice))


def inf_diagram_2vB():
    graph = Graph()
    start = "throw_A"
    end1 = "util1to2"
    end2 = "util3"
    choice = "end result"
    graph.add_edge(start, end1, -2)
    graph.add_edge(start, end2, -10)
    graph.add_edge(end2, choice, 0)
    graph.add_edge(end1, choice, 0)
    dijkstra = DijkstraSPF(graph, start)
    print(dijkstra.get_distance(choice)*-1)
    print(dijkstra.get_path(choice))


def inf_diagram_1v1():
    graph = Graph()
    start = "throw_A"
    end1 = "util1"
    end2 = "util2"
    choice = "end result"
    graph.add_edge(start, end1, -1)
    graph.add_edge(start, end2, -1)
    graph.add_edge(end2, choice, 0)
    graph.add_edge(end1, choice, 0)
    dijkstra = DijkstraSPF(graph, start)
    print(dijkstra.get_distance(choice)*-1)
    print(dijkstra.get_path(choice))


def inf_diagram_1vB():
    graph = Graph()
    start = "throw_A"
    end1 = "util1"
    end2 = "util2"
    choice = "end result"
    graph.add_edge(start, end1, -1)
    graph.add_edge(start, end2, -10)
    graph.add_edge(end2, choice, 0)
    graph.add_edge(end1, choice, 0)
    dijkstra = DijkstraSPF(graph, start)
    print(dijkstra.get_distance(choice)*-1)
    print(dijkstra.get_path(choice))


def inf_diagram_Bv1():
    graph = Graph()
    start = "throw_A"
    end1 = "util1"
    end2 = "util2"
    choice = "end result"
    graph.add_edge(start, end1, -10)
    graph.add_edge(start, end2, -1)
    graph.add_edge(end2, choice, 0)
    graph.add_edge(end1, choice, 0)
    dijkstra = DijkstraSPF(graph, start)
    print(dijkstra.get_distance(choice)*-1)
    print(dijkstra.get_path(choice))


def inf_diagram_1v2():
    graph = Graph()
    start = "throw_A"
    end1 = "util1"
    end2 = "util2to3"
    choice = "end result"
    graph.add_edge(start, end1, -1)
    graph.add_edge(start, end2, -2)
    graph.add_edge(end2, choice, 0)
    graph.add_edge(end1, choice, 0)
    dijkstra = DijkstraSPF(graph, start)
    print(dijkstra.get_distance(choice)*-1)
    print(dijkstra.get_path(choice))


def inf_diagram_Bv2():
    graph = Graph()
    start = "throw_A"
    end1 = "util1"
    end2 = "util2to3"
    choice = "end result"
    graph.add_edge(start, end1, -10)
    graph.add_edge(start, end2, -2)
    graph.add_edge(end2, choice, 0)
    graph.add_edge(end1, choice, 0)
    dijkstra = DijkstraSPF(graph, start)
    print(dijkstra.get_distance(choice)*-1)
    print(dijkstra.get_path(choice))


def inf_diagram_1v5():
    graph = Graph()
    start = "throw_A"
    end1 = "util1"
    end2 = "util2to6"
    choice = "end result"
    graph.add_edge(start, end1, -1)
    graph.add_edge(start, end2, -5)
    graph.add_edge(end2, choice, 0)
    graph.add_edge(end1, choice, 0)
    dijkstra = DijkstraSPF(graph, start)
    print(dijkstra.get_distance(choice)*-1)
    print(dijkstra.get_path(choice))


def inf_diagram_Bv5():
    graph = Graph()
    start = "throw_A"
    end1 = "util1"
    end2 = "util2to6"
    choice = "end result"
    graph.add_edge(start, end2, -10)
    graph.add_edge(start, end1, -5)
    graph.add_edge(end2, choice, 0)
    graph.add_edge(end1, choice, 0)
    dijkstra = DijkstraSPF(graph, start)
    print(dijkstra.get_distance(choice)*-1)
    print(dijkstra.get_path(choice))


inf_diagram()