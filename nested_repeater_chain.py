"""
A simple script for evaluating throughput and latency of 2^k-link equidistant repeater chain with nested entanglement swapping (ES).

Two end nodes have 1 memory each. The intermediate (repeater) nodes have 2 memories. 
No memory decoherence, cutoff time, or operation error is considered.
Different levels of ES have different durations.

@author: Allen Zang (email: yzang@uchicago.edu)
"""


import numpy as np
from numpy.random import default_rng
import time


def get_level(index, tot_level):
    """Function to obtain the corresponding level of a node in a nested repeater chain.
    The node on the i-th level with in total k levels should have index in the form of (2^i)n + 2^(i-1).
    
    Args:
        index (int): the index of the node.
        tot_level (int): the total number of levels in a nested repeater chain.

    Return:
        level (int): the entanglement swapping level of the node.
    """

    for idx in range(1, tot_level+1):
        if index % (2**idx) == 2**(idx-1):
            return idx


class Node():
    """Repeater chain node class.
    
    Attributes:
        index (int): index of the node. For 2^k-link chain, indices are from 0 (leftmost end node) to 2^k (rightmost end node).
        rng: random number generator to determine if entanglement generation is successful.
        seed (int): seed for the rng.
        p_gen (float): success probability of entanglement generation.
        right_partner (Node): another node with larger index that is entangled with the current node.
        left_partner (Node): another node with smaller index that is entangled with the current node.
        tot_level (int): the total number of levels in the nested repeater chain. Corresponding to the `k` in the top doc string.
        level (int): the level of swapping for the node. Will determine time needed for swapping (level * unit time step).
        swap_cc_time (int): the time needed for entanglement swapping. Will determine the time for establishing entanglement after swapping.
        left_partner_idx_for_swap (int): index of a node on the left with which the node needs to be entangled to perform entanglement swapping.
        right_partner_idx_for_swap (int): index of a node on the right with which the node needs to be entangled to perform entanglement swapping.
        time_gen_left (int): time scheduled to establish entanglement with a node on the left.
        left_partner_to_be (Node): a node on the left that is scheduled to entangle with the node in the future.
        time_gen_right (int): time scheduled to establish entanglement with a node on the right.
        right_partner_to_be (Node): a node on the right that is scheduled to entangle with the node in the future.
    """

    def __init__(self, index, seed, p_gen, tot_level):
        self.index = index
        self.rng = default_rng(seed)
        
        self.tot_level = tot_level
        if self.index == 0 or self.index == 2**self.tot_level:
            self.level = 0
        else:
            self.level = get_level(self.index, self.tot_level)
        assert isinstance(self.level, int), "Node level must be an integer from 0 to total number of levels."

        self.swap_cc_time = self.level
        if self.level > 0:
            self.left_partner_idx_for_swap = self.index - 2**(self.level-1)
            self.right_partner_idx_for_swap = self.index + 2**(self.level-1)
        else:
            self.left_partner_idx_for_swap = -1
            self.right_partner_idx_for_swap = -1

        assert 0 <= p_gen <= 1, "Entanglement generation success probability invalid."
        self.p_gen = p_gen

        self.right_partner = None
        self.left_partner = None
        
        self.time_gen_left = None
        self.left_partner_to_be = None
        
        self.time_gen_right = None
        self.right_partner_to_be = None

    def ent_gen(self, current_time, node):
        """Method to determine entanglement generation success with direct neighbor.
        
        Will determine the success or failure using rng. 
        Nodes should only attempt entanglement generation with their right neighbor.
        If successful, will schedule the establishment of entanglement.

        Before using this method, need to make sure that the left neighbor is avaiable.
        Also need to make sure that the partner schedule entanglement if successful.
        """

        if self.rng.random() <= self.p_gen:
            time_to_entangle = current_time + 1
            self.right_schedule_entangle(node, time_to_entangle)
            node.left_schedule_entangle(self, time_to_entangle)
        
    def swap(self, current_time, left_node, right_node):
        """Method to perform entanglement swapping.
        
        Will modify state of the current node, and also other nodes.
        """
        
        assert left_node.index < right_node.index, "left_node should be on the left of right_node."
        
        time_to_entangle = current_time + self.swap_cc_time

        right_node.left_schedule_entangle(left_node, time_to_entangle)
        left_node.right_schedule_entangle(right_node, time_to_entangle)
        self.reinitialize()
        
    def left_schedule_entangle(self, node, time):
        """Method to schedule establishement of entanglement with a node on the left in the following.
        
        Will modify time_gen_left, left_partner_to_be.
        """

        assert node.index < self.index, "Left schedule expects node on the left."
        
        self.time_gen_left = time
        self.left_partner_to_be = node
        
    def right_schedule_entangle(self, node, time):
        """Method to schedule establishement of entanglement with a node on the right in the following.
        
        Will modify time_gen_right, right_partner_to_be.
        """

        assert node.index > self.index, "Right schedule expects node on the left."

        self.time_gen_right = time
        self.right_partner_to_be = node

    def entangle(self, node):
        """Method to establish entanglement with another node.

        Will modify the state of the current node.
        """
        
        if node.index < self.index:
            self.left_partner = node
        
            # re-initialize scheduling
            self.time_gen_left = None
            self.left_partner_to_be = None
        elif node.index > self.index:
            self.right_partner = node
            
            # re-initialize scheduling
            self.time_gen_right = None
            self.right_partner_to_be = None

    def reinitialize(self):
        """Method to reinitialize the node when the final entanglement distribution is successful.
        
        Will modify right_partner, right_partner_to_be, time_gen_right, left_partner, left_partner_to_be, time_gen_left
        """
        self.right_partner = None
        self.left_partner = None
        
        self.time_gen_left = None
        self.left_partner_to_be = None
        
        self.time_gen_right = None
        self.right_partner_to_be = None


# global simulation parameters
NUM_TRIALS = 1000  # number of runs of the simulation to obtain statistics
P_GEN = 0.3  # success probability for entanglement generation, non-negative and not greter than 1
CUT_OFF_TIME = 10000  # maximal time for continuous entanglement distribution
LEVEL = 3  # number of levels in the nested repeater chain

LEVEL_LIST = [1]
P_GEN_LIST = np.linspace(0.05, 0.5, 10)

def run_sim(cut_off_time, nodes, level):
    """Main simulation function.

    Args:
        cut_off_time (int): time for the continuous entanglement distribution to end in unit of elementary link 1-way CC time
        nodes (list[Node]): list of nodes in the nested repeater chain, should be indexed from 0 to 2^level
        level (int): total number of levels in the nested repeater chain

    Return:
        throughput (float): average number of distributed EPR pairs in unit time, equals to total number of EPR pairs divided by cut_off_time
        latency (int): time for the fisrt EPR pair to be distributed
    """

    assert len(nodes) == 2**level + 1, "Total number of nodes is not compatible with repeater chain level."

    t = 0  # initial time
    epr_count = 0  # number of EPR pairs distributed
    # latency = cut_off_time  # time for the first EPR pair to be distributed, if never successful, use cut-off time as lower bound

    while t <= cut_off_time:
        # first establish all entanglement links to be established
        for idx, node in enumerate(nodes):
            if node.time_gen_left == t:
                node.entangle(node.left_partner_to_be)
            if node.time_gen_right == t:
                node.entangle(node.right_partner_to_be)

        # determine if the final EPR pair has been established
        if nodes[0].right_partner == nodes[-1] and nodes[-1].left_partner == nodes[0]:
            epr_count += 1
            # reinitialize all nodes
            for node in nodes:
                node.reinitialize()

        # determine the first EPR pair and record latency
        # if epr_count == 1:
        #     latency = t

        # then perform entanglement operations
        for idx, node in enumerate(nodes):
            # determine if entanglement generation can be attempted
            # only if the node has no right partner, and the intended partner has no left partner
            if idx < 2**level:
                right_neighbor = nodes[idx+1]
                if (idx < 2**level) and (node.right_partner is None) and (right_neighbor.left_partner is None):
                    node.ent_gen(t, right_neighbor)
        
            # determine if to do entanglement swapping at repeater nodes when both sides are equipped with EPR pairs
            # only if desired links are established on both sides
            left_partner = node.left_partner
            right_partner = node.right_partner
            if (right_partner is not None) and (left_partner is not None) and (left_partner.index == node.left_partner_idx_for_swap) and (right_partner.index == node.right_partner_idx_for_swap):
                node.swap(t, left_partner, right_partner)

        t += 1

    throughput = epr_count / cut_off_time
    # return throughput, latency
    return throughput

# run the simulation
if __name__ == '__main__':
    
    for level in LEVEL_LIST:
        for p_gen in P_GEN_LIST:
            throughput_res = []
            # latency_res = []
        
            tick = time.time()
            for trial in range(NUM_TRIALS):
                # set up links 
                seed_start = (2**level+1) * trial  # seeds for nodes' rngs
                nodes = [Node(idx, seed_start+idx, p_gen, level) for idx in range(2**level+1)]
        
                # call the main simulation function
                # throughput, latency = run_sim(CUT_OFF_TIME, nodes, level)
                throughput = run_sim(CUT_OFF_TIME, nodes, level)
                throughput_res.append(throughput)
                # latency_res.append(latency)

    # assert len(throughput_res) == len(latency_res) == NUM_TRIALS, "The number of results should equal the number of simulation trials."

            sim_time = time.time() - tick
            print(f"Time taken for {NUM_TRIALS} trials of level-{level} repeater chain with {CUT_OFF_TIME} cut-off time: {(sim_time)*10**3:.03f}ms")
        
            # data analysis: mean throughput, mean latency, variance of throughput, and variance of latency
            avg_throughput = np.mean(throughput_res)
            print(f"Average throughput for level-{level} repeater chain with entanglement generation success probability {p_gen} is {avg_throughput}.")

    # var_throughput = np.var(throughput_res)
    # print(f"Variance of throughput for level-{LEVEL} repeater chain with entanglement generation success probability {P_GEN} is {var_throughput}.")


    # avg_latency = np.mean(latency_res)
    # print(f"Average latency for level-{LEVEL} repeater chain with entanglement generation success probability {P_GEN} is {avg_latency}.")

    # var_latency = np.var(latency_res)
    # print(f"Variance of latency for level-{LEVEL} repeater chain with entanglement generation success probability {P_GEN} is {var_latency}.")