import sys

'''
WRITE YOUR CODE BELOW.
'''
from numpy import zeros, float32
#  pgmpy͏︆͏󠄃͏󠄌͏󠄍͏︅͏︀͏︋͏︋͏󠄄͏︁͏︀
import pgmpy
import numpy as np
import random
from numpy.random import choice

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
#You are not allowed to use following set of modules from 'pgmpy' Library.͏︆͏󠄃͏󠄌͏󠄍͏︅͏︀͏︋͏︋͏󠄄͏︁͏︀
#
# pgmpy.sampling.*͏︆͏󠄃͏󠄌͏󠄍͏︅͏︀͏︋͏︋͏󠄄͏︁͏︀
# pgmpy.factors.*͏︆͏󠄃͏󠄌͏󠄍͏︅͏︀͏︋͏︋͏󠄄͏︁͏︀
# pgmpy.estimators.*͏︆͏󠄃͏󠄌͏󠄍͏︅͏︀͏︋͏︋͏󠄄͏︁͏︀

def make_security_system_net():
    """Create a Bayes Net representation of the above security system problem. 
    Use the following as the name attribute: "H","C", "M","B", "Q", 'K",
    "D"'. (for the tests to work.)
    """
    BayesNet = BayesianModel()
    BayesNet.add_node("H")
    BayesNet.add_node("C")
    BayesNet.add_node("M")
    BayesNet.add_node("B")
    BayesNet.add_node("Q")
    BayesNet.add_node("K")
    BayesNet.add_node("D")
    BayesNet.add_edge("H","Q")
    BayesNet.add_edge("C","Q")
    BayesNet.add_edge("Q","D")
    BayesNet.add_edge("M","K")
    BayesNet.add_edge("B","K")
    BayesNet.add_edge("K","D")
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏︅͏︀͏︋͏︋͏󠄄͏︁͏︀    
    # raise NotImplementedError
    return BayesNet


def set_probability(bayes_net):
    """Set probability distribution for each node in the security system.
    Use the following as the name attribute: "H","C", "M","B", "Q", 'K",
    "D"'. (for the tests to work.)
    """
    # TODO: set the probability distribution for each node͏︆͏󠄃͏󠄌͏󠄍͏︅͏︀͏︋͏︋͏󠄄͏︁͏︀
    # raise NotImplementedError    
    cpd_H=TabularCPD("H",2,values=[[0.5],[0.5]])
    cpd_C=TabularCPD("C",2,values=[[0.7],[0.3]])
    cpd_B=TabularCPD("B",2,values=[[0.5],[0.5]])
    cpd_M=TabularCPD("M",2,values=[[0.2],[0.8]])
    cpd_Q = TabularCPD('Q', 2, values=[[0.55, 0.9, 0.25, 0.05], \
                    [0.45, 0.1, 0.75, 0.95]], evidence=['H', 'C'], evidence_card=[2, 2])
    cpd_K = TabularCPD('K', 2, values=[[0.15, 0.01, 0.95, 0.75], \
                    [0.85, 0.99, 0.05, 0.25]], evidence=['B', 'M'], evidence_card=[2, 2])
    cpd_D = TabularCPD('D', 2, values=[[0.99, 0.02, 0.6, 0.35], \
                    [0.01, 0.98, 0.4, 0.65]], evidence=['Q', 'K'], evidence_card=[2, 2])
    bayes_net.add_cpds(cpd_H,cpd_C,cpd_Q,cpd_B,cpd_M,cpd_K,cpd_D)
    return bayes_net


def get_marginal_double0(bayes_net):
    """Calculate the marginal probability that Double-0 gets compromised.
    """
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏︅͏︀͏︋͏︋͏󠄄͏︁͏︀
    # raise NotImplementedError
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['D'], joint=False)
    double0_prob = marginal_prob['D'].values
    return double0_prob[1]


def get_conditional_double0_given_no_contra(bayes_net):
    """Calculate the conditional probability that Double-0 gets compromised
    given Contra is shut down.
    """
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏︅͏︀͏︋͏︋͏󠄄͏︁͏︀
    # raise NotImplementedError
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['D'], evidence={'C':0}, joint=False)
    double0_prob = marginal_prob['D'].values
    return double0_prob[1]


def get_conditional_double0_given_no_contra_and_bond_guarding(bayes_net):
    """Calculate the conditional probability that Double-0 gets compromised
    given Contra is shut down and Bond is reassigned to protect M.
    """
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏︅͏︀͏︋͏︋͏󠄄͏︁͏︀
    # raise NotImplementedError
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['D'], evidence={'C':0, 'B':1}, joint=False)
    double0_prob = marginal_prob['D'].values
    return double0_prob[1]


def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    BayesNet = BayesianModel()
    # TODO: fill this out͏︆͏󠄃͏󠄌͏󠄍͏︅͏︀͏︋͏︋͏󠄄͏︁͏︀
    # raise NotImplementedError
    BayesNet.add_node("A")
    BayesNet.add_node("B")
    BayesNet.add_node("C")
    BayesNet.add_node("AvB")
    BayesNet.add_node("BvC")
    BayesNet.add_node("CvA")
    BayesNet.add_edge("A","AvB")
    BayesNet.add_edge("B","AvB")
    BayesNet.add_edge("B","BvC")
    BayesNet.add_edge("C","BvC")
    BayesNet.add_edge("C","CvA")
    BayesNet.add_edge("A","CvA")

    skill_level_cpd = {
        'A': [0.15, 0.45, 0.30, 0.10],
        'B': [0.15, 0.45, 0.30, 0.10],
        'C': [0.15, 0.45, 0.30, 0.10]
    }

    # outcome_cpd = {
    #     'AvB': [[0.10, 0.10, 0.80],#[0,0]
    #             [0.20, 0.60, 0.20],#[0,1]
    #             [0.15, 0.75, 0.10],#[0,2]
    #             [0.05, 0.90, 0.05]],#[0,3]
    #     'BvC': [[0.10, 0.10, 0.80],
    #             [0.60, 0.20, 0.20],
    #             [0.75, 0.15, 0.10],
    #             [0.90, 0.05, 0.05]],
    #     'CvA': [[0.10, 0.10, 0.80],
    #             [0.20, 0.60, 0.20],
    #             [0.15, 0.75, 0.10],
    #             [0.05, 0.90, 0.05]]
    # }
    outcome_cpd = {
                #[0,0],[0,1],[0,2],[0,3], [1,0],[1,1],[1,2],[1,3],  [2,0],[2,1],[2,2],[2,3],  [3,0],[3,1],[3,2],[3,3]
        'AvB': [[0.10, 0.20, 0.15, 0.05,  0.60, 0.10, 0.20, 0.15,   0.75, 0.60, 0.10, 0.20,   0.90, 0.75, 0.60, 0.10],
                [0.10, 0.60, 0.75, 0.90,  0.20, 0.10, 0.60, 0.75,   0.15, 0.20, 0.10, 0.60,   0.05, 0.15, 0.20, 0.10],
                [0.80, 0.20, 0.10, 0.05,  0.20, 0.80, 0.20, 0.10,   0.10, 0.20, 0.80, 0.20,   0.05, 0.10, 0.20, 0.80]
                ],
        'BvC': [[0.10, 0.20, 0.15, 0.05,  0.60, 0.10, 0.20, 0.15,   0.75, 0.60, 0.10, 0.20,   0.90, 0.75, 0.60, 0.10],
                [0.10, 0.60, 0.75, 0.90,  0.20, 0.10, 0.60, 0.75,   0.15, 0.20, 0.10, 0.60,   0.05, 0.15, 0.20, 0.10],
                [0.80, 0.20, 0.10, 0.05,  0.20, 0.80, 0.20, 0.10,   0.10, 0.20, 0.80, 0.20,   0.05, 0.10, 0.20, 0.80]
                ],
        'CvA': [[0.10, 0.20, 0.15, 0.05,  0.60, 0.10, 0.20, 0.15,   0.75, 0.60, 0.10, 0.20,   0.90, 0.75, 0.60, 0.10],
                [0.10, 0.60, 0.75, 0.90,  0.20, 0.10, 0.60, 0.75,   0.15, 0.20, 0.10, 0.60,   0.05, 0.15, 0.20, 0.10],
                [0.80, 0.20, 0.10, 0.05,  0.20, 0.80, 0.20, 0.10,   0.10, 0.20, 0.80, 0.20,   0.05, 0.10, 0.20, 0.80]
                ]
    }
    cpd_A = TabularCPD('A', variable_card=4, values=[skill_level_cpd['A']])
    cpd_B = TabularCPD('B', variable_card=4, values=[skill_level_cpd['B']])
    cpd_C = TabularCPD('C', variable_card=4, values=[skill_level_cpd['C']])
    cpd_AvB = TabularCPD(variable='AvB', variable_card=3, values=outcome_cpd['AvB'],
                        evidence=['A','B'], evidence_card=[4,4])
    cpd_BvC = TabularCPD(variable='BvC', variable_card=3, values=outcome_cpd['BvC'],
                        evidence=['B','C'], evidence_card=[4,4])
    cpd_CvA = TabularCPD(variable='CvA', variable_card=3, values=outcome_cpd['CvA'],
                        evidence=['C','A'], evidence_card=[4,4])
    
    BayesNet.add_cpds(cpd_A,cpd_B,cpd_C,cpd_AvB,cpd_BvC,cpd_CvA)
    return BayesNet


def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0,0,0]
    # posterior = []
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏︅͏︀͏︋͏︋͏󠄄͏︁͏︀    
    # raise NotImplementedError
    solver = VariableElimination(bayes_net)
    posterior_distribution = solver.query(variables=['BvC'], evidence={'AvB': 0, 'CvA': 2})
    posterior = posterior_distribution.values
    return posterior # list 


def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """
    # sample = tuple(initial_state)    
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏︅͏︀͏︋͏︋͏󠄄͏︁͏︀
    # raise NotImplementedError
    # return sample
    if initial_state == None or len(initial_state)<6:
        sample = [0]*6
        for i in range(0,3):
            sample[i] = random.choices([0,1,2])[0]
        sample[3] = 0
        sample[4] = random.choices([0,1,2])[0]
        sample[5] = 2
        return tuple(sample)

    sample = tuple(initial_state)
    team_table = ['A','B','C','AvB','BvC','CvA']
    sample_index = random.choice([0,1,2,4])
    # sample_index = 4
    if sample_index == 4:
        match_table = bayes_net.get_cpds('BvC').values
        BvC_prob = [
            match_table[0,sample[1],sample[2]],
            match_table[1,sample[1],sample[2]],
            match_table[2,sample[1],sample[2]]
        ]
        sample = list(sample)
        sample[4] = random.choices([0,1,2],BvC_prob)[0]
        return tuple(sample)
    else:
        mappings = {
            0: ('B', 'C', 'AvB', 'CvA', 'BvC', [1, 2, 3, 5, 4]),
            1: ('A', 'C', 'AvB', 'BvC', 'CvA', [0, 2, 3, 4, 5]),
            2: ('A', 'B', 'CvA', 'BvC', 'AvB', [0, 1, 5, 4, 3])
        }
        match_team = team_table[sample_index]
        item1, item2, item3, item4, item5, item6 = mappings[sample_index]
        item1_idx, item2_idx, item3_idx, item4_idx, _ = item6
        match_team3_prob = bayes_net.get_cpds(item3).values
        match_team4_prob = bayes_net.get_cpds(item4).values
        match_team_prob = bayes_net.get_cpds(match_team).values
        match_team1_prob = bayes_net.get_cpds(item1).values
        match_team2_prob = bayes_net.get_cpds(item2).values
        team_prob = []
        team1_prob = match_team1_prob[initial_state[item1_idx]]
        team2_prob = match_team2_prob[initial_state[item2_idx]]
        for i in range(4):
            match_sample = match_team_prob[i]
            prob_1 = match_team3_prob[initial_state[item3_idx]][i][initial_state[item1_idx]] if item3[0] == match_team else match_team3_prob[initial_state[item3_idx]][initial_state[item1_idx]][i]
            prob_2 = match_team4_prob[initial_state[item4_idx]][i][initial_state[item2_idx]] if item5[0] == match_team else match_team4_prob[initial_state[item4_idx]][initial_state[item2_idx]][i]

            team_prob.append(match_sample * team1_prob * team2_prob * prob_1 * prob_2)

        updated_match_prob = [team_prob[i] / sum(team_prob) for i in range(4)]
        updated_sample = list(initial_state)
        updated_sample[sample_index] = random.choices([0, 1, 2, 3], updated_match_prob)[0]
        return tuple(updated_sample)


def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """

    A_cpd = bayes_net.get_cpds("A")      
    AvB_cpd = bayes_net.get_cpds("AvB")
    match_table = AvB_cpd.values
    team_table = A_cpd.values
    # sample = tuple(initial_state)
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏︅͏︀͏︋͏︋͏󠄄͏︁͏︀
    # raise NotImplementedError    
    # return sample
    #[a,b,c,avb,bvc,cva]
    if initial_state == None or len(initial_state)<6:
        sample = [0]*6
        for i in range(0,3):
            sample[i] = random.choices([0,1,2])[0]
        sample[3] = 0
        sample[4] = random.choices([0,1,2])[0]
        sample[5] = 2
        return tuple(sample)
    else:
        sample=[0]*6
        for i in range(0,3):
            sample[i]=random.choices([0,1,2])[0]
        sample[3]=0
        sample[5]=2
        sample[4]=random.choices([0,1,2])[0]
        sample=tuple(sample)
        current_prob = team_table[initial_state[0]] * team_table[initial_state[1]] * team_table[initial_state[2]] * match_table[initial_state[3],initial_state[0],initial_state[1]]* match_table[initial_state[4],initial_state[1],initial_state[2]] * match_table[initial_state[5],initial_state[2],initial_state[0]]
        proposed_prob = team_table[sample[0]] * team_table[sample[1]] * team_table[sample[2]] * match_table[sample[3],sample[0],sample[1]]* match_table[sample[4],sample[1],sample[2]] * match_table[sample[5],sample[2],sample[0]]
        
        acceptance_prob = min(1, proposed_prob / current_prob)
        if random.random() < acceptance_prob:
            print(proposed_prob)
            print(current_prob)
            print(acceptance_prob)
            return tuple(sample)
        else:
            print(proposed_prob)
            print(current_prob)
            print(acceptance_prob)
            return tuple(initial_state)
    
def compare_sampling(bayes_net, initial_state):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""    
    # Gibbs_count = 0
    # MH_count = 0
    # MH_rejection_count = 0
    # Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs 
    # MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH
    # # # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏︅͏︀͏︋͏︋͏󠄄͏︁͏︀
    # # raise NotImplementedError        
    # # return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count
    # # Gibbs_count = 0
    # # Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs
    # delta = 0.0001
    # N = 100
    # # Gibbs
    # sample_g=initial_state
    # diffs=[]
    # sample_rounds=100
    # target_index=4
    # result_g=[0,0,2]
    # count=0
    # while count<N:
    #     # print(Gibbs_convergence)
    #     # print(count)
    #     sample_g=Gibbs_sampler(bayes_net,sample_g)
    #     Gibbs_count=Gibbs_count+1
    #     # print(Gibbs_count,sample_rounds,N)
    #     if Gibbs_count>sample_rounds:
    #         result_g[sample_g[target_index]]= result_g[sample_g[target_index]]+1
    #         sum_g=sum(result_g)
    #         Gibbs_convergence=[result_g[0]/sum_g,result_g[1]/sum_g,result_g[2]/sum_g]
    #         diffs.append(Gibbs_convergence)
    #         count=0
    #         if Gibbs_count-sample_rounds>N:
    #             for i in range(N):
    #                 avg_sum=0
    #                 for j in range(3):
    #                     avg_sum=avg_sum+(abs(diffs[-(i+1)][j]-diffs[-(i+2)][j]))
    #                 avg=avg_sum/3
    #                 if avg<delta:
    #                     count=count+1
    # print(Gibbs_count, Gibbs_convergence)
    # # MH
    # MH_count = 0
    # MH_rejection_count = 0
    # MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH
    # sample_mh=initial_state
    # count=0
    # diffs=[]
    # result_mh=[0,0,2]
    # while count<N:
    #     # print(MH_convergence)
    #     sample_prior=sample_mh
    #     sample_mh=MH_sampler(bayes_net,sample_mh)
    #     MH_count=MH_count+1
    #     if sample_prior==sample_mh:
    #         MH_rejection_count=MH_rejection_count+1
    #     if MH_count>sample_rounds:
    #         result_mh[sample_mh[target_index]]=result_mh[sample_mh[target_index]]+1
    #         sum_mh=sum(result_mh)
    #         MH_convergence=[result_mh[0]/sum_mh,result_mh[1]/sum_mh,result_mh[2]/sum_mh]
    #         diffs.append(MH_convergence)
    #         count=0
    #         if MH_count-sample_rounds>N:
    #             for i in range(N):
    #                 avg_sum=0
    #                 for j in range(3):
    #                     avg_sum=avg_sum+(abs(diffs[-(i+1)][j]-diffs[-(i+2)][j]))
    #                 avg=avg_sum/3
    #                 if avg<delta:
    #                     count=count+1
    # return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count
    # delta = 0.0001
    # N = 100
    # # Gibbs
    # sample_g=initial_state
    # diffs=[]
    # sample_rounds=100
    # target_index=4
    # result_g=[0,0,2]
    # count=0
    # general_converage=[]
    # while count<N:
    #     # print(Gibbs_convergence)
    #     # print(count)
    #     sample_g=Gibbs_sampler(bayes_net,sample_g)
    #     Gibbs_count=Gibbs_count+1
    #     # print(Gibbs_count,sample_rounds,N)
    #     if Gibbs_count>sample_rounds:
    #         result_g[sample_g[target_index]]= result_g[sample_g[target_index]]+1
    #         for i in range(3):
    #             converage=result_g[i]/sum(result_g)
    #             diffs.append(converage)
    #         count=0
    #         if Gibbs_count-sample_rounds>N:
    #             for i in range(N):
    #                 avg_sum=0
    #                 for j in range(3):
    #                     avg_sum=avg_sum+(abs(diffs[-(i+1)][j]-diffs[-(i+2)][j]))
    #                 avg=avg_sum/3
    #                 if avg<delta:
    #                     break
    # # print(Gibbs_count, Gibbs_convergence)
    # # # MH
    # MH_count = 0
    # MH_rejection_count = 0
    # MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH
    # sample_mh=initial_state
    # count=0
    # diffs=[]
    # result_mh=[0,0,2]
    # while count<N:
    #     # print(MH_convergence)
    #     sample_prior=sample_mh
    #     sample_mh=MH_sampler(bayes_net,sample_mh)
    #     MH_count=MH_count+1
    #     if sample_prior==sample_mh:
    #         MH_rejection_count=MH_rejection_count+1
    #     if MH_count>sample_rounds:
    #         result_mh[sample_mh[target_index]]=result_mh[sample_mh[target_index]]+1
    #         for i in range(3):
    #             converage=result_g[i]/sum(result_g)
    #             diffs.append(converage)
    #         count=0
    #         if MH_count-sample_rounds>N:
    #             for i in range(N):
    #                 avg_sum=0
    #                 for j in range(3):
    #                     avg_sum+=(abs(diffs[-(i+1)][j]-diffs[-(i+2)][j]))
    #                 avg=avg_sum/3
    #                 if avg<delta:
    #                     break
    # return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0.0, 0.0, 0.0]
    MH_convergence = [0.0, 0.0, 0.0]
    delta = 0.01
    N = 1000
    Gibbs_flag = 1
    MH_flag = 1
    Gibbs_samples = [0.0, 0.0, 0.0]
    MH_samples = [0.0, 0.0, 0.0]
    Gibbs_ref = [0.0, 0.0, 0.0]
    MH_ref = [0.0, 0.0, 0.0]
    N_count = 0
    backup_state=list(initial_state)
    # initial_state1 = list(initial_state)
    # initial_state2 = list(initial_state)

    while Gibbs_flag:
        backup_state = Gibbs_sampler(bayes_net, backup_state)
        Gibbs_count += 1
        Gibbs_samples[backup_state[4]] += 1
        Gibbs_convergence = [sample/sum(Gibbs_samples) for sample in Gibbs_samples]
        if all((convergence - ref) < delta for convergence, ref in zip(Gibbs_convergence, Gibbs_ref)):
            N_count += 1
            if N < N_count:
                Gibbs_flag = 0
                N_count = 0
        else:
            Gibbs_ref = Gibbs_convergence
            N_count = 0

    while MH_flag:
        pre_state=backup_state
        backup_state = MH_sampler(bayes_net, backup_state)
        if backup_state == pre_state:
            MH_rejection_count += 1
        MH_count += 1
        MH_samples[backup_state[4]] += 1
        MH_convergence = [sample/sum(MH_samples) for sample in MH_samples]
        if all((convergence - ref) < delta for convergence, ref in zip(MH_convergence, MH_ref)):
            N_count += 1
            if N < N_count:
                MH_flag = 0
        else:
            MH_ref = MH_convergence
            N_count = 0

    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count

def sampling_question():
    """Question about sampling performance."""
    # TODO: assign value to choice and factor͏︆͏󠄃͏󠄌͏󠄍͏︅͏︀͏︋͏︋͏󠄄͏︁͏︀
    raise NotImplementedError
    choice = 2
    options = ['Gibbs','Metropolis-Hastings']
    factor = 0
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏︅͏︀͏︋͏︋͏󠄄͏︁͏︀
    raise NotImplementedError
