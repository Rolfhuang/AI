# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""

import heapq
import os
import pickle
import math


class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""

        self.queue = []
        self.counter=0

    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """
        if not self.queue:
            return None
        else:
            return heapq.heappop(self.queue)[-1]
        # raise NotImplementedError

    def remove(self, node):
        """
        Remove a node from the queue.

        Hint: You might require this in ucs. However, you may
        choose not to use it or to define your own method.

        Args:
            node (tuple): The node to remove from the queue.
        """

        raise NotImplementedError

    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """

        # TODO: finish this function!
        # _,item = node
        # entry = 
        heapq.heappush(self.queue, (node[0],self.counter, node))
        self.counter += 1
        # heapq.heappush(self.queue,node)
        # raise NotImplementedError
        
    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n[-1] for n in self.queue]

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self.queue == other.queue

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in the queue.
        """

        return self.queue[0]


def breadth_first_search(graph, start, goal):
    """
    Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    if start == goal:
        return []
    visited = []
    pq = [start]
    path = {}
    path[start]=[start]
    find_short=False
    while pq:
        node = pq.pop(0)
        visited.append(node)
        if find_short:
            return path[goal]
        for neighbor in sorted(graph[node]):
            if neighbor not in visited and neighbor not in pq:
                path[neighbor]=path[node]+[neighbor]
                
                if neighbor == goal:
                    # return path[neighbor]
                    find_short=True
                pq.append(neighbor)

    # # raise NotImplementedError


def uniform_cost_search(graph, start, goal):
    """
    Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    if start==goal:
        return []
    qp=PriorityQueue()
    visited={start:0}
    qp.append((0,start))
    parent={}
    find_goal=False
    min_cost=0
    while qp.size() >0:
        weight, node = qp.pop()
        if find_goal and cost>min_cost:
            qp.clear()
            break
        for neighbor in graph[node]:
            cost=weight+graph.get_edge_weight(node,neighbor)
            if neighbor not in visited:
                if neighbor==goal:
                    find_goal=True
                    if min_cost == 0 or min_cost > cost:
                        min_cost = cost
                parent[neighbor]=node
                visited[neighbor]=cost
                qp.append((cost, neighbor))
            else:
                if visited[neighbor]>cost:
                    visited[neighbor]=cost
                    parent[neighbor]=node

    path=[]
    path.append(goal)
    add=goal
    while add != start:
        path.insert(0,parent[add])
        add=parent[add]
    return path
    # TODO: finish this function!
    # raise NotImplementedError


def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node
    """
    x1,y1=graph.nodes[v]['pos']
    x2,y2=graph.nodes[goal]['pos']
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    # return 0
    # # TODO: finish this function!
    # raise NotImplementedError


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    if start==goal:
        return []
    pq=PriorityQueue()
    pq.append((0,start))
    parent={start:0}
    visited={start:0}
    while pq:
        _ , node = pq.pop()
        if node == goal:
            pq.clear()
            break
        for next_node in graph[node]:
            new_cost = parent[node] + graph.get_edge_weight(node, next_node)
            if next_node not in parent or new_cost < parent[next_node]:
                visited[next_node] = node
                parent[next_node] = new_cost
                priority = new_cost + heuristic(graph, next_node, goal)
                pq.append((priority,next_node))
    path = []
    while node in visited:
        path.insert(0, node)
        node = visited[node]
    return path
    # TODO: finish this function!
    # raise NotImplementedError


def bidirectional_ucs(graph, start, goal):
    """
    Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    #ucs
    if start==goal:
        return []
    forward_parent={}
    backward_parent={}
    forward_visited={start:0}
    backward_visited={goal:0}
    forward_pq=PriorityQueue()
    backward_pq=PriorityQueue()
    forward_pq.append((0,start))
    backward_pq.append((0,goal))
    min_cost=0
    goal_node=None
    pre_total=0

    while forward_pq and backward_pq:
        f_weight,f_node=forward_pq.pop()
        b_weight,b_node=backward_pq.pop()

        for f_neighbors in graph[f_node]:
            f_cost=f_weight+graph.get_edge_weight(f_node,f_neighbors)
            if f_neighbors not in forward_visited:
                forward_parent[f_neighbors]=f_node
                forward_visited[f_neighbors]=f_cost
                forward_pq.append((f_cost,f_neighbors))
            else:
                if forward_visited[f_neighbors]>f_cost:
                    forward_visited[f_neighbors]=f_cost
                    forward_parent[f_neighbors]=f_node
        
        for b_neighbors in graph[b_node]:
            b_cost=b_weight+graph.get_edge_weight(b_node,b_neighbors)
            if b_neighbors not in backward_visited:
                backward_parent[b_neighbors]=b_node
                backward_visited[b_neighbors]=b_cost
                backward_pq.append((b_cost,b_neighbors))

            else:
                if backward_visited[b_neighbors]>b_cost:
                    backward_visited[b_neighbors]=b_cost
                    backward_parent[b_neighbors]=b_node

        if f_node in backward_visited or b_node in forward_visited:
            break
    
    same_node=[]
    for items in forward_visited.keys():
        if items in backward_visited:
            same_node.append(items)
    for item in same_node:
        pre_total=forward_visited[item]+backward_visited[item]
        if min_cost==0 or min_cost>pre_total:
            min_cost=pre_total
            goal_node=item
    f_path=[]
    b_path=[]
    goal_node_b=goal_node
    while goal_node is not None:
        f_path.insert(0, goal_node)
        goal_node = forward_parent.get(goal_node)

    while goal_node_b is not None:
        b_path.insert(0, goal_node_b)
        goal_node_b = backward_parent.get(goal_node_b)
    new_path=f_path[:-1] + b_path[::-1]
    return new_path

def bidirectional_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
    """
    Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    # if start == goal:#not finish
    #     return []
    # forward_parent={start:0}
    # backward_parent={goal:0}
    # forward_visited={}
    # backward_visited={}
    # forward_pq=PriorityQueue()
    # backward_pq=PriorityQueue()
    # forward_pq.append((0,start))
    # backward_pq.append((0,goal))
    # min_cost=0
    # pre_total=0
    # pre_totalb=0
    # goal_node=None
    # next_node=False
    # updated_cost=0
    # f_element=[]
    # b_element=[]
    # fnode=None
    # bnode=None
    # meet=False
    # goal_node_pre=None
    # f_backupPri=float('inf')
    # b_backupPri=float('inf')
    # s_g_est=heuristic(graph, start , goal)+heuristic(graph, goal , start)
    # while forward_pq and backward_pq:
    #     f_weight ,f_node=forward_pq.pop()
    #     b_weight ,b_node=backward_pq.pop()
    #     # if f_weight+b_weight>pre_total:
    #     #     break
    #     if f_node in backward_visited:
    #         if pre_total>f_weight:
    #             goal_node=f_node
    #             break
    #         # else:
    #         #     goal_node=fnode
    #         #     break
    #     # if meet:
    #     #     # if (f_weight+b_weight)>pre_total:
    #     #     #     goal_node=fnode
    #     #     #     break
    #     # if b_weight+f_weight>pre_total:
    #     #     break
    #     # if start=='n' and goal=='a':
    #     #     print(forward_visited)
    #     #     print(backward_visited)
    #     # if f_node in backward_visited and forward_parent[f_node]+backward_parent[f_node]<pre_total:
    #     #     pre_total=forward_parent[f_node]+backward_parent[f_node]
    #     # if f_node in backward_visited or b_node in forward_visited:
    #     #     goal_node=f_node
    #     #     if f_weight+b_weight>2*pre_total:
    #     #         forward_pq.clear()
    #     #         backward_pq.clear()
    #     #         break
    #     #     try:
    #     #         stop_correct=max(min([item for (item,_) in forward_pq]),min([item for (item,_) in backward_pq]))
    #     #     except ValueError:
    #     #         continue
    #     #     if pre_total<=stop_correct:
    #     #         goal_node=f_node
    #     #         break
    #     # if meet and f_weight+b_weight>=pre_total:
    #     #     forward_pq.clear()
    #     #     backward_pq.clear()
    #     #     break
    #     # elif b_node in forward_visited and forward_parent[b_node]+backward_parent[b_node]<pre_total:
    #     #     goal_node=b_node
    #     #     pre_total=forward_parent[b_node]+backward_parent[b_node]

    #     for f_neighbors in graph[f_node]:
    #         new_cost = forward_parent[f_node] + graph.get_edge_weight(f_node, f_neighbors)
    #         if f_neighbors not in forward_parent or new_cost < forward_parent[f_neighbors]:
    #             if forward_parent[f_node]==0 and f_neighbors == goal:
    #                 next_node=True
    #             #     goal_node_pre=f_neighbors
    #                 # meet=True
    #                 # pre_total=new_cost+backward_parent[f_neighbors]
    #                 # if s_g_est>2*pre_total:
    #                 #     goal_node=f_neighbors
    #                 #     break
    #                 # if min_cost==0 or min_cost>pre_total:
    #                 #     min_cost=pre_total
    #                 # pre_total=new_cost + heuristic(graph, f_neighbors , goal)
    #                 # if min_cost==0 or min_cost>pre_total:
    #                 #     min_cost=pre_total
    #             #         if pre_total>min_cost:
    #             #             goal_node=f_neighbors
    #             #             break
    #             # fnode=f_neighbors
    #             forward_visited[f_neighbors] = f_node
    #             forward_parent[f_neighbors] = new_cost
    #             priority = new_cost + heuristic(graph, f_neighbors , goal)
    #             forward_pq.append((priority,f_neighbors))
    #             if f_neighbors in backward_visited:
    #                 meet=True
    #                 updated_cost=new_cost+backward_parent[f_neighbors]
    #                 if pre_total==0 or pre_total>updated_cost:
    #                     pre_total=updated_cost
    #                     goal_node=f_neighbors
    #             # if f_neighbors in backward_visited and new_cost+backward_parent[f_neighbors]<pre_total:
    #             #     pre_total=forward_parent[f_neighbors]+backward_parent[f_neighbors]
    #                 # goal_node=f_neighbors
    #             # if f_neighbors in backward_visited:
    #             #     if min_cost==0 or min_cost>(new_cost+backward_parent[f_neighbors]):
    #             #         min_cost=new_cost+backward_parent[f_neighbors]
    #             # if f_neighbors in backward_visited and new_cost+backward_parent[f_neighbors]<pre_total:
    #             #     goal_node=f_neighbors
    #             #     pre_total=new_cost+backward_parent[f_neighbors]
    #     for b_neighbors in graph[b_node]:
    #         new_cost = backward_parent[b_node] + graph.get_edge_weight(b_node, b_neighbors)
    #         if b_neighbors not in backward_parent or new_cost < backward_parent[b_neighbors]:
    #             if backward_parent[b_node]==0 and b_neighbors == start:
    #                 next_node=True
    #             # if b_neighbors in forward_visited:
    #             #     pre_total=new_cost+forward_parent[b_neighbors]
    #             #     if min_cost==0 or min_cost>pre_total:
    #             #         min_cost=pre_total
    #             # bnode=b_neighbors
    #             backward_visited[b_neighbors] = b_node
    #             backward_parent[b_neighbors] = new_cost
    #             priority = new_cost + heuristic(graph, b_neighbors , start)
    #             backward_pq.append((priority,b_neighbors))
    #             # if b_neighbors in forward_visited and forward_parent[b_neighbors]+new_cost<pre_total:
    #             #     goal_node=b_neighbors
    #             #     pre_total=forward_parent[b_neighbors]+new_cost

    #     # if forward_pq.pop()[0]+backward_pq.pop()[0]>pre_total:
    #     #     break

    #     # top1,top2=forward_pq.top()[0],backward_pq.top()[0]
    #     # if f_weight+b_weight>=pre_total:
    #     #     break
    #     # if fnode in backward_visited and bnode in forward_visited:
    #     #     goal_node=fnode
    #     #     break
    # same_node=[]
    # for items in forward_visited.keys():
    #     if items in backward_visited:
    #         same_node.append(items)
    # for item in same_node:
    #     pre_total=forward_parent[item]+backward_parent[item]
    #     if min_cost==0 or min_cost>pre_total:
    #         min_cost=pre_total
    #         goal_node=item
    # # if start=='n' and goal=='a':
    # #     print(forward_visited)
    # #     print(backward_visited)
    # # for item in set(f_element+b_element):
    # #     if item in forward_visited and item in backward_visited:
    # #         total=forward_parent[item]+backward_parent[item]
    # #         if min_cost==0 or min_cost>total:
    # #             min_cost=total
    # #             goal_node=item
    # f_path=[]
    # b_path=[]

    # if not next_node:
    #     goal_node_b=goal_node
    # else:
    #     goal_node=start
    #     goal_node_b=goal
    # while goal_node is not None:
    #     if goal_node!=0:
    #         f_path.insert(0, goal_node)
    #         goal_node = forward_visited.get(goal_node)
    #     else:
    #         goal_node=None

    # while goal_node_b is not None:
    #     if goal_node_b!=0:
    #         b_path.insert(0, goal_node_b)
    #         goal_node_b = backward_visited.get(goal_node_b)
    #     else:
    #         goal_node_b=None
    # if not next_node:
    #     new_path=f_path[:-1] + b_path[::-1]
    # else:
    #     new_path=f_path+b_path
    # return new_path
    #----------------------------------------------------------------------------------------------------------
    if start == goal:#not finish
        return []
    forward_parent={start:0}
    backward_parent={goal:0}
    forward_visited={}
    backward_visited={}
    forward_pq=PriorityQueue()
    backward_pq=PriorityQueue()
    forward_pq.append((0,0,start))
    backward_pq.append((0,0,goal))
    min_cost=0
    pre_total=0
    pre_totalb=0
    goal_node=None
    next_node=False
    updated_cost=0
    f_element=[]
    b_element=[]
    fnode=None
    bnode=None
    meet=False
    while forward_pq and backward_pq:
        f_priority, f_weight ,f_node=forward_pq.pop()
        b_priority, b_weight ,b_node=backward_pq.pop()

        for f_neighbors in graph[f_node]:
            new_cost = f_weight + graph.get_edge_weight(f_node, f_neighbors)
            if f_neighbors not in forward_parent or new_cost < forward_parent[f_neighbors]:
                if forward_parent[f_node]==0 and f_neighbors == goal:
                    next_node=True
                forward_visited[f_neighbors] = f_node
                priority = new_cost + heuristic(graph, f_neighbors , goal)
                forward_parent[f_neighbors] = new_cost
                forward_pq.append((priority,new_cost,f_neighbors))
                if f_neighbors in backward_visited:
                    meet=True
                    updated_cost=new_cost+backward_parent[f_neighbors]
                    if pre_total==0 or pre_total>updated_cost:
                        pre_total=updated_cost
                        goal_node=f_neighbors
        for b_neighbors in graph[b_node]:
            new_cost = backward_parent[b_node] + graph.get_edge_weight(b_node, b_neighbors)
            if b_neighbors not in backward_parent or new_cost < backward_parent[b_neighbors]:
                if backward_parent[b_node]==0 and b_neighbors == start:
                    next_node=True
                backward_visited[b_neighbors] = b_node
                priority = new_cost + heuristic(graph, b_neighbors , start)
                backward_parent[b_neighbors] = new_cost
                backward_pq.append((priority,new_cost,b_neighbors))

        if f_node in backward_visited or b_node in forward_visited:
            break
    same_node=[]
    for items in forward_visited.keys():
        if items in backward_visited:
            same_node.append(items)
    for item in same_node:
        pre_total=forward_parent[item]+backward_parent[item]
        if min_cost==0 or min_cost>pre_total:
            min_cost=pre_total
            goal_node=item
    for item in set(f_element+b_element):
        if item in forward_visited and item in backward_visited:
            total=forward_parent[item]+backward_parent[item]
            if min_cost==0 or min_cost>total:
                min_cost=total
                goal_node=item
    f_path=[]
    b_path=[]

    if not next_node:
        goal_node_b=goal_node
    else:
        goal_node=start
        goal_node_b=goal
    while goal_node is not None:
        if goal_node!=0:
            f_path.insert(0, goal_node)
            goal_node = forward_visited.get(goal_node)
        else:
            goal_node=None

    while goal_node_b is not None:
        if goal_node_b!=0:
            b_path.insert(0, goal_node_b)
            goal_node_b = backward_visited.get(goal_node_b)
        else:
            goal_node_b=None
    if not next_node:
        new_path=f_path[:-1] + b_path[::-1]
    else:
        new_path=f_path+b_path
    return new_path
    # # TODO: finish this function!
    # raise NotImplementedError


def tridirectional_search(graph, goals):
    """
    Exercise 3: Tridirectional UCS Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # if 1<len(set(goals))<3:
    #     return [item for item in set(goals)]
    # elif len(set(goals))<=1:
    #     return []
    # pq1,pq2,pq3=PriorityQueue(),PriorityQueue(),PriorityQueue()
    # node1,node2,node3=goals
    # parent1,parent2,parent3={node1:0},{node2:0},{node3:0}
    # visited1,visited2,visited3={},{},{}
    # pq1.append((0,node1))
    # pq2.append((0,node2))
    # pq3.append((0,node3))
    # node1_in23=False
    # node2_in13=False
    # node3_in12=False
    # while pq1 and pq2 and pq3:
    #     _,node1=pq1.pop()
    #     _,node2=pq2.pop()
    #     _,node3=pq3.pop()
    #     # if (node1 in parent2 and node1 in parent3) or (node2 in parent1 and node2 in parent3) or (node3 in parent2 and node3 in parent1):

    #     for neighbors1 in graph[node1]:
    #         cost=parent1[node1]+graph.get_edge_weight(node1,neighbors1)
    #         if neighbors1 not in visited1 or cost < parent1[neighbors1]:
    #             parent1[neighbors1]=cost
    #             visited1[neighbors1]=node1
    #             pq1.append((cost,neighbors1))
    #     for neighbors2 in graph[node2]:
    #         cost=parent2[node2]+graph.get_edge_weight(node2,neighbors2)
    #         if neighbors2 not in visited2 or cost < parent2[neighbors2]:
    #             parent2[neighbors2]=cost
    #             visited2[neighbors2]=node2
    #             pq2.append((cost,neighbors2))
    #     for neighbors3 in graph[node3]:
    #         cost=parent3[node3]+graph.get_edge_weight(node3,neighbors3)
    #         if neighbors3 not in visited3 or cost < parent3[neighbors3]:
    #             parent3[neighbors3]=cost
    #             visited3[neighbors3]=node3
    #             pq3.append((cost,neighbors3))

    #     if node1 in visited2 or node1 in visited3:
    #         node1_in23=True
    #         goal_node=node1
    #         break
    #     elif node2 in visited1 or node2 in visited3:
    #         node2_in13=True
    #         goal_node=node2
    #         break
    #     elif node3 in visited2 or node3 in visited1:
    #         node3_in12=True
    #         goal_node=node3
    #         break
    # same_node=[]
    # if 1<len(set(goals))<3:
    #     return [item for item in set(goals)]
    # elif len(set(goals))<=1:
    #     return []
    # # node1,node2,node3=goals
    # if goals==['g', 'c', 'v']:
    #     print(True)
    # pq1,pq2,pq3=PriorityQueue(),PriorityQueue(),PriorityQueue()
    # start_node1,start_node2,start_node3=goals
    # parent1,parent2,parent3={start_node1:0},{start_node2:0},{start_node3:0}
    # visited1,visited2,visited3={},{},{}
    # pq1.append((0,start_node1))
    # pq2.append((0,start_node2))
    # pq3.append((0,start_node3))
    # node1_in23=False
    # node2_in13=False
    # node3_in12=False
    # min_cost=0
    # goal_node=None
    # pre_total=0
    # middle_node=None
    # new_path=[]
    # new_path_13=[]
    # while pq1 and pq3:
    #     weight1,node1=pq1.pop()
    #     weight3,node3=pq3.pop()

    #     for f_neighbors in graph[node1]:
    #         f_cost=weight1+graph.get_edge_weight(node1,f_neighbors)
    #         if f_neighbors not in visited1:
    #             visited1[f_neighbors]=node1
    #             parent1[f_neighbors]=f_cost
    #             pq1.append((f_cost,f_neighbors))
    #         else:
    #             if parent1[f_neighbors]>f_cost:
    #                 parent1[f_neighbors]=f_cost
    #                 visited1[f_neighbors]=node1
        
    #     for b_neighbors in graph[node3]:
    #         b_cost=weight3+graph.get_edge_weight(node3,b_neighbors)
    #         if b_neighbors not in visited3:
    #             visited3[b_neighbors]=node3
    #             parent3[b_neighbors]=b_cost
    #             pq3.append((b_cost,b_neighbors))

    #         else:
    #             if parent3[b_neighbors]>b_cost:
    #                 parent3[b_neighbors]=b_cost
    #                 visited3[b_neighbors]=node3
    #     if node1 in visited3 or node3 in visited1:
    #         if start_node2 in visited1 or start_node2 in visited3:
    #             goal_node=start_node2
    #             break
    #         else:
    #             # same_node=[]
    #             # for items in visited1.keys():
    #             #     if items in visited3:
    #             #         same_node.append(items)
    #             # for item in same_node:
    #             #     pre_total=parent1[item]+parent3[item]
    #             #     if min_cost==0 or min_cost>pre_total:
    #             #         min_cost=pre_total
    #             #         goal_node=item
    #             goal_node=node1
    #             middle_node=goal_node
    #             f_path=[]
    #             b_path=[]
    #             goal_node_b=goal_node
    #             while goal_node is not None:
    #                 f_path.insert(0, goal_node)
    #                 goal_node = visited1.get(goal_node)
    #                 if goal_node==start_node1:
    #                     f_path.insert(0, goal_node)
    #                     goal_node=None
    #             while goal_node_b is not None:
    #                 b_path.insert(0, goal_node_b)
    #                 goal_node_b = visited3.get(goal_node_b)
    #                 if goal_node_b==start_node3:
    #                     b_path.insert(0, goal_node_b)
    #                     goal_node_b=None
    #             new_path_13=f_path[:-1] + b_path[::-1]
    #             break
    # # if goal_node!=start_node2 and goal_node!=None:
    # #     new_pq1=PriorityQueue()
    # #     new_pq1.append((0,start_node1))
    # #     new_parent1={start_node1:0}
    # #     new_visited1={}
    # #     while new_pq1 and pq2:
    # #         weight1,node1=new_pq1.pop()
    # #         weight2,node2=pq2.pop()

    # #         for f_neighbors in graph[node1]:
    # #             f_cost=weight1+graph.get_edge_weight(node1,f_neighbors)
    # #             if f_neighbors not in visited1:
    # #                 new_visited1[f_neighbors]=node1
    # #                 new_parent1[f_neighbors]=f_cost
    # #                 new_pq1.append((f_cost,f_neighbors))
    # #             else:
    # #                 if new_parent1[f_neighbors]>f_cost:
    # #                     new_parent1[f_neighbors]=f_cost
    # #                     new_visited1[f_neighbors]=node1
            
    # #         for b_neighbors in graph[node3]:
    # #             b_cost=weight3+graph.get_edge_weight(node3,b_neighbors)
    # #             if b_neighbors not in visited3:
    # #                 visited3[b_neighbors]=node3
    # #                 parent3[b_neighbors]=b_cost
    # #                 pq3.append((b_cost,b_neighbors))

    # #             else:
    # #                 if parent3[b_neighbors]>b_cost:
    # #                     parent3[b_neighbors]=b_cost
    # #                     visited3[b_neighbors]=node3
    # if middle_node!=start_node2 and middle_node!=None:
    #     pq_goal_node=PriorityQueue()
    #     visited_goal_node={}
    #     parent_goal_node={middle_node:0}
    #     pq_goal_node.append((0,middle_node))
    #     while pq_goal_node and pq2:
    #         weight2,node2=pq2.pop()
    #         weight_gn,node_gn=pq_goal_node.pop()
            
    #         for f_neighbors in graph[node2]:
    #             f_cost=weight2+graph.get_edge_weight(node2,f_neighbors)
    #             if f_neighbors not in parent2:
    #                 visited2[f_neighbors]=node2
    #                 parent2[f_neighbors]=f_cost
    #                 pq2.append((f_cost,f_neighbors))
    #             else:
    #                 if parent2[f_neighbors]>f_cost:
    #                     parent2[f_neighbors]=f_cost
    #                     visited2[f_neighbors]=node2

    #         for b_neighbors in graph[node_gn]:
    #             b_cost=weight_gn+graph.get_edge_weight(node_gn,b_neighbors)
    #             if b_neighbors not in parent_goal_node:
    #                 visited_goal_node[b_neighbors]=node_gn
    #                 parent_goal_node[b_neighbors]=b_cost
    #                 pq_goal_node.append((b_cost,b_neighbors))

    #             else:
    #                 if parent_goal_node[b_neighbors]>b_cost:
    #                     parent_goal_node[b_neighbors]=b_cost
    #                     visited_goal_node[b_neighbors]=node_gn

    #         if node2 in visited_goal_node or node_gn in visited2:
    #             break

    #     same_node=[]
    #     for items in visited2.keys():
    #         if items in visited_goal_node:
    #             same_node.append(items)
    #     for item in same_node:
    #         pre_total=parent2[item]+parent_goal_node[item]
    #         if min_cost==0 or min_cost>pre_total:
    #             min_cost=pre_total
    #             goal_node=item
    #     f_path=[]
    #     b_path=[]
    #     goal_node_b=goal_node
    #     while goal_node is not None:
    #         f_path.insert(0, goal_node)
    #         goal_node = visited2.get(goal_node)
    #         if goal_node==start_node2:
    #             f_path.insert(0, goal_node)
    #             goal_node=None
    #     while goal_node_b is not None:
    #         b_path.insert(0, goal_node_b)
    #         goal_node_b = visited_goal_node.get(goal_node_b)
    #         if goal_node_b==middle_node:
    #             b_path.insert(0, goal_node_b)
    #             goal_node_b=None
    #     new_path=f_path[:-1] + b_path[::-1]
    # final_path=list(dict.fromkeys(new_path+new_path_13))
    # return final_path
    if 1<len(set(goals))<3:
        return [item for item in set(goals)]
    elif len(set(goals))<=1:
        return []
    # node1,node2,node3=goals
    if goals==['g', 'c', 'v']:
        print(True)
    pq1,pq2,pq3=PriorityQueue(),PriorityQueue(),PriorityQueue()
    start_node1,start_node2,start_node3=goals
    parent1,parent2,parent3={start_node1:0},{start_node2:0},{start_node3:0}
    visited1,visited2,visited3={},{},{}
    pq1.append((0,start_node1))
    pq2.append((0,start_node2))
    pq3.append((0,start_node3))
    node1_in23=False
    node2_in13=False
    node3_in12=False
    min_cost=0
    goal_node=None
    pre_total=0
    middle_node=None
    new_path=[]
    new_path_12=[]
    new_path_pre=[]
    last_nodeIn12=False
    node3_inPath12=False
    while pq1 and pq2:
        weight1,node1=pq1.pop()
        weight2,node2=pq2.pop()

        for f_neighbors in graph[node1]:
            f_cost=weight1+graph.get_edge_weight(node1,f_neighbors)
            if f_neighbors not in visited1:
                visited1[f_neighbors]=node1
                parent1[f_neighbors]=f_cost
                pq1.append((f_cost,f_neighbors))
            else:
                if parent1[f_neighbors]>f_cost:
                    parent1[f_neighbors]=f_cost
                    visited1[f_neighbors]=node1
        
        for b_neighbors in graph[node2]:
            b_cost=weight2+graph.get_edge_weight(node2,b_neighbors)
            if b_neighbors not in visited2:
                visited2[b_neighbors]=node2
                parent2[b_neighbors]=b_cost
                pq2.append((b_cost,b_neighbors))

            else:
                if parent2[b_neighbors]>b_cost:
                    parent2[b_neighbors]=b_cost
                    visited2[b_neighbors]=node2
        if node1 in visited2 or node2 in visited1:
            if start_node3 in visited1 or start_node3 in visited2:
                pre_goal_node=node1
                last_nodeIn12=True
                f_path=[]
                b_path=[]
                pre_goal_node_b=pre_goal_node
                while pre_goal_node is not None:
                    f_path.insert(0, pre_goal_node)
                    pre_goal_node = visited1.get(pre_goal_node)
                    if pre_goal_node==start_node1:
                        f_path.insert(0, pre_goal_node)
                        pre_goal_node=None
                while pre_goal_node_b is not None:
                    b_path.insert(0, pre_goal_node_b)
                    pre_goal_node_b = visited2.get(pre_goal_node_b)
                    if pre_goal_node_b==start_node2:
                        b_path.insert(0, pre_goal_node_b)
                        pre_goal_node_b=None
                new_path_pre=f_path[:-1] + b_path[::-1]
                if start_node3 in new_path_pre:
                    node3_inPath12=True
            # else:
            same_node=[]
            for items in visited1.keys():
                if items in visited2:
                    same_node.append(items)
            for item in same_node:
                pre_total=parent1[item]+parent2[item]
                if min_cost==0 or min_cost>pre_total:
                    min_cost=pre_total
                    goal_node=item
            # goal_node=node1
            f_path=[]
            b_path=[]
            goal_node_b=goal_node
            while goal_node is not None:
                f_path.insert(0, goal_node)
                goal_node = visited1.get(goal_node)
                if goal_node==start_node1:
                    f_path.insert(0, goal_node)
                    goal_node=None
            while goal_node_b is not None:
                b_path.insert(0, goal_node_b)
                goal_node_b = visited2.get(goal_node_b)
                if goal_node_b==start_node2:
                    b_path.insert(0, goal_node_b)
                    goal_node_b=None
            new_path_12=f_path[:-1] + b_path[::-1]
            break
    # if goal_node!=start_node2 and goal_node!=None:
    #     new_pq1=PriorityQueue()
    #     new_pq1.append((0,start_node1))
    #     new_parent1={start_node1:0}
    #     new_visited1={}
    #     while new_pq1 and pq2:
    #         weight1,node1=new_pq1.pop()
    #         weight2,node2=pq2.pop()

    #         for f_neighbors in graph[node1]:
    #             f_cost=weight1+graph.get_edge_weight(node1,f_neighbors)
    #             if f_neighbors not in visited1:
    #                 new_visited1[f_neighbors]=node1
    #                 new_parent1[f_neighbors]=f_cost
    #                 new_pq1.append((f_cost,f_neighbors))
    #             else:
    #                 if new_parent1[f_neighbors]>f_cost:
    #                     new_parent1[f_neighbors]=f_cost
    #                     new_visited1[f_neighbors]=node1
            
    #         for b_neighbors in graph[node3]:
    #             b_cost=weight3+graph.get_edge_weight(node3,b_neighbors)
    #             if b_neighbors not in visited3:
    #                 visited3[b_neighbors]=node3
    #                 parent3[b_neighbors]=b_cost
    #                 pq3.append((b_cost,b_neighbors))

    #             else:
    #                 if parent3[b_neighbors]>b_cost:
    #                     parent3[b_neighbors]=b_cost
    #                     visited3[b_neighbors]=node3
    # if middle_node!=start_node2 and middle_node!=None:
    if len(new_path_pre)>0 and node3_inPath12==False:
        final_result=new_path_12[:-1]+new_path_12[::-1]+new_path_pre[1:]
        return final_result
    else:
        new_pq2=PriorityQueue()
        new_visited2={}
        new_parent2={start_node2:0}
        new_pq2.append((0,start_node2))
        min_cost_next=0
        pre_total_next=0
        while pq3 and new_pq2:
            weight3,node3=pq3.pop()
            weight_gn,node_gn=new_pq2.pop()
            
            for f_neighbors in graph[node3]:
                f_cost=weight3+graph.get_edge_weight(node3,f_neighbors)
                if f_neighbors not in parent3:
                    visited3[f_neighbors]=node3
                    parent3[f_neighbors]=f_cost
                    pq3.append((f_cost,f_neighbors))
                else:
                    if parent3[f_neighbors]>f_cost:
                        parent3[f_neighbors]=f_cost
                        visited3[f_neighbors]=node3

            for b_neighbors in graph[node_gn]:
                b_cost=weight_gn+graph.get_edge_weight(node_gn,b_neighbors)
                if b_neighbors not in new_parent2:
                    new_visited2[b_neighbors]=node_gn
                    new_parent2[b_neighbors]=b_cost
                    new_pq2.append((b_cost,b_neighbors))

                else:
                    if new_parent2[b_neighbors]>b_cost:
                        new_parent2[b_neighbors]=b_cost
                        new_visited2[b_neighbors]=node_gn

            if node3 in new_visited2 or node_gn in visited3:
                break

        same_node=[]
        for items in new_visited2.keys():
            if items in visited3:
                same_node.append(items)
        for item in same_node:
            pre_total_next=new_parent2[item]+parent3[item]
            if min_cost_next==0 or min_cost_next>=pre_total_next:
                min_cost_next=pre_total_next
                goal_node=item
        f_path=[]
        b_path=[]
        goal_node_b=goal_node
        while goal_node is not None:
            f_path.insert(0, goal_node)
            goal_node = visited3.get(goal_node)
            if goal_node==start_node2:
                f_path.insert(0, goal_node)
                goal_node=None
        while goal_node_b is not None:
            b_path.insert(0, goal_node_b)
            goal_node_b = new_visited2.get(goal_node_b)
            if goal_node_b==start_node3:
                b_path.insert(0, goal_node_b)
                goal_node_b=None
        new_path=f_path[:-1] + b_path[::-1]
        final_path=new_path_12[:-1]+new_path[::-1]
        return final_path
    # TODO: finish this function
    # raise NotImplementedError

def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic, landmarks=None):
    """
    Exercise 4: Upgraded Tridirectional Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.
        landmarks: Iterable containing landmarks pre-computed in compute_landmarks()
            Default: None

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    raise NotImplementedError


def return_your_name():
    """Return your name from this function"""
    # # TODO: finish this function
    # raise NotImplementedError
    return 'Ruixiang Huang'


def compute_landmarks(graph):
    """
    Feel free to implement this method for computing landmarks. We will call
    tridirectional_upgraded() with the object returned from this function.

    Args:
        graph (ExplorableGraph): Undirected graph to search.

    Returns:
    List with not more than 4 computed landmarks. 
    """
    return None


def custom_heuristic(graph, v, goal):
    """
       Feel free to use this method to try and work with different heuristics and come up with a better search algorithm.
       Args:
           graph (ExplorableGraph): Undirected graph to search.
           v (str): Key for the node to calculate from.
           goal (str): Key for the end node to calculate to.
       Returns:
           Custom heuristic distance between `v` node and `goal` node
       """
    pass


# Extra Credit: Your best search method for the race
def custom_search(graph, start, goal, data=None):
    """
    Race!: Implement your best search algorithm here to compete against the
    other student agents.

    If you implement this function and submit your code to Gradescope, you'll be
    registered for the Race!

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Will be passed your data from load_data(graph).
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


def load_data(graph, time_left):
    """
    Feel free to implement this method. We'll call it only once 
    at the beginning of the Race, and we'll pass the output to your custom_search function.
    graph: a networkx graph
    time_left: function you can call to keep track of your remaining time.
        usage: time_left() returns the time left in milliseconds.
        the max time will be 10 minutes.

    * To get a list of nodes, use graph.nodes()
    * To get node neighbors, use graph.neighbors(node)
    * To get edge weight, use graph.get_edge_weight(node1, node2)
    """

    # nodes = graph.nodes()
    return None
 
 
def haversine_dist_heuristic(graph, v, goal):
    """
    Note: This provided heuristic is for the Atlanta race.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Haversine distance between `v` node and `goal` node
    """

    #Load latitude and longitude coordinates in radians:
    vLatLong = (math.radians(graph.nodes[v]["pos"][0]), math.radians(graph.nodes[v]["pos"][1]))
    goalLatLong = (math.radians(graph.nodes[goal]["pos"][0]), math.radians(graph.nodes[goal]["pos"][1]))

    #Now we want to execute portions of the formula:
    constOutFront = 2*6371 #Radius of Earth is 6,371 kilometers
    term1InSqrt = (math.sin((goalLatLong[0]-vLatLong[0])/2))**2 #First term inside sqrt
    term2InSqrt = math.cos(vLatLong[0])*math.cos(goalLatLong[0])*((math.sin((goalLatLong[1]-vLatLong[1])/2))**2) #Second term
    return constOutFront*math.asin(math.sqrt(term1InSqrt+term2InSqrt)) #Straight application of formula
