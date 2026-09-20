# 单源
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np  #当前为numpy=2.0.2,numexpr=2.10.2, pandas=2.3.2
import copy
import pandas as pd
from guro_opt import *
from lpsi import lpsi_deg_source_identification, lpsi_source_identification
import time
# import networkx.utils

random.seed(0)
np.random.seed(0)
# nx.utils.misc.np.random.seed(0)
# networkx.utils.misc.numpy.random.seed(0)

class SL:
    def __init__(self, num_tasks, max_dependencies, read=False, L_num=[1,2]):
        self.num_tasks = num_tasks
        self.max_dependencies = max_dependencies
        self.graph = nx.DiGraph()
        self.mini_graph = nx.DiGraph()
        self.L_num = L_num
        self.mapping = {}          # 父节点 -> 子节点列表
        self.reverse_mapping = {}  # 子节点 -> 父节点
        self.create_task_network(read)

    def create_task_network(self, read=False):
        if read:
            # 读取下级网络节点和边数据
            node_df = pd.read_excel('data/nodelist_L'+str(self.L_num[1])+'.xlsx', dtype={'id': str})  # id列为 'id'
            edge_df = pd.read_excel('data/edgelist_L'+str(self.L_num[1])+'.xlsx', dtype={'source': str, 'target': str})  #  'source' 和 'target'列为str
            for index, row in node_df.iterrows():
                node_id = row['id'].strip()
                # node_label = row['label']
                self.graph.add_node(node_id)
                # self.graph.nodes[node_id]['label'] = node_label
                # self.graph.nodes[node_id]['time'] = row['time']
                # self.graph.nodes[node_id]['cost'] = row['cost']
                self.graph.nodes[node_id]['status'] = 1  # 初始化状态为1，表示未执行
            for index, row in edge_df.iterrows():
                source = row['source'].strip()
                target = row['target'].strip()
                # relation_attr = row['relation']
                self.graph.add_edge(source, target)
                # self.graph[source][target]['relation'] = relation_attr
            print('节点数：', len(self.graph.nodes))
            print('边数：', len(self.graph.edges))

            # 读取上级网络节点和边数据
            node_df = pd.read_excel('data/nodelist_L'+str(self.L_num[0])+'.xlsx', dtype={'id': str})  # id列为 'id'
            edge_df = pd.read_excel('data/edgelist_L'+str(self.L_num[0])+'.xlsx', dtype={'source': str, 'target': str})  #  'source' 和 'target'列为str
            for index, row in node_df.iterrows():
                node_id = row['id'].strip()
                # node_label = row['label']
                self.mini_graph.add_node(node_id)
                # self.mini_graph.nodes[node_id]['label'] = node_label
                # self.mini_graph.nodes[node_id]['time'] = row['time']
                # self.mini_graph.nodes[node_id]['cost'] = row['cost']
                self.mini_graph.nodes[node_id]['status'] = 1  # 初始化状态为1，表示未执行
                self.mini_graph.nodes[node_id]['not_done_son'] = 0
            for index, row in edge_df.iterrows():
                source = row['source'].strip()
                target = row['target'].strip()
                # relation_attr = row['relation']
                self.mini_graph.add_edge(source, target)
                # self.mini_graph[source][target]['relation'] = relation_attr
            print('节点数：', len(self.mini_graph.nodes))
            print('边数：', len(self.mini_graph.edges))

            # # 建立上层网络mini_graph和下层网络graph之间的mapping关系
            # self.mapping = {}
            # self.graph = nx.relabel_nodes(self.graph, mapping=lambda x: x.strip())
            # self.mini_graph = nx.relabel_nodes(self.mini_graph, mapping=lambda x: x.strip())
            # for node in self.graph.nodes:
            #     if self.L_num[0] == 1 and self.L_num[1] == 2:
            #         parent_node = node[:2] + '0000'
            #     elif self.L_num[0] == 2 and self.L_num[1] == 3:
            #         parent_node = node[:4] + '00'
            #     if parent_node in self.mini_graph.nodes:
            #         self.mapping[node] = parent_node
            #     else:
            #         print('Error: Parent node', parent_node, 'not found for node', node)

            # 建立上层网络mini_graph和下层网络graph之间的双向mapping关系
            self.graph = nx.relabel_nodes(self.graph, mapping=lambda x: x.strip())
            self.mini_graph = nx.relabel_nodes(self.mini_graph, mapping=lambda x: x.strip())
            
            for node in self.graph.nodes:
                if self.L_num[0] == 1 and self.L_num[1] == 2:
                    parent_node = node[:2] + '0000'
                elif self.L_num[0] == 2 and self.L_num[1] == 3:
                    parent_node = node[:4] + '00'
                    
                if parent_node in self.mini_graph.nodes:
                    # 建立反向映射：子节点 -> 父节点
                    self.reverse_mapping[node] = [parent_node]

                    # 建立正向映射：父节点 -> 子节点列表
                    if parent_node not in self.mapping:
                        self.mapping[parent_node] = []
                    self.mapping[parent_node].append(node)
                else:
                    print('Error: Parent node', parent_node, 'not found for node', node)


    def reset_status(self):
        # 重置所有节点的状态为1
        for node in self.graph.nodes:
            self.graph.nodes[node]['status'] = 1
        for node in self.mini_graph.nodes:
            self.mini_graph.nodes[node]['status'] = 1
            self.mini_graph.nodes[node]['not_done_son'] = 0

    def reachable_sets(self, adj, nodes):
        """返回每个节点的可达集 R(v)。adj: {u: [v1, v2, ...]}"""
        def dfs(start):
            seen, st = set(), [start]
            while st:
                x = st.pop()
                if x in seen: 
                    continue
                seen.add(x)
                for y in adj.get(x, []):
                    if y not in seen:
                        st.append(y)
            return seen
        R = {v: dfs(v) for v in nodes}
        return R

    def begin_task(self, error_nodes, noise='normal', noise_param=None):
        # noise_list = ['obs_error', 'prob_prop'] # 添加噪声的类型，[观测误差，概率传播]
        # noise_params = {'obs_error': {'error_prop':0.1}, 
        #                 'prob_prop': {'prop_prob':0.5}} # 噪声参数

        # 找到没有前序节点的任务节点，开始执行
        start_nodes = [node for node in self.graph.nodes if self.graph.in_degree(node) == 0]
        # print('start_nodes:', start_nodes)
        for node in start_nodes:
            if node in error_nodes:
                self.graph.nodes[node]['status'] = 1
            else:
                self.graph.nodes[node]['status'] = 0

        # 遍历status属性为1的节点，将其中所有前序节点均为0的节点的status属性更新为0
        change = True
        while change == True:
            change = False
            # 筛选出status属性为1并且不在error_nodes中的节点
            candi_nodes = [node for node in self.graph.nodes if self.graph.nodes[node]['status'] == 1 and node not in error_nodes]
            for node in candi_nodes:
                predecessors = list(self.graph.predecessors(node))
                if all(self.graph.nodes[pred]['status'] == 0 for pred in predecessors):
                    self.graph.nodes[node]['status'] = 0
                    # print('节点', node, '被设置为0')
                    change = True

        # 记录没有执行、已经执行的任务节点
        not_done = [node for node in self.graph.nodes if self.graph.nodes[node]['status'] == 1]
        # print('没有执行任务节点：', not_done)
        done = [node for node in self.graph.nodes if self.graph.nodes[node]['status'] == 0]

        if 'prob_prop' in noise: # 针对所有没有执行的节点（树/森林结构），模拟概率传播
            # 从error_nodes开始，沿着有向边，以prop_prob的概率将status属性从1传播到其后继节点
            # 根据not_done生成子图，将子图和原图中的故障节点都改为status=0
            subgraph = self.graph.subgraph(not_done).copy()
            for node in not_done:
                if node not in error_nodes: # error_nodes是100%会发生故障的节点，除此以外均调整为正常节点
                    subgraph.nodes[node]['status'] = 0
                    self.graph.nodes[node]['status'] = 0
            # 在子图中开始传播
            new_error_frontier = error_nodes.copy()
            while new_error_frontier:
                error_frontier = new_error_frontier.copy()
                new_error_frontier = []
                for node in error_frontier:
                    # 子节点以prop_prob的概率传播故障
                    for sub_node in list(subgraph.successors(node)):
                        if random.random() < noise_param['prop_prob']: # 概率传播，小于故障传播概率就将该节点改为未完成，即status=1
                            subgraph.nodes[sub_node]['status'] = 1
                            self.graph.nodes[sub_node]['status'] = 1
                            new_error_frontier.append(sub_node)

            not_done = [node for node in self.graph.nodes if self.graph.nodes[node]['status'] == 1]
            done = [node for node in self.graph.nodes if self.graph.nodes[node]['status'] == 0]


        if 'obs_error' in noise: # 从所有节点中随机选取一部分节点，更换其status属性，模拟观测误差
            # 从not_done和done中随机选取一部分节点，更换其status属性，并且确保数量为num_errors
            num_errors = int(len(not_done + done) * noise_param['error_prop'])
            candidates = sorted(not_done + done)
            error_nodes = random.sample(candidates, num_errors)
            for node in error_nodes:
                self.graph.nodes[node]['status'] = 1 - self.graph.nodes[node]['status']

            not_done = [node for node in self.graph.nodes if self.graph.nodes[node]['status'] == 1]
            # print('没有执行任务节点(有观测误差)：', not_done)
            done = [node for node in self.graph.nodes if self.graph.nodes[node]['status'] == 0]
            
        # 根据下级网络中的任务执行情况，获取上级网络中的任务执行情况
        for node in self.mini_graph.nodes:
            self.mini_graph.nodes[node]['status'] = 0 # 先全部设定为0
        parent_nodes = [parent_node for not_done_node in not_done for parent_node in self.reverse_mapping[not_done_node]]
        for parent_node in parent_nodes:
            self.mini_graph.nodes[parent_node]['status'] = 1
            self.mini_graph.nodes[parent_node]['not_done_son'] += 1

        mini_not_done = [node for node in self.mini_graph.nodes if self.mini_graph.nodes[node]['status'] == 1]

        return not_done, mini_not_done

    def snapshot(self, show_graph, show=False):
        pos = nx.spring_layout(show_graph)
        statuses = [show_graph.nodes[node]['status'] for node in show_graph.nodes]
        colors = ['green' if status == 0 else 'red' for status in statuses]

        # # 在节点附近标明status属性值
        # for node, color in zip(self.graph.nodes, colors):
        #     plt.annotate(f'{self.graph.nodes[node]["status"]}', (pos[node][0], pos[node][1]), fontsize=12, color=color)
        # nx.draw_networkx_labels(self.graph, pos, {node: f'{status}' for node, status in zip(self.graph.nodes, statuses)})

        nx.draw(show_graph, pos, with_labels=True, node_color=colors, node_size=800, font_size=16)
        if show:
            plt.show()

    def source_locate_single_layer(self, sin_lay_graph, method, method_params): #这完全是一个外置函数，不一定要包含在SL类中
        if method == 'mini_set':
            not_done = [node for node in sin_lay_graph.nodes if sin_lay_graph.nodes[node]['status'] == 1]
            # 修改1: 使用排序后的节点列表确保顺序一致性
            sorted_not_done = sorted(not_done)
            graph_propagate = sin_lay_graph.subgraph(sorted_not_done).copy()
            
            # 修改2: 使用排序后的节点列表
            A_propagate = nx.to_numpy_array(graph_propagate, nodelist=sorted_not_done)
            
            # 修改3: 使用相同的排序列表进行索引映射
            zero_in_degree_indexes = [i for i in range(A_propagate.shape[0]) if np.all(A_propagate[:, i] == 0)]
            zero_out_degree_indexes = [i for i in range(A_propagate.shape[0]) if np.all(A_propagate[i, :] == 0)]
            zero_in_degree_nodes = [sorted_not_done[i] for i in zero_in_degree_indexes if i not in zero_out_degree_indexes]
            # zero_in_degree_nodes = [sorted_not_done[i] for i in zero_in_degree_indexes]

            return zero_in_degree_nodes

        if method == 'ILP':
            # 一个小型 DAG 示例
            # adj = {
            #     "a": ["b", "c"],
            #     "b": ["d"],
            #     "c": ["d", "e"],
            #     "d": ["f"],
            #     "e": ["f"],
            #     "f": []
            # }
            # # 将sin_lay_graph转换为adj字典形式,key为节点，value为该节点的出度节点列表
            # adj = {node: [neighbor for neighbor in sin_lay_graph.neighbors(node)] for node in sin_lay_graph.nodes} 

            # 修改后
            sorted_nodes = sorted(sin_lay_graph.nodes())
            adj = {node: sorted([neighbor for neighbor in sin_lay_graph.neighbors(node)]) for node in sorted_nodes}

            # sin_lay_graph.neighbors(node) 只会返回node的出度节点（successors），而不会返回入度节点（predecessors）
            # 获取故障节点（status=1）和健康节点（status=0）
            failed = sorted([node for node in sin_lay_graph.nodes if sin_lay_graph.nodes[node]['status'] == 1])
            healthy = sorted([node for node in sin_lay_graph.nodes if sin_lay_graph.nodes[node]['status'] == 0])
            not_done_son_dict = None
            if sin_lay_graph.nodes and 'not_done_son' in sin_lay_graph.nodes[next(iter(sin_lay_graph.nodes()))]:
                not_done_son_dict = nx.get_node_attributes(sin_lay_graph, 'not_done_son')
            if method_params['find_in_failed']:
                candidates = failed
            else:
                candidates = None
            # print('failed:', failed)
            # try:
            #     S_opt, model = ilp_min_source_cover_hard(adj, failed, healthy, cardinality_constraint=1, log_to_console=False)
            #     # print("Optimal sources (hard):", S_opt)
            #     return list(S_opt)
            # except:
            if noise == 'prob_prop' or noise == 'normal':
            # if True:
                S_soft, stats, model = ilp_min_source_cover_soft_prob(
                    adj, failed, healthy,candidates=candidates,
                    cardinality_constraint=method_params['cardinality_constraint'], 
                    alpha_uncovered=method_params['alpha_uncovered'],   # 未覆盖故障的惩罚（越大越强制覆盖）
                    beta_touchH=method_params['beta_touchH'],       # 触达健康的惩罚（越大越避免误伤）
                    log_to_console=False
                )
                return list(S_soft)
            else:
                S_soft, stats, model = ilp_min_source_cover_soft_dynamic(
                    adj, failed, healthy,candidates=candidates,not_done_son=not_done_son_dict,
                    cardinality_constraint=method_params['cardinality_constraint'],
                    alpha_uncovered=method_params['alpha_uncovered'],   # 未覆盖故障的惩罚（越大越强制覆盖）
                    beta_touchH=method_params['beta_touchH'],       # 触达健康的惩罚（越大越避免误伤）
                    log_to_console=False
                )
                return list(S_soft)
        
        if method == 'LPSI':
            sources, _, _ = lpsi_source_identification(
                sin_lay_graph, source_count=None, **method_params
            )
            return sources

        if method == 'LPSI_deg':
            sources, _, _ = lpsi_deg_source_identification(
                sin_lay_graph, source_count=None, **method_params
            )
            return sources

        if method == 'label_prop':
            not_done = [node for node in sin_lay_graph.nodes if sin_lay_graph.nodes[node]['status'] == 1]
            # 同样使用排序确保一致性
            sorted_not_done = sorted(not_done)
            graph_propagate = sin_lay_graph.subgraph(sorted_not_done).copy()
            
            # 边处理也需要保持顺序一致性
            edges_to_reverse = []
            for edge in list(graph_propagate.edges()):
                if graph_propagate.nodes[edge[0]]['status'] == 1 and graph_propagate.nodes[edge[1]]['status'] == 1:
                    edges_to_reverse.append(edge)
            
            for edge in edges_to_reverse:
                graph_propagate.add_edge(edge[1], edge[0])
                graph_propagate.remove_edge(edge[0], edge[1])
            
            # 使用排序后的节点列表
            A_propagate = nx.to_numpy_array(graph_propagate, nodelist=sorted_not_done)
            A_propagate = A_propagate + np.eye(A_propagate.shape[0])
            row_sums = A_propagate.sum(axis=1, keepdims=True)
            A_propagate = A_propagate / row_sums

            Y_propagate = np.abs(np.array([graph_propagate.nodes[node]['status'] for node in sorted_not_done]))
            Y_propagate = Y_propagate.reshape(1, -1)
            
            for _ in range(method_params['times']):
                # print(np.dot(Y_propagate, A_propagate))
                # print(np.sum(Y_propagate)/len(Y_propagate[0]))
                # k = input()
                Y_propagate = method_params['weight'][0]*np.dot(Y_propagate, A_propagate) + method_params['weight'][1]*np.sum(Y_propagate)/len(Y_propagate[0])
            
            Y = Y_propagate
            # 获取Y中最大的正值及其对应的节点
            # print('Y:', Y)
            # print('sum', sum(Y[0]), len(Y[0]), sum(Y[0])/len(Y[0]))
            # 获取Y值大于1的序号
            if noise == 'obs_error':
                max_status_change_index = [i for i in range(len(Y[0])) if Y[0, i] > 1]
            else:
                max_status_change_index = [i for i in range(len(Y[0])) if Y[0, i] >= 1]
            max_status_change_node = [sorted_not_done[i] for i in max_status_change_index]
            # error_nodes_estimated = [list(graph_propagate.nodes)[i] for i in range(len(Y[0])) if Y[0, i] > 1]

            return max_status_change_node
            
    def source_locate(self, method, method_params):
        # mini_graph_split = split_graph(self.mini_graph) # 主要用来进一步细分mini图中的节点，目的是缩小逐层溯源和单层溯源的性能差，是否是分出分叉节点更好？？
        # lower_constraint = method_params['cardinality_constraint'][0] # 留存备份
        # method_params['cardinality_constraint'][0] = 1 # 上层网络的故障源节点下限为1
        mini_source_list = self.source_locate_single_layer(self.mini_graph, method, method_params)
        subgraph = self.graph.subgraph(list(set([node for mini_source in mini_source_list for node in self.mapping[mini_source]])))
        print(subgraph.nodes)
        # method_params['cardinality_constraint'][0] = lower_constraint
        source_list = self.source_locate_single_layer(subgraph, method, method_params)

        return source_list        
        
# 生成一个有向无环图
num_tasks = 10
max_dependencies = 10
L_num = [2,3]
sl = SL(num_tasks, max_dependencies, read=True, L_num=L_num)

# 开展溯源实验
# 初始化参数
method_list = ['LPSI'] # 选择溯源算法['mini_set', 'label_prop', 'LPSI', 'LPSI_deg', 'ILP']
method_params = {'mini_set': {},
                 'ILP': {'find_in_failed':1,
                     'cardinality_constraint':[1,5], #在多层多源情况下，有时候上层溯源只会产生一个源，这时候如果在下层溯源设定[2,5]的范围，就会报错，应该改为[1,5]
                    'alpha_uncovered':10.0,   # 未覆盖故障的惩罚（越大越强制覆盖）
                    'beta_touchH':10.0},    # 触达健康的惩罚（越大越避免误伤）},
                 'label_prop': {'times':100, 'weight':[1.0, 0.0, 0.0]},
                 'LPSI': {'alpha':0.5, 'max_iter':5, 'tol':1e-8,
                          'mode':'iterative', 'graph_mode':'undirected'},
                 'LPSI_deg': {'alpha':0.5, 'max_iter':5, 'tol':1e-8,
                              'mode':'iterative', 'graph_mode':'undirected',
                              'degree_power':1.0, 'degree_normalization':'none'}}
noise_list = ['obs_error'] # 添加噪声的类型，[无误差，观测误差，概率传播]=['normal', 'prob_prop', 'obs_error']
noise_params = {'normal':{},
        'obs_error': {'error_prop':0.05},
        'prob_prop': {'prop_prob':0.8}} # 噪声参数
tic = time.time()
for method in method_list:
    print('---------------------method:', method, '-------------------')
    for noise in noise_list:
        print('=================noise:', noise, '=================')
        F1_score_all = []
        for i in range(100):
        # for error_nodes in sl.graph.nodes():
            # 每次随机从网络节点中选择2-5个节点
            random_num = random.randint(2,5)
            error_nodes = random.sample(list(sl.graph.nodes()), random_num)
            # error_nodes = [error_nodes]
            # error_nodes = ['030100']
            # print('error_nodes', error_nodes)
            # error_nodes = ['030100', '080100', '140200']
            print('*********************error_nodes:', error_nodes, '********************')
            
            # 将所有节点的status标签重置为1
            sl.reset_status()
            not_done = sl.begin_task(error_nodes, noise=noise, noise_param=noise_params[noise])
            # print("未完成的任务节点:", not_done)

            # sl.snapshot(show=True)

            error_nodes_estimated = sl.source_locate(method, method_params[method])
            print(error_nodes_estimated)
            # 使用error_nodes_estimated和error_nodes计算F1_score
            N_right = len(set(error_nodes_estimated) & set(error_nodes))
            N_precision = len(error_nodes_estimated)
            N_recall = len(error_nodes)
            F1_score = 2 * N_right / (N_precision + N_recall)
            
            # error_nodes_estimated == error_nodes
            F1_score_all.append(F1_score)
        print('平均F1-score为：', np.average(F1_score_all))
toc = time.time()
print('总耗时：', toc - tic)
