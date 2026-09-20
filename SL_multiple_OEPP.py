# 生成一个单层任务网络（有向无环图），当任务执行卡住后，生成网络快照，将已经执行完毕的任务节点标记为0，没有执行完毕的任务节点标记为1
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np  #当前为numpy=2.0.2,numexpr=2.10.2, pandas=2.3.2
import copy
import pandas as pd
from guro_opt import *
from lpsi import lpsi_deg_source_identification, lpsi_source_identification
import time
import json

random.seed(0)

class SL:
    def __init__(self, num_tasks, max_dependencies, read=False, L_num=2):
        self.num_tasks = num_tasks
        self.max_dependencies = max_dependencies
        self.graph = nx.DiGraph()
        self.L_num = L_num
        self.create_task_network(read)

    def create_task_network(self, read=False):
        if read:
            # 读取节点和边数据
            node_df = pd.read_excel('data/nodelist_L'+str(self.L_num)+'.xlsx', dtype={'id': str})  # id列为 'id'
            edge_df = pd.read_excel('data/edgelist_L'+str(self.L_num)+'.xlsx', dtype={'source': str, 'target': str})  #  'source' 和 'target'列为str

            # 添加节点
            for index, row in node_df.iterrows():
                node_id = row['id']
                # if '0000' not in node_id: #筛选出二级节点
                # node_label = row['label']
                self.graph.add_node(node_id)
                # self.graph.nodes[node_id]['label'] = node_label
                # self.graph.nodes[node_id]['time'] = row['time']
                # self.graph.nodes[node_id]['cost'] = row['cost']
                self.graph.nodes[node_id]['status'] = 1  # 初始化状态为1，表示未执行

            # 添加边
            for index, row in edge_df.iterrows():
                source = row['source']
                target = row['target']
                # if '0000' not in source and '0000' not in target:
                # relation_attr = row['relation']
                self.graph.add_edge(source, target)
                # self.graph[source][target]['relation'] = relation_attr
            print('节点数：', len(self.graph.nodes))
            print('边数：', len(self.graph.edges))
        else:
            # 添加任务节点
            for i in range(self.num_tasks):
                self.graph.add_node(i, status=0)  # 初始状态为0，表示未执行

            # 添加依赖关系（有向边）
            for i in range(self.num_tasks):
                num_deps = random.randint(0, min(self.max_dependencies, i))
                deps = random.sample(range(i), num_deps)
                for dep in deps:
                    self.graph.add_edge(dep, i)
            print('节点数：', len(self.graph.nodes))
            print('边数：', len(self.graph.edges))

    def reset_status(self):
        for node in self.graph.nodes:
            self.graph.nodes[node]['status'] = 1

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


        if 'obs_error' in noise: # 从所有节点中随机选取一部分节点，更换status属性为0，模拟观测误差
            # 从not_done和done中随机选取一部分节点，更换其status属性，并且确保数量为num_errors
            num_errors = int(len(not_done + done) * noise_param['error_prop'])
            candidates = sorted(not_done + done)
            obs_error_nodes = random.sample(candidates, num_errors)
            for node in obs_error_nodes:
                self.graph.nodes[node]['status'] = 1 - self.graph.nodes[node]['status']

            # num_errors = int(len(not_done + done) * noise_param['error_prop'])
            # obs_error_nodes = []
            # obs_error_candidates = sorted(not_done + done)
            # while len(obs_error_nodes) < num_errors:
            #     obs_error_node = random.choice(obs_error_candidates)
            #     if obs_error_node not in error_nodes:
            #         obs_error_nodes.append(obs_error_node)

            not_done = [node for node in self.graph.nodes if self.graph.nodes[node]['status'] == 1]
            done = [node for node in self.graph.nodes if self.graph.nodes[node]['status'] == 0]

        return not_done

    def snapshot(self, show=False):
        pos = nx.spring_layout(self.graph)
        statuses = [self.graph.nodes[node]['status'] for node in self.graph.nodes]
        colors = ['green' if status == 0 else 'red' for status in statuses]

        # # 在节点附近标明status属性值
        # for node, color in zip(self.graph.nodes, colors):
        #     plt.annotate(f'{self.graph.nodes[node]["status"]}', (pos[node][0], pos[node][1]), fontsize=12, color=color)
        # nx.draw_networkx_labels(self.graph, pos, {node: f'{status}' for node, status in zip(self.graph.nodes, statuses)})

        nx.draw(self.graph, pos, with_labels=True, node_color=colors, node_size=800, font_size=16)
        if show:
            plt.show()

    def write_status(self):
        # 将节点状态状态标签写入文件
        status_list = [{'id': node, 'status': self.graph.nodes[node]['status']} for node in self.graph.nodes()]
        status_df = pd.DataFrame(status_list)
        status_df.to_excel('data/status_L'+str(self.L_num)+'.xlsx', index=False)

    def source_locate(self, method, method_params, error_nodes):
        if method == 'mini_set':
            not_done = [node for node in self.graph.nodes if self.graph.nodes[node]['status'] == 1]
            # 修改1: 使用排序后的节点列表确保顺序一致性
            sorted_not_done = sorted(not_done)
            graph_propagate = self.graph.subgraph(sorted_not_done).copy()
            
            # 修改2: 使用排序后的节点列表
            A_propagate = nx.to_numpy_array(graph_propagate, nodelist=sorted_not_done)
            
            # 修改3: 使用相同的排序列表进行索引映射
            zero_in_degree_indexes = [i for i in range(A_propagate.shape[0]) if np.all(A_propagate[:, i] == 0)]
            zero_in_degree_nodes = [sorted_not_done[i] for i in zero_in_degree_indexes]
            zero_out_degree_indexes = [i for i in range(A_propagate.shape[0]) if np.all(A_propagate[i, :] == 0)]
            zero_in_degree_nodes = [sorted_not_done[i] for i in zero_in_degree_indexes if i not in zero_out_degree_indexes]
            # zero_in_degree_nodes = [sorted_not_done[i] for i in zero_in_degree_indexes]
            # print('zero_in_degree_nodes:', zero_in_degree_nodes)

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
            # 将self.graph转换为adj字典形式,key为节点，value为该节点的出度节点列表
            sorted_nodes = sorted(self.graph.nodes())
            adj = {node: sorted([neighbor for neighbor in self.graph.neighbors(node)]) for node in sorted_nodes}
            # self.graph.neighbors(node) 只会返回node的出度节点（successors），而不会返回入度节点（predecessors）
            # 获取故障节点（status=1）和健康节点（status=0）
            failed = sorted([node for node in self.graph.nodes if self.graph.nodes[node]['status'] == 1])
            healthy = sorted([node for node in self.graph.nodes if self.graph.nodes[node]['status'] == 0])
            if method_params['find_in_failed']:
                candidates = failed
            else:
                candidates = None
            # print('failed:', failed)
            # try:
            #     S_opt, model = ilp_min_source_cover_hard(adj, failed, healthy, log_to_console=False)
            #     return list(S_opt)
            #     # print("Optimal sources (hard):", S_opt)
            # except:
            if noise == 'prob_prop' or noise == 'normal':
                S_soft, stats, model = ilp_min_source_cover_soft_prob(
                    adj, failed, healthy,cardinality_constraint=method_params['cardinality_constraint'], candidates=candidates,
                    alpha_uncovered=method_params['alpha_uncovered'],   # 未覆盖故障的惩罚（越大越强制覆盖）
                    beta_touchH=method_params['beta_touchH'],       # 触达健康的惩罚（越大越避免误伤）
                    log_to_console=False
                )
                return list(S_soft)
            else:
                S_soft, stats, model = ilp_min_source_cover_soft(
                    adj, failed, healthy,cardinality_constraint=method_params['cardinality_constraint'],candidates=candidates,
                    alpha_uncovered=method_params['alpha_uncovered'],   # 未覆盖故障的惩罚（越大越强制覆盖）
                    beta_touchH=method_params['beta_touchH'],       # 触达健康的惩罚（越大越避免误伤）
                    log_to_console=False
                )
                return list(S_soft)
        

        if method == 'LPSI':
            sources, _, _ = lpsi_source_identification(
                self.graph, source_count=None, **method_params
            )
            return sources

        if method == 'LPSI_deg':
            sources, _, _ = lpsi_deg_source_identification(
                self.graph, source_count=None, **method_params
            )
            return sources

        if method == 'label_prop':
            not_done = [node for node in self.graph.nodes if self.graph.nodes[node]['status'] == 1]
            sorted_not_done = sorted(not_done)
            graph_propagate = self.graph.subgraph(sorted_not_done).copy()
            # print('graph_propagate:', graph_propagate)
            # print('graph_propagate edges:', graph_propagate.edges)
            edges_to_reverse = []
            for edge in list(graph_propagate.edges()):
                if graph_propagate.nodes[edge[0]]['status'] == 1 and graph_propagate.nodes[edge[1]]['status'] == 1:
                    edges_to_reverse.append(edge)
            
            for edge in edges_to_reverse:
                graph_propagate.add_edge(edge[1], edge[0])
                graph_propagate.remove_edge(edge[0], edge[1])

            # # 计算网络直径，确定至少需要迭代多少次
            # max_path_length = nx.dag_longest_path_length(self.graph)
            # print('graph_propagate max_path_length:', max_path_length)

            # 获取传播矩阵，即邻接矩阵
            A_propagate = nx.to_numpy_array(graph_propagate, nodelist=sorted_not_done)  # shape (n, n)
            # print('A_propagate:\n', A_propagate)
            # if error_nodes==['103508', '070303', '102009']:
            #     # 构造有向图
            #     G = nx.from_numpy_array(A_propagate, create_using=nx.DiGraph)

            #     # 绘图
            #     plt.figure(figsize=(8,6))
            #     pos = nx.spring_layout(G, seed=42)  # 选择一种布局
            #     nx.draw(G, pos, with_labels=True, node_size=500, node_color="lightblue", arrowsize=15)
            #     plt.show()

            # # katz中心性
            # katz_centrality = nx.katz_centrality(graph_propagate, alpha=0.1)
            # print('katz_centrality:', katz_centrality)

            # 将A矩阵加上单位矩阵I，表示节点本身也参与传播
            A_propagate = A_propagate + np.eye(A_propagate.shape[0])
            # 归一化A矩阵，使得每行之和为1
            row_sums = A_propagate.sum(axis=1, keepdims=True)
            A_propagate = A_propagate / row_sums

            Y_propagate = np.abs(np.array([graph_propagate.nodes[node]['status'] for node in sorted_not_done]))
            Y_propagate = Y_propagate.reshape(1, -1)  # shape (1, n)
            # 迭代传播
            for _ in range(method_params['times']):
                 Y_propagate = method_params['weight'][0]*np.dot(Y_propagate, A_propagate) + method_params['weight'][1]*np.sum(Y_propagate)/len(Y_propagate[0])
                # Y = weight[0] * Y + weight[1] * np.dot(A.T, Y) + weight[2] * Y
            
            # 将传播后的标签值重新赋值给原图中的节点
            Y = Y_propagate
            # 获取Y值大于1的序号
            if noise == 'obs_error':
                max_status_change_index = [i for i in range(len(Y[0])) if Y[0, i] > 1]
            else:
                max_status_change_index = [i for i in range(len(Y[0])) if Y[0, i] >= 1]
            max_status_change_node = [sorted_not_done[i] for i in max_status_change_index]
            # error_nodes_estimated = [list(graph_propagate.nodes)[i] for i in range(len(Y[0])) if Y[0, i] > 1]

            return max_status_change_node

# 生成一个有向无环图
num_tasks = 10
max_dependencies = 10
L_num = 3
sl = SL(num_tasks, max_dependencies, read=True,L_num=L_num)
print('节点数：', sl.graph.number_of_nodes(), '边数：', sl.graph.number_of_edges())
# k = input()

# 开展溯源实验
# 初始化参数
error_prop_list = list(np.arange(0.00, 0.51, 0.01))
metric_list = []
time_list = []
# print(error_prop_list)
for error_prob in error_prop_list:
    print('=================error_prob:', error_prob, '=================')
    method_list = ['LPSI'] # 选择溯源算法['mini_set', 'label_prop', 'LPSI', 'LPSI_deg', 'ILP']
    method_params = {'mini_set': {},
                    'ILP': {'find_in_failed':1,
                            'cardinality_constraint':[1,5],
                        'alpha_uncovered':20.0,   # 未覆盖故障的惩罚（越大越强制覆盖）
                        'beta_touchH':10.0},    # 触达健康的惩罚（越大越避免误伤）},
                    'label_prop': {'times':300, 'weight':[1.0, 0.0, 0.0]},
                    'LPSI': {'alpha':0.5, 'max_iter':5, 'tol':1e-8,
                             'mode':'iterative', 'graph_mode':'undirected'},
                    'LPSI_deg': {'alpha':0.5, 'max_iter':5, 'tol':1e-8,
                                 'mode':'iterative', 'graph_mode':'undirected',
                                 'degree_power':1.0, 'degree_normalization':'none'}}
    noise_list = ['obs_error'] # 添加噪声的类型，[无误差，观测误差，概率传播]=['normal', 'prob_prop', 'obs_error']
    noise_params = {'normal':{},
            'obs_error': {'error_prop':error_prob}, 
            'prob_prop': {'prop_prob':0.8}} # 噪声参数
    
    tic = time.time()
    for method in method_list:
        # print('---------------------method:', method, '-------------------')
        for noise in noise_list:
            # print('=================noise:', noise, '=================')
            F1_score_all = []
            for i in range(100):
            # for error_nodes in sl.graph.nodes():
                # 每次随机从网络节点中选择2-5个节点
                random_num = random.randint(2,5)
                error_nodes = random.sample(list(sl.graph.nodes()), random_num)
                # error_nodes = [error_nodes]
                # error_nodes = ['030100']
                # print('error_nodes', error_nodes)
                # error_nodes = ['081000', '140200']
                # print('*********************error_nodes:', error_nodes, '********************')
                
                # 将所有节点的status标签重置为1
                sl.reset_status()
                not_done = sl.begin_task(error_nodes, noise=noise, noise_param=noise_params[noise])
                # print("未完成的任务节点:", not_done)

                # sl.snapshot(show=True)
                # # 将节点状态状态标签写入文件
                # sl.write_status()

                error_nodes_estimated = sl.source_locate(method, method_params[method], error_nodes)
                # print(error_nodes_estimated)
                # 使用error_nodes_estimated和error_nodes计算F1_score
                N_right = len(set(error_nodes_estimated) & set(error_nodes))
                N_precision = len(error_nodes_estimated)
                N_recall = len(error_nodes)
                F1_score = 2 * N_right / (N_precision + N_recall)
                
                # error_nodes_estimated == error_nodes
                F1_score_all.append(F1_score)
            metric_list.append(np.average(F1_score_all))
            print('平均F1-score为：', np.average(F1_score_all))
    toc = time.time()
    time_list.append(toc - tic)
    print('总耗时：', toc - tic)
    # 为避免代码运行中断，及时将结果保存为json文件
    with open('results/SL_multiple_'+method_list[0]+'_OE_L'+str(L_num)+'.json', 'w') as f:
        json.dump({'error_prop_list': error_prop_list, 'precision': metric_list, 'time': time_list}, f)
    # k = input()

# 将结果保存为json文件
with open('results/SL_multiple_'+method_list[0]+'_OE_L'+str(L_num)+'.json', 'w') as f:
    json.dump({'error_prop_list': error_prop_list, 'precision': metric_list, 'time': time_list}, f)


# # 绘制准确率和求解时间随着误差比例变化的图像
# with open('results/SL_multiple_MS_PP_L'+str(L_num)+'.json', 'r') as f:
#     results = json.load(f)
# error_prop_list = results['error_prop_list']
# precision = results['precision']
# time = results['time']
# # 绘制准确率曲线
# plt.figure(figsize=(10, 4))
# plt.plot(error_prop_list, precision)
# plt.xlabel('Observation Error Proportion')
# plt.ylabel('Precision')
# plt.title('SL Source Localization Precision vs Observation Error Proportion (L'+str(L_num)+')')
# plt.grid()
# plt.savefig('results/SL_single_obs_error_precision_L'+str(L_num)+'.png')
# plt.show()
# # 绘制求解时间曲线
# plt.figure(figsize=(10, 4))
# plt.plot(error_prop_list, time)
# plt.xlabel('Observation Error Proportion')
# plt.ylabel('Time (s)')
# plt.title('SL Source Localization Time vs Observation Error Proportion (L'+str(L_num)+')')
# plt.grid()
# plt.savefig('results/SL_single_obs_error_time_L'+str(L_num)+'.png')
# plt.show()

# # 返回节点及其status属性值
# print(sl.graph.nodes(data=True))
# sl.snapshot(show=True)

# # # 将节点的status标签沿着连边传播
# from dag_bottleneck_toolkit import dag_bottleneck_scores

# # A: 邻接矩阵（行 i -> 列 j），Y: 快照标签（0 完成，1 未完成）
# A = nx.to_numpy_array(sl.graph, nodelist=sorted(sl.graph.nodes))  # shape (n, n)
# Y = np.array([sl.graph.nodes[node]['status'] for node in sorted(sl.graph.nodes)])  # shape (n,)
# Y = Y.reshape(-1, 1)  # shape (n, 1)

# out = dag_bottleneck_scores(A, Y, nodes=None, gamma=0.7, K=3, lam=0.8, mu=0.2)
# print(out["df"])         # 节点分数与局部峰值
# H = out["H"]             # 数值分数（越大越像阻塞）
# peaks = out["local_peaks"]
