import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# 读取节点和边数据
node_df = pd.read_excel('nodelist.xlsx', dtype={'id': str})  # id列为 'id'
edge_df = pd.read_excel('edgelist_detailed.xlsx', dtype={'source': str, 'target': str})  #  'source' 和 'target'列为str

# 创建空的有向图
G = nx.DiGraph()

# 添加节点
for index, row in node_df.iterrows():
    node_id = row['id']
    node_label = row['label']
    G.add_node(node_id)
    G.nodes[node_id]['label'] = node_label
    G.nodes[node_id]['time'] = row['time']
    G.nodes[node_id]['cost'] = row['cost']

# 添加边
for index, row in edge_df.iterrows():
    source = row['source']
    target = row['target']
    relation_attr = row['relation']
    G.add_edge(source, target)
    G[source][target]['relation'] = relation_attr


DSM = nx.adjacency_matrix(G).toarray()# 这一行报错AttributeError: module 'scipy.sparse' has no attribute 'coo_array'，原因是什么？


# print(DSM.shape)  # 打印DSM的形状

# 计算每个节点的最早完成时间（EF）
ef = {}  # earliest finish time
topo_order = nx.topological_sort(G)
# print("拓扑排序结果:", list(topo_order))

for node in topo_order:
    predecessors = G.predecessors(node)
    max_ef = 0
    for pred in predecessors:
        max_ef = max(max_ef, ef[pred])
    ef[node] = max_ef + G.nodes[node]['time']

# 整个项目的完成时间就是最后一个节点的 EF
project_time = max(ef.values())
print(f"项目总完成时间: {project_time} 单位时间")

# # 绘制网络图
# plt.figure(figsize=(10, 8))
# pos = nx.spring_layout(G, k=0.6, seed=42)
# nx.draw(G, pos, with_labels=True, node_size=300, node_color='skyblue', font_size=10, edge_color='gray')
# plt.title("Network Graph")
# plt.show()