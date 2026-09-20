# 使用gurobi求解器求解最小源集合问题
# 硬约束版本：必须覆盖所有故障，且不能触达健康
# 软约束版本：允许未覆盖故障和触达健康，但会有惩罚
try:
    import gurobipy as gp
    from gurobipy import GRB
except ModuleNotFoundError:
    gp = None
    GRB = None


def _require_gurobi():
    if gp is None:
        raise RuntimeError(
            "ILP requires the optional 'gurobipy' package and a usable Gurobi "
            "license. LPSI and the other non-ILP methods do not require it."
        )

def reachable_sets(adj, nodes):
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

def reachable_sets_prob(adj, nodes, failed=None):
    """
    返回每个节点的可达集 R(v)。
    如果提供了 failed 参数，则只考虑属于同一连通分量的故障节点。
    
    Args:
        adj: 邻接表 {u: [v1, v2, ...]}
        nodes: 节点集合
        failed: 故障节点集合（可选）
        
    Returns:
        每个节点的可达集字典
    """
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
    def dfs_error(start):
        """仅在故障子图中搜索"""
        seen, st = set(), [start]
        while st:
            x = st.pop()
            if x in seen or x not in failed: 
                continue
            seen.add(x)
            for y in adj.get(x, []):
                if y not in seen and y in failed:
                    st.append(y)
        return seen
    
    # 如果没有指定故障节点，按原来的方式处理
    if failed is None:
        R = {v: dfs(v) for v in nodes}
        return R
    
    # 如果指定了故障节点，则找出每个节点在故障子图中的可达集
    failed = set(failed)
    R = dict()
    for v in nodes:
        if v in failed:
            R[v] = dfs_error(v)
        else:
            R[v] = dfs(v)
    return R

def ilp_min_source_cover_hard(adj, failed, healthy=None, cardinality_constraint=None, candidates=None, log_to_console=False):
    """
    最小源集合 (硬约束):适用于理想模型（normal），因为后续函数比本函数适用性更广，当前没有使用
      - 覆盖全部 failed
      - 任一被选候选源 v 的可达集不得包含 healthy 中的点
    返回: (选中的源集合 S*, 模型对象)
    """
    _require_gurobi()
    failed = set(failed)
    H = set(healthy) if healthy else set()

    all_nodes = set(adj.keys()) | {w for vs in adj.values() for w in vs}
    cand = set(candidates) if candidates is not None else all_nodes #可以简化候选集，缩小到故障节点集？？

    # 仅保留能触达故障的候选
    R = reachable_sets(adj, cand)
    R_F = {v: (R[v] & failed) for v in cand if (R[v] & failed)} # 能够触达故障节点的候选节点，及其故障可达集
    touch_H = {v for v in R if H and (R[v] & H)} # 能够触达健康节点的候选节点

    # 若有故障节点无法被任何候选触达，模型应判定不可行（与传播假设不符）
    coverable = set().union(*R_F.values()) if R_F else set() # 所有能被覆盖的故障节点
    if not failed.issubset(coverable):
        raise ValueError(f"Some failed nodes cannot be covered by any candidate sources. "
                         f"Failed: {failed}, coverable: {coverable}")

    m = gp.Model("PrimarySource_SetCover_Hard") # 自定义名字为"PrimarySource_SetCover_Hard"
    # m.setParam('Seed', 0) # 固定Gurobi的随机种子
    # m.setParam('Deterministic', 1)  # 启用确定性模式
    # m.setParam('Method', 1)  # 固定使用单纯形法，确保结果一致性
    m.Params.OutputFlag = 1 if log_to_console else 0  # 是否打印求解日志

    # 决策变量：x_v = 1 选择 v 为源
    x = {v: m.addVar(vtype=GRB.BINARY, name=f"x_{v}") for v in R_F.keys()}

    # 目标：min Σ x_v
    m.setObjective(gp.quicksum(x.values()), GRB.MINIMIZE)

    # 覆盖所有故障：∀u∈F, Σ_{v: u∈R(v)} x_v ≥ 1
    for u in failed:
        vars_covering_u = [x[v] for v, cov in R_F.items() if u in cov]
        if vars_covering_u:
            m.addConstr(gp.quicksum(vars_covering_u) >= 1, name=f"cover_{u}")
        else:
            # 无法覆盖，强制不可行以显式暴露数据或候选集问题
            # (也可 raise ValueError 提前返回)
            m.addConstr(0 >= 1, name=f"infeasible_{u}")

    # 健康硬约束：触达 H 的候选不可选
    for v in touch_H:
        if v in x:
            m.addConstr(x[v] == 0, name=f"forbid_{v}_touchH")
    
    # 添加基数约束（如果提供了）
    if cardinality_constraint is not None:
        m.addConstr(sum(x[v] for v in cand) == cardinality_constraint, 
                       name="cardinality_constraint")

    m.optimize()

    S_opt = {v for v in x if x[v].X > 0.5 and m.Status == GRB.OPTIMAL}
    return S_opt, m

def ilp_min_source_cover_soft_prob(adj, failed, healthy=None, cardinality_constraint=None, candidates=None,
                              alpha_uncovered=1.0, beta_touchH=0.5, log_to_console=False):
    """
    软约束版本（容错/带惩罚）：适用于单层溯源、多层溯源框架下的，理想模型（normal）和概率传播（prob_prop）模型
      目标: min Σ x_v  + α * (# 未覆盖故障) + β * (# 被触达的健康)
      - z_u ∈ {0,1} 表示故障 u 是否未被覆盖
      - y_w ∈ {0,1} 表示是否选了触达健康节点 w 的源（合并统计）
    """
    _require_gurobi()
    failed = set(failed)
    H = set(healthy) if healthy else set()

    all_nodes = set(adj.keys()) | {w for vs in adj.values() for w in vs}
    cand = set(candidates) if candidates is not None else all_nodes

    R = reachable_sets_prob(adj, cand, failed)
    R_F = {v: (R[v] & failed) for v in cand if (R[v] & failed)}

    # 健康触达统计：若选择了某 v 且 R(v) ∩ H ≠ ∅，就会“误伤”这些健康节点
    v_touchH = {v: (R[v] & H) for v in cand if H and (R[v] & H)}

    # 计算所有故障节点的可达集合并集，即正常情况下可能被故障影响的节点
    failed_reachable_set = set()
    for v in failed:
        failed_reachable_set |= R.get(v, set())

    # 只有当健康节点不在故障可达集合内时，才应该被惩罚
    # 即: H_not_reachable = H - failed_reachable_set
    H_should_not_touch = H - failed_reachable_set

    m = gp.Model("PrimarySource_SetCover_Soft")
    # m.setParam('Seed', 0) # 固定Gurobi的随机种子
    # m.setParam('Deterministic', 1)  # 启用确定性模式
    # m.setParam('Method', 1)  # 固定使用单纯形法，确保结果一致性
    m.Params.OutputFlag = 1 if log_to_console else 0

    # 变量：
    # x_v：是否选 v 为源
    x = {v: m.addVar(vtype=GRB.BINARY, name=f"x_{v}") for v in cand}

    # z_u：故障 u 是否未被覆盖（1=未解释/未覆盖；0=已覆盖）
    z = {u: m.addVar(vtype=GRB.BINARY, name=f"z_uncovered_{u}") for u in failed}

    # y_w：健康 w 是否被“误触达”（1=至少一个被选源能到达 w）
    y = {w: m.addVar(vtype=GRB.BINARY, name=f"y_touchH_{w}") for w in H}

    # 约束1：覆盖或未覆盖（二选一）
    #   对每个故障 u：sum_{v:u∈R(v)} x_v + z_u >= 1
    for u in failed:
        cover_vars = [x[v] for v, cov in R_F.items() if u in cov]
        if cover_vars:
            m.addConstr(gp.quicksum(cover_vars) + z[u] >= 1, name=f"cover_or_uncovered_{u}")
        else:
            # 若无候选能覆盖 u，只能靠 z_u=1 来承担罚分
            m.addConstr(z[u] == 1, name=f"must_uncovered_{u}")

    # 约束2：健康“误触达”的线性化
    #   若存在 v 被选中 且 w ∈ R(v)，则 y_w 必须为 1
    #   用 y_w >= x_v  (对所有 w∈R(v)∩H) 来放松（这会略微过惩罚，但简单稳健）
    # if noise == 'prob_prop':
    for v, Hw in v_touchH.items():
        for w in Hw:
            if  w in H_should_not_touch: # 只有误触那些当前故障节点无法触及的健康节点，才会被惩罚
                m.addConstr(y[w] >= x[v], name=f"touchH_link_{v}_{w}")
    # else:
    #     for v, Hw in v_touchH.items():
    #         for w in Hw:
    #             m.addConstr(y[w] >= x[v], name=f"touchH_link_{v}_{w}")
    
    # 约束3：添加基数约束（如果提供了）
    # if cardinality_constraint is not None:
    #     m.addConstr(sum(x[v] for v in cand) == cardinality_constraint, 
    #                    name="cardinality_constraint")
    if cardinality_constraint is not None:
        a, b = cardinality_constraint  # 假设传进来的是 (a, b) 或 [a, b]
        m.addConstr(gp.quicksum(x[v] for v in cand) >= a, name="cardinality_lb")
        m.addConstr(gp.quicksum(x[v] for v in cand) <= b, name="cardinality_ub")


    # 目标：min  Σ x_v + α Σ z_u + β Σ y_w
    m.setObjective(
        gp.quicksum(x.values()) +
        alpha_uncovered * gp.quicksum(z.values()) +
        (beta_touchH * gp.quicksum(y.values()) if H else 0),
        GRB.MINIMIZE
    )

    m.optimize()

    S_soft = {v for v in x if x[v].X > 0.5 and m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL)}
    stats = {
        "uncovered_failed": [u for u in z if z[u].X > 0.5],
        "touched_healthy": [w for w in y if y[w].X > 0.5],
        "obj": m.ObjVal if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) else None,
        "status": m.Status
    }
    return S_soft, stats, m

def ilp_min_source_cover_soft(adj, failed, healthy=None, cardinality_constraint=None,candidates=None,
                              alpha_uncovered=1.0, beta_touchH=0.5,  log_to_console=False):
    """
    软约束版本（容错/带惩罚）：适用于单层溯源框架下的，观测误差（obs_error）模型
      目标: min Σ x_v  + α * (# 未覆盖故障) + β * (# 被触达的健康)
      - z_u ∈ {0,1} 表示故障 u 是否未被覆盖
      - y_w ∈ {0,1} 表示是否选了触达健康节点 w 的源（合并统计）
    """
    _require_gurobi()
    failed = set(failed)
    H = set(healthy) if healthy else set()

    all_nodes = set(adj.keys()) | {w for vs in adj.values() for w in vs}
    cand = set(candidates) if candidates is not None else all_nodes

    R = reachable_sets(adj, cand)
    R_F = {v: (R[v] & failed) for v in cand if (R[v] & failed)}

    # 健康触达统计：若选择了某 v 且 R(v) ∩ H ≠ ∅，就会“误伤”这些健康节点
    v_touchH = {v: (R[v] & H) for v in cand if H and (R[v] & H)}

    # 计算所有故障节点的可达集合并集，即正常情况下可能被故障影响的节点
    failed_reachable_set = set()
    for v in failed:
        failed_reachable_set |= R.get(v, set())

    m = gp.Model("PrimarySource_SetCover_Soft")
    # m.setParam('Seed', 0) # 固定Gurobi的随机种子
    # m.setParam('Deterministic', 1)  # 启用确定性模式
    # m.setParam('Method', 1)  # 固定使用单纯形法，确保结果一致性
    m.Params.OutputFlag = 1 if log_to_console else 0

    # 变量：
    # x_v：是否选 v 为源
    x = {v: m.addVar(vtype=GRB.BINARY, name=f"x_{v}") for v in cand}

    # z_u：故障 u 是否未被覆盖（1=未解释/未覆盖；0=已覆盖）
    z = {u: m.addVar(vtype=GRB.BINARY, name=f"z_uncovered_{u}") for u in failed}

    # y_w：健康 w 是否被“误触达”（1=至少一个被选源能到达 w）
    y = {w: m.addVar(vtype=GRB.BINARY, name=f"y_touchH_{w}") for w in H}

    # 约束1：覆盖或未覆盖（二选一）
    #   对每个故障 u：sum_{v:u∈R(v)} x_v + z_u >= 1
    for u in failed:
        cover_vars = [x[v] for v, cov in R_F.items() if u in cov]
        if cover_vars:
            m.addConstr(gp.quicksum(cover_vars) + z[u] >= 1, name=f"cover_or_uncovered_{u}")
        else:
            # 若无候选能覆盖 u，只能靠 z_u=1 来承担罚分
            m.addConstr(z[u] == 1, name=f"must_uncovered_{u}")

    # 约束2：健康“误触达”的线性化
    #   若存在 v 被选中 且 w ∈ R(v)，则 y_w 必须为 1
    #   用 y_w >= x_v  (对所有 w∈R(v)∩H) 来放松（这会略微过惩罚，但简单稳健）
    for v, Hw in v_touchH.items():
        for w in Hw:
            m.addConstr(y[w] >= x[v], name=f"touchH_link_{v}_{w}")
    
    # 约束3：添加基数约束（如果提供了）
    # if cardinality_constraint is not None:
    #     m.addConstr(sum(x[v] for v in cand) == cardinality_constraint, 
    #                    name="cardinality_constraint")
    if cardinality_constraint is not None:
        a, b = cardinality_constraint  # 假设传进来的是 (a, b) 或 [a, b]
        m.addConstr(gp.quicksum(x[v] for v in cand) >= a, name="cardinality_lb")
        m.addConstr(gp.quicksum(x[v] for v in cand) <= b, name="cardinality_ub")

    # # 目标：min  Σ x_v + α Σ z_u + β Σ y_w
    # m.setObjective(
    #     gp.quicksum(x.values()) +
    #     alpha_uncovered * gp.quicksum(z.values()) +
    #     (beta_touchH * gp.quicksum(y.values()) if H else 0),
    #     GRB.MINIMIZE
    # )
    # 目标：min  Σ x_v + α Σ z_u + β Σ y_w
    m.setObjective(
        gp.quicksum(x.values()) +
        alpha_uncovered * gp.quicksum(z.values()) +
        (beta_touchH * gp.quicksum(y.values()) if H else 0),
        GRB.MINIMIZE
    )

    m.optimize()

    S_soft = {v for v in x if x[v].X > 0.5 and m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL)}
    stats = {
        "uncovered_failed": [u for u in z if z[u].X > 0.5],
        "touched_healthy": [w for w in y if y[w].X > 0.5],
        "obj": m.ObjVal if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) else None,
        "status": m.Status
    }
    return S_soft, stats, m

def ilp_min_source_cover_soft_dynamic(adj, failed, healthy=None, cardinality_constraint=None, not_done_son=None, candidates=None,
                              alpha_uncovered=1.0, beta_touchH=0.5, log_to_console=False):
    """
    软约束版本（容错/带惩罚）：适用于逐层溯源框架下的，观测误差（obs_error）模型
      目标: min Σ x_v  + Σ (not_done_son[u] or alpha_uncovered) * z_u + β * (# 被触达的健康)
      - z_u ∈ {0,1} 表示故障 u 是否未被覆盖
      - y_w ∈ {0,1} 表示是否选了触达健康节点 w 的源（合并统计）
    """
    _require_gurobi()
    failed = set(failed)
    H = set(healthy) if healthy else set()

    all_nodes = set(adj.keys()) | {w for vs in adj.values() for w in vs}
    cand = set(candidates) if candidates is not None else all_nodes

    R = reachable_sets(adj, cand)
    R_F = {v: (R[v] & failed) for v in cand if (R[v] & failed)}

    # 健康触达统计：若选择了某 v 且 R(v) ∩ H ≠ ∅，就会"误伤"这些健康节点
    v_touchH = {v: (R[v] & H) for v in cand if H and (R[v] & H)}

    # 计算所有故障节点的可达集合并集，即正常情况下可能被故障影响的节点
    failed_reachable_set = set()
    for v in failed:
        failed_reachable_set |= R.get(v, set())

    m = gp.Model("PrimarySource_SetCover_Soft")
    # m.setParam('Seed', 0)  # 固定Gurobi的随机种子
    m.Params.OutputFlag = 1 if log_to_console else 0

    # 变量：
    # x_v：是否选 v 为源
    x = {v: m.addVar(vtype=GRB.BINARY, name=f"x_{v}") for v in cand}

    # z_u：故障 u 是否未被覆盖（1=未解释/未覆盖；0=已覆盖）
    z = {u: m.addVar(vtype=GRB.BINARY, name=f"z_uncovered_{u}") for u in failed}

    # y_w：健康 w 是否被"误触达"（1=至少一个被选源能到达 w）
    y = {w: m.addVar(vtype=GRB.BINARY, name=f"y_touchH_{w}") for w in H}

    # 约束1：覆盖或未覆盖（二选一）
    #   对每个故障 u：sum_{v:u∈R(v)} x_v + z_u >= 1
    for u in failed:
        cover_vars = [x[v] for v, cov in R_F.items() if u in cov]
        if cover_vars:
            m.addConstr(gp.quicksum(cover_vars) + z[u] >= 1, name=f"cover_or_uncovered_{u}")
        else:
            # 若无候选能覆盖 u，只能靠 z_u=1 来承担罚分
            m.addConstr(z[u] == 1, name=f"must_uncovered_{u}")

    # 约束2：健康"误触达"的线性化
    #   若存在 v 被选中 且 w ∈ R(v)，则 y_w 必须为 1
    #   用 y_w >= x_v  (对所有 w∈R(v)∩H) 来放松（这会略微过惩罚，但简单稳健）
    for v, Hw in v_touchH.items():
        for w in Hw:
            m.addConstr(y[w] >= x[v], name=f"touchH_link_{v}_{w}")
    
    # 约束3：添加基数约束（如果提供了）
    if cardinality_constraint is not None:
        a, b = cardinality_constraint  # 假设传进来的是 (a, b) 或 [a, b]
        m.addConstr(gp.quicksum(x[v] for v in cand) >= a, name="cardinality_lb")
        m.addConstr(gp.quicksum(x[v] for v in cand) <= b, name="cardinality_ub")

    # 动态alpha值：根据not_done_son属性设置
    alpha_terms = []
    if not_done_son is not None:
        for u in failed:
            # 使用not_done_son属性作为alpha值，如果不存在则使用默认值
            alpha_u = not_done_son.get(u, alpha_uncovered)
            alpha_terms.append(alpha_u * z[u])
    else:
        # 如果没有提供not_done_son，则使用默认值
        alpha_terms = [alpha_uncovered * z[u] for u in failed]

    # 目标：min  Σ x_v + Σ alpha_u * z_u + β Σ y_w
    m.setObjective(
        gp.quicksum(x.values()) +
        gp.quicksum(alpha_terms) +
        (beta_touchH * gp.quicksum(y.values()) if H else 0),
        GRB.MINIMIZE
    )

    m.optimize()

    S_soft = {v for v in x if x[v].X > 0.5 and m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL)}
    stats = {
        "uncovered_failed": [u for u in z if z[u].X > 0.5],
        "touched_healthy": [w for w in y if y[w].X > 0.5],
        "obj": m.ObjVal if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) else None,
        "status": m.Status
    }
    return S_soft, stats, m

def ilp_min_source_cover_soft_dynamic_plus(adj, failed, healthy=None, cardinality_constraint=None, not_done_son=None, candidates=None,
                              alpha_uncovered=1.0, beta_touchH=0.5, num_solutions=10, log_to_console=False):
    """
    软约束版本（容错/带惩罚）：适用于逐层溯源框架下的，观测误差（obs_error）模型
      目标: min Σ x_v  + Σ (not_done_son[u] or alpha_uncovered) * z_u + β * (# 被触达的健康)
      - z_u ∈ {0,1} 表示故障 u 是否未被覆盖
      - y_w ∈ {0,1} 表示是否选了触达健康节点 w 的源（合并统计）
      
    新增参数:
      num_solutions: 返回的候选解数量
    返回:
      (最佳解, 候选解列表及置信度, 统计信息, 模型对象)
    """
    _require_gurobi()
    failed = set(failed)
    H = set(healthy) if healthy else set()

    all_nodes = set(adj.keys()) | {w for vs in adj.values() for w in vs}
    cand = set(candidates) if candidates is not None else all_nodes

    R = reachable_sets(adj, cand)
    # R = reachable_sets_prob(adj, cand, failed)

    R_F = {v: (R[v] & failed) for v in cand if (R[v] & failed)}

    # 健康触达统计：若选择了某 v 且 R(v) ∩ H ≠ ∅，就会"误伤"这些健康节点
    v_touchH = {v: (R[v] & H) for v in cand if H and (R[v] & H)}

    m = gp.Model("PrimarySource_SetCover_Soft")
    # m.setParam('Seed', 0)  # 固定Gurobi的随机种子
    m.Params.OutputFlag = 1 if log_to_console else 0
    
    # 设置参数以获取多个解
    m.setParam(GRB.Param.PoolSolutions, num_solutions)  # 存储的解的数量
    m.setParam(GRB.Param.PoolSearchMode, 2)  # 专注于寻找n个最优解

    # 变量：
    # x_v：是否选 v 为源
    x = {v: m.addVar(vtype=GRB.BINARY, name=f"x_{v}") for v in cand}

    # z_u：故障 u 是否未被覆盖（1=未解释/未覆盖；0=已覆盖）
    z = {u: m.addVar(vtype=GRB.BINARY, name=f"z_uncovered_{u}") for u in failed}

    # y_w：健康 w 是否被"误触达"（1=至少一个被选源能到达 w）
    y = {w: m.addVar(vtype=GRB.BINARY, name=f"y_touchH_{w}") for w in H}

    # 约束1：覆盖或未覆盖（二选一）
    #   对每个故障 u：sum_{v:u∈R(v)} x_v + z_u >= 1
    for u in failed:
        cover_vars = [x[v] for v, cov in R_F.items() if u in cov]
        if cover_vars:
            m.addConstr(gp.quicksum(cover_vars) + z[u] >= 1, name=f"cover_or_uncovered_{u}")
        else:
            # 若无候选能覆盖 u，只能靠 z_u=1 来承担罚分
            m.addConstr(z[u] == 1, name=f"must_uncovered_{u}")

    # 约束2：健康"误触达"的线性化
    #   若存在 v 被选中 且 w ∈ R(v)，则 y_w 必须为 1
    #   用 y_w >= x_v  (对所有 w∈R(v)∩H) 来放松（这会略微过惩罚，但简单稳健）
    for v, Hw in v_touchH.items():
        for w in Hw:
            m.addConstr(y[w] >= x[v], name=f"touchH_link_{v}_{w}")
    
    # 约束3：添加基数约束（如果提供了）
    if cardinality_constraint is not None:
        a, b = cardinality_constraint  # 假设传进来的是 (a, b) 或 [a, b]
        m.addConstr(gp.quicksum(x[v] for v in cand) >= a, name="cardinality_lb")
        m.addConstr(gp.quicksum(x[v] for v in cand) <= b, name="cardinality_ub")

    # 动态alpha值：根据not_done_son属性设置
    alpha_terms = []
    if not_done_son is not None:
        for u in failed:
            # 使用not_done_son属性作为alpha值，如果不存在则使用默认值
            alpha_u = not_done_son.get(u, alpha_uncovered)
            alpha_terms.append(alpha_u * z[u])
    else:
        # 如果没有提供not_done_son，则使用默认值
        alpha_terms = [alpha_uncovered * z[u] for u in failed]

    # 目标：min  Σ x_v + Σ alpha_u * z_u + β Σ y_w
    m.setObjective(
        gp.quicksum(x.values()) +
        gp.quicksum(alpha_terms) +
        (beta_touchH * gp.quicksum(y.values()) if H else 0),
        GRB.MINIMIZE
    )

    m.optimize()

    # 获取多个解
    solutions = []
    if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        # 获取解的数量
        num_found = m.SolCount
        for i in range(min(num_solutions, num_found)):
            m.setParam(GRB.Param.SolutionNumber, i)
            solution = {v for v in x if x[v].Xn > 0.5}
            obj_val = m.PoolObjVal
            # 置信度可以基于目标函数值计算，值越小置信度越高
            solutions.append({
                'solution': solution,
                'objective': obj_val,
                'confidence': 1.0 / (1.0 + obj_val)  # 简单的置信度计算方式
            })

    # 按置信度排序
    solutions.sort(key=lambda s: s['confidence'], reverse=True)
    
    # 最佳解
    S_soft = solutions[0]['solution'] if solutions else set()
    
    stats = {
        "uncovered_failed": [u for u in z if z[u].X > 0.5] if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) else [],
        "touched_healthy": [w for w in y if y[w].X > 0.5] if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) else [],
        "obj": m.ObjVal if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) else None,
        "status": m.Status,
        "solutions_count": len(solutions)
    }
    
    return S_soft, solutions, stats, m

# 在需要调用该函数的地方，修改调用方式：
# 原来: S_soft, stats, model = ilp_min_source_cover_soft_dynamic(...)
# 现在: S_soft, candidate_solutions, stats, model = ilp_min_source_cover_soft_dynamic(...)

# ====== 示例用法 ======
if __name__ == "__main__":
    # 硬约束版本示例
    # 一个小型 DAG 示例
    adj = {
        "a": ["b", "c"],
        "b": ["d"],
        "c": ["d", "e"],
        "d": ["f"],
        "e": ["f"],
        "f": []
    }
    failed = {"a", "d", "e", "f"}   # 最终观测为故障的节点
    healthy = {"b"}            # 已知健康的节点（硬约束）
    S_opt, model = ilp_min_source_cover_hard(adj, failed, healthy, log_to_console=True)
    print("Optimal sources (hard):", S_opt)

    # 软约束版本示例
    # adj = {
    #     "a": ["b", "c"],
    #     "b": ["d"],
    #     "c": ["d", "e"],
    #     "d": ["f"],
    #     "e": ["f"],
    #     "f": []
    # }
    # failed = {"d", "e", "f"}
    # healthy = {"c"}  # 允许被误触达，但要付罚分

    # S_soft, stats, model = ilp_min_source_cover_soft(
    #     adj, failed, healthy,
    #     alpha_uncovered=4.0,   # 未覆盖故障的惩罚（越大越强制覆盖）
    #     beta_touchH=8.0,       # 触达健康的惩罚（越大越避免误伤）
    #     log_to_console=True
    # )
    # print("Selected sources (soft):", S_soft)
    # print("Uncovered failed:", stats["uncovered_failed"])
    # print("Touched healthy:", stats["touched_healthy"])
    # print("Objective:", stats["obj"], "Status:", stats["status"])
