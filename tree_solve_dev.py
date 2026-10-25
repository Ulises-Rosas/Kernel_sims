
import copy
from collections import defaultdict


import numpy as np
from phylokrr.treeio import is_leaf

def get_postorder(root):

    S = [root]
    L = []
    while S:
        v = S.pop()
        L.append(v)

        if is_leaf(v):
            continue

        l,r = v.left, v.right
        S.append(l)
        S.append(r)

    return L


def prune_tree(LP, root):
    """
    In a binary tree, prune a tree in place by removing the leaves and their ancestors 
    and by-passing the ancestors of the leaves to be pruned. 

    LP: list of leaves to be pruned from the tree. Each leaf is a node object.

    returns the new root of the pruned tree. If the original root is pruned, then the new root will be returned.
    """
    # prune species from the tree from scratch
    # define 4 nodes 
    # y = ancestor of the species to be pruned, 
    # x = species to be pruned, 
    # z = sibling of x,
    # q = ancestor of y if y is not the root node, otherwise q = None

    if len(LP) == 0:
        print("No leaves to prune. Returning the original root.")
        return root

    for x in LP:
        y = x.ancestor
        z = y.left if y.right == x else y.right

        # check if y is root or not
        # we go first for non-root node
        if y.ancestor is not None:
            q = y.ancestor
            if q.left == y:
                q.left = z
            else:
                q.right = z

            z.ancestor = q
            z.branch_length += y.branch_length
        else:
            # we go for the case of root node
            # if we y has no ancestors and the right child is not a list (i.e., rooted tree).
            assert not isinstance(z.right, list), "the right child of the root node is a list. We only support rooted trees for now."
            z.ancestor = None
            z.branch_length = 0.0
            root = z

    return root

def clone_nodes(nodes: list) -> tuple:
    """
    Clone a list of nodes and return the new root and the new list of nodes.

    returns:
    --------
    new_root: 
        the root node of the cloned tree.
    new_nodes: 
        the list of nodes in the cloned tree.
    """

    # Pass 1: shallow-copy every node (keeps weights, labels, etc.)
    mapping = {n.index: copy.copy(n) for n in nodes}

    new_root = None
    # Pass 2: remap pointers to the new nodes
    for new in mapping.values():
        # print("new node: ", new)
        # new = mapping[0]

        if not new.ancestor:
            new_root = new
        else:
            new.ancestor = mapping[new.ancestor.index]

        if is_leaf(new):
            continue

        new.left = mapping[new.left.index]
        if isinstance(new.right, list):
            new.right = [mapping[n.index] for n in new.right]
        else:
            new.right = mapping[new.right.index]

    return new_root, [mapping[n.index] for n in nodes]

def get_training_tree(full_nodes, training_leaves):
    """
    from a set of full tree nodes, get the pruned tree with only training leaves.
    
    full_nodes: the list of nodes in the full tree
    training_leaves: list of training leaves

    returns:
    --------
    new_nodes: 
        the list of nodes in the pruned tree
    new_root: 
        the root node of the pruned tree.
    """

    training_leaves_set = set(training_leaves)

    new_root, new_nodes = clone_nodes(full_nodes)
    T_nodes = [ n for n in new_nodes\
                if is_leaf(n) and (n.name not in training_leaves_set) ]
    
    new_root = prune_tree(T_nodes, new_root)
    new_nodes = get_postorder(new_root)

    return (new_nodes, new_root)

def get_tree_info(new_root, new_nodes):
    """
    get tree infor for fast A^{-1}b computation.
    This is assuming that the tree is pruned, rooted and binary.

    The pruned tree is aim to be the tree with training leaves only.
    
    new_root: the root node of the pruned tree. 
    new_nodes: the list of nodes in the pruned tree

    returns:
    --------
    children: 
        a dictionary where children[n] is a list of the children of node n.
    parent:
        a dictionary where parent[n] is the parent of node n.
    length:
        a dictionary where length[n] is the branch length of node n.
    root:
        the index of the root node.
    leaves:
        a list of the indices of the leaf nodes.
    """
    
    new_nodes = get_postorder(new_root)
    leaves = [n.index for n in new_nodes if is_leaf(n)]
    len(leaves)

    children = {}
    parent = {}
    length = {}
    root = new_root.index
    leaves = []

    for n in new_nodes:
        if is_leaf(n):
            leaves.append(n.index)
        else:
            children[n.index] = [n.left.index, n.right.index]
            parent[n.left.index] = n.index
            parent[n.right.index] = n.index

        if n == new_root:
            continue

        length[n.index] = n.branch_length

    return children, parent, length, root, leaves


def build_diag_rho_w(children, parent, length, alpha, root):
    """
    Build diagonal entries of the sparse joint precision matrix Q over
    ALL nodes (leaves + internal), for the OU process running along the
    tree with unit stationary variance and per-edge correlation
    rho_e = exp(-alpha * length_e).
    """
    diag = defaultdict(float)
    rho, w = {}, {}
    for v, l in length.items():
        # v,l
        # if v == root:
        #     continue
        r = np.exp( -alpha * l )
        rho[v] = r
        # wv = r / (1 - r**2)
        wv = r / -np.expm1(-2 * alpha * l) 
        w[v] = wv
        # child's own precision contribution
        # diag[v] += 1.0 /(1 - r**2) 
        diag[v] += 1.0 / -np.expm1(-2 * alpha * l) 

        # parent's contribution from this edge
        # diag[parent[v]] += r**2 / (1 - r**2)
        # print(f"v={v}, l={l:.3f}, r={r:.3f}, wv={wv:.3f}, diag[v]={diag[v]:.3f}, diag[parent[v]]={diag[parent[v]]:.3f}")
        # print(f"v={v}, l={l:.3f}, r={r:.3f}, wv={wv:.3f},")
        # print("parent", parent)
        diag[parent[v]] += r**2 / -np.expm1(-2 * alpha * l)  
    
        
    diag[root] += 1.0  # stationary N(0,1) prior at the root
    return diag, rho, w

def solve_A_inv_b(children, parent, length, alpha, root, b, leaves):
    """
    Exact O(n) computation of x = A^{-1} b, where
        A_ij = exp(-alpha * t_ij),   t_ij = tree path length between leaves i,j.

    children : dict node -> list of child nodes
    parent   : dict node -> parent node        (root excluded)
    length   : dict node -> branch length to its parent (root excluded)
    b        : dict leaf -> value
    leaves   : set/list of leaf node ids
    """
    diag, rho, w = build_diag_rho_w(children, parent, length, alpha, root)
    hidden = set(children.keys())  # internal nodes (have children), includes root

    # RHS contribution to hidden nodes from their LEAF children: Q_HL @ b
    d = defaultdict(float)
    for v in leaves:
        d[parent[v]] += -w[v] * b[v]

    # Sub-tree induced on hidden nodes only
    hidden_parent = {u: parent[u] for u in hidden if u != root}
    hidden_children = defaultdict(list)
    for u, p in hidden_parent.items():
        hidden_children[p].append(u)

    # iterative post-order traversal of the hidden sub-tree
    postorder = []
    stack = [(root, False)]
    while stack:
        node, processed = stack.pop()
        if processed:
            postorder.append(node)
        else:
            stack.append((node, True))
            for c in hidden_children[node]:
                stack.append((c, False))

    # ---- Step 3: solve Q_HH z = d via fill-free tree elimination ----
    diag_eff = dict(diag)
    d_eff = dict(d)
    factors = {}
    for u in postorder:
        if u == root:
            continue
        p = hidden_parent[u]
        wu = w[u]
        diag_eff[p] -= wu * wu / diag_eff[u]
        d_eff[p] = d_eff.get(p, 0.0) + wu * d_eff.get(u, 0.0) / diag_eff[u]
        factors[u] = (wu, diag_eff[u], d_eff[u])

    z = {root: d_eff.get(root, 0.0) / diag_eff[root]}
    for u in reversed(postorder):
        if u == root:
            continue
        wu, diagu, du = factors[u]
        z[u] = (du + wu * z[hidden_parent[u]]) / diagu

    # ---- Step 4: x = Q_LL b - Q_LH z  (leaf-level Schur-complement apply) ----
    x = {}
    for v in leaves:
        p = parent[v]
        x[v] = b[v] / (1 - rho[v]**2) + w[v] * z[p]
    return x


# ---------------------------------------------------------------------
# Verification against brute-force dense solve on a random tree
# ---------------------------------------------------------------------
def random_tree(n_leaves, seed=0):
    rng = np.random.default_rng(seed)
    node_id = [0]
    def new_id():
        node_id[0] += 1
        return node_id[0]

    children, parent, length = defaultdict(list), {}, {}

    def build(depth, max_depth):
        nid = new_id()
        if depth >= max_depth or (depth > 1 and rng.random() < 0.35):
            return nid, True  # leaf
        k = rng.integers(2, 3)  # 2 or 3 children -> variable degree
        is_leaf = False
        for _ in range(k):
            cid, _ = build(depth + 1, max_depth)
            l = rng.uniform(0.05, 1.5)
            children[nid].append(cid)
            parent[cid] = nid
            length[cid] = l
        return nid, is_leaf

    root, _ = build(0, 2)
    leaves = [v for v in range(1, node_id[0] + 1) if v not in children]
    return dict(children), parent, length, root, leaves


def patristic_distances(children, parent, length, root, leaves):
    # distance from every node to the root
    depth = {root: 0.0}
    order = [root]
    i = 0
    while i < len(order):
        u = order[i]; i += 1
        for c in children.get(u, []):
            depth[c] = depth[u] + length[c]
            order.append(c)
    t = np.zeros((len(leaves), len(leaves)))
    anc = {}  # ancestor path (as set) for each leaf, for MRCA computation
    def ancestors(v):
        path = []
        while v != root:
            path.append(v)
            v = parent[v]
        path.append(root)
        return path
    paths = {v: ancestors(v) for v in leaves}
    for a, i_ in enumerate(leaves):
        set_i = set(paths[i_])
        for bb, j_ in enumerate(leaves):
            if i_ == j_:
                continue
            # find MRCA
            for node in paths[j_]:
                if node in set_i:
                    mrca = node
                    break
            t[a, bb] = (depth[i_] - depth[mrca]) + (depth[j_] - depth[mrca])
    return t


if __name__ == "__main__":
    children, parent, length, root, leaves = random_tree(4, seed=1)
    alpha = 0.7
    n = len(leaves)
    idx = {v: i for i, v in enumerate(leaves)}

    t = patristic_distances(children, parent, length, root, leaves)
    A = np.exp(-alpha * t)
    np.fill_diagonal(A, 1.0)

    rng = np.random.default_rng(42)
    b_vec = rng.normal(size=n)
    b = {leaves[i]: b_vec[i] for i in range(n)}

    x_fast = solve_A_inv_b(children, parent, length, alpha, root, b, leaves)
    x_fast_vec = np.array([x_fast[v] for v in leaves])

    x_brute = np.linalg.solve(A, b_vec)

    print("n leaves:", n)
    print("max abs error:", np.max(np.abs(x_fast_vec - x_brute)))
    print("fast  x[:5]:", x_fast_vec[:5])
    print("brute x[:5]:", x_brute[:5])
