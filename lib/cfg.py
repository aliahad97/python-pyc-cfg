import opcode, sys, dis, pygraphviz
from xml.etree.ElementPath import find
from pickletools import opcodes
from os.path import exists
from difflib import SequenceMatcher
# Opcodes
ops_list = [ 
#NOP
'NOP',
# load/store
'LOAD_CONST', 'LOAD_FAST', 'STORE_FAST', 'STORE_NAME', 'LOAD_NAME', 'STORE_ATTR', 'LOAD_ATTR', 'LOAD_GLOBAL', 'STORE_GLOBAL',
'STORE_SUBSCR', 'STORE_DEREF', 'LOAD_CLOSURE', 'GET_AWAITABLE', 'LOAD_DEREF', 'STORE_ANNOTATION', 'FORMAT_VALUE',
# compare
'COMPARE_OP',
# arithmetic
'INPLACE_ADD', 'INPLACE_SUBTRACT', 'INPLACE_MULTIPLY','INPLACE_TRUE_DIVIDE', 'INPLACE_MODULO', 'BINARY_XOR', 'INPLACE_POWER', 'INPLACE_LSHIFT', 'INPLACE_RSHIFT', 'INPLACE_AND', 'INPLACE_XOR',
'UNPACK_SEQUENCE','UNPACK_EX', 'DELETE_FAST', 'BINARY_ADD', 'BINARY_MODULO', 'BINARY_SUBSCR',
'BINARY_MULTIPLY', 'BINARY_ADD', 'DELETE_SUBSCR', 'UNARY_NOT', 'LIST_APPEND', 'MAP_ADD', 'SET_ADD', 'BINARY_OR', 
'BINARY_FLOOR_DIVIDE','BINARY_RSHIFT','BINARY_LSHIFT', 'BINARY_SUBTRACT', 'ROT_TWO', 'ROT_THREE','ROT_FOUR', 'DUP_TOP_TWO', 'UNARY_POSITIVE', 'UNARY_NEGATIVE',
'BINARY_MATRIX_MULTIPLY','UNARY_INVERT', 'INPLACE_MATRIX_MULTIPLY', 'BINARY_POWER', 'INPLACE_FLOOR_DIVIDE', 'BINARY_TRUE_DIVIDE', 'INPLACE_OR', 'BINARY_AND',
# functions
'MAKE_FUNCTION', 'CALL_FUNCTION', 'CALL_FUNCTION_KW', 'CALL_FUNCTION_EX', 'RETURN_VALUE', 'EXTENDED_ARG', 'LOAD_METHOD', 'CALL_METHOD', 'MAKE_FUNCTION','PRINT_EXPR',
# class
'LOAD_BUILD_CLASS', 'DELETE_NAME', 'DELETE_GLOBAL', 'DELETE_DEREF',
# stack
'POP_BLOCK', 'POP_TOP', 'DUP_TOP', 
# data structure
'BUILD_LIST', 'BUILD_MAP',  'BUILD_TUPLE', 'BUILD_CONST_KEY_MAP', 'BUILD_SLICE', 'BUILD_STRING', 'BUILD_SET',
# for
'FOR_ITER', 'BREAK_LOOP', 'GET_ITER', 'GET_AITER', 'GET_ANEXT', 'CONTINUE_LOOP', 'GET_YIELD_FROM_ITER',
# try catch
'END_FINALLY', 'RAISE_VARARGS', 'POP_EXCEPT', 'BEGIN_FINALLY', 'POP_FINALLY',
# import
'IMPORT_FROM', 'IMPORT_NAME', 'IMPORT_STAR',
# Delete
'DELETE_ATTR',
# etc
'YIELD_FROM', 'YIELD_VALUE', 'SETUP_ANNOTATIONS', 'BEFORE_ASYNC_WITH', 'WITH_CLEANUP_START',
'WITH_CLEANUP_FINISH', 'BUILD_MAP_UNPACK', 'END_ASYNC_FOR', 'LOAD_CLASSDEREF', 'BUILD_LIST_UNPACK', 'BUILD_MAP_UNPACK_WITH_CALL', 'BUILD_TUPLE_UNPACK', 'BUILD_SET_UNPACK', 'BUILD_TUPLE_UNPACK_WITH_CALL'
# ???
];
ops_jumps = ['POP_JUMP_IF_FALSE', 'POP_JUMP_IF_TRUE', 'JUMP_IF_TRUE_OR_POP', 'JUMP_IF_FALSE_OR_POP', 'JUMP_ABSOLUTE'] + [ opcode.opname[i] for i in opcode.hasjabs] ;
ops_relative_jumps = ['JUMP_FORWARD'] + [ opcode.opname[i] for i in opcode.hasjrel];
ops_setups = ['SETUP_EXCEPT', 'SETUP_FINALLY', 'SETUP_ASYNC_WITH', 'SETUP_WITH', 'FOR_ITER']; #TODO: For Iter creates an edge
# ops_list = ops_list + 

class CFGNode:
    def __init__(self, i, bid, isTarget= False, line_no = None):
        self.i = i
        '''
        i.opname
        i.opcode
        i.arg
        i.argrepr (human-readable)
        i.offset
        i.starts_line (line start or None)
        i.is_jump_target
        '''
        self.first_node_id = -1
        self.last_node_id = -1
        self.nid = 0 # Node id
        self.bid = bid # Block id
        self.block = False
        self.ins = [] # Remaining instructions in the block
        self.children = [] # Children cfg node
        self.parent = [] # Parent cfg nodes
        self.props = {}
        self.isTarget = isTarget
        self.layer= -1
        self.line_no = line_no
    def add_inst(self, n):
        self.ins.append(n)        
    def add_child(self, n): 
        self.children.append(n)
        n.parent.append(self)
    def get_line_no(self):
        return self.line_no
    
def get_last_instr(node):
    try:
        return node.ins[-1].i.opname
    except:
        return node.i.opname
    
def get_last_instr_nid(node):
    try:
        return (node.ins[-1].i.opname, node.nid)
    except:
        return (node.i.opname, node.nid)

def get_all_dfs_paths(cfg, offset = 0, end = None, nid = False) -> list:
    '''returns all DFS paths given a CFG object. Optional parameters include instruction offset, end node, and whether nid's should be returned with path'''
    paths = [[cfg.opcodes[offset]]] #initialize return value with the first node
    stack = []; visited = {}
    if offset % 2 != 0:
        offset = offset - 1 if offset > 0 else 0 #offset will always be a positive even integer
    if end == None:
        curr_key = 0
        for k, v in cfg.opcodes.items():
            if k > curr_key:
                curr_key = k
                end = v
    stack.append((cfg.opcodes[offset], [cfg.opcodes[offset]]))
    while len(stack) > 0:
        cnode, cpath = stack.pop()
        for adj in cnode.children:
            if adj not in cpath:
                if adj != end:
                    stack.append((adj, cpath + [adj]))
                if cpath + [adj] not in paths: 
                    ''' 
                    Do not add the path if already in the path set
                    Would implement it with a built-in python set, but lists are not hashable.
                    If running time is a concern, the complexity can be easily reduced using hashmaps
                    '''
                    paths.append((cpath + [adj]).copy())
                
    for path in paths:
        for i in range(len(path)):
            if nid == True:
                path[i] = get_last_instr_nid(path[i])
            elif nid == False:
                path[i] = get_last_instr(path[i])
    paths.sort(key = lambda x: len(x))
    return paths

def get_all_dfs_paths_all_offsets(cfg) -> list:
    # print(cfg.opcodes.keys())
    paths = []
    for nid in cfg.opcodes.keys():
        paths += get_all_dfs_paths(cfg, offset = nid)
    print(f"\nNUM PATHS: {len(paths)}\n")
    return paths

# def is_subgraph_dfs(cfg1, cfg2) -> bool:
#     cfg1_dfs_paths = get_all_dfs_paths(cfg1, nid = True)
#     cfg2_dfs_paths = get_all_dfs_paths(cfg2, nid = True)
#     max_path_len = max (max([len(path) for path in cfg1_dfs_paths]), max([len(path) for path in cfg2_dfs_paths]))
#     min_path_len = min (max([len(path) for path in cfg1_dfs_paths]), max([len(path) for path in cfg2_dfs_paths]))
#     for i in range(min_path_len):
#         for j in range(i):
#             if 

def is_subgraph_dfs_path(cfg1, cfg1_offset, cfg2, cfg2_offset) -> bool:
    #TODO: adapt this to work to find subgraphs by testing all possible offsets
    "if the graphs are the same, then they should have all of the same DFS paths"
    cfg1_dfs_paths = get_all_dfs_paths(cfg1, cfg1_offset)
    cfg2_dfs_paths = get_all_dfs_paths(cfg2, cfg2_offset)
    for path in cfg1_dfs_paths:
        if path in cfg2_dfs_paths:
            cfg2_dfs_paths.remove(path)
    return len(cfg2_dfs_paths) == 0


def get_isomorphic_dfs_path(cfg1, cfg2) -> list:
    cfg1_dfs_paths = get_all_dfs_paths(cfg1)
    cfg2_dfs_paths = get_all_dfs_paths(cfg2)
    for path1 in cfg1_dfs_paths:
        for path2 in cfg2_dfs_paths:
            if path1 == path2:
                return path1
    return [] 

def max_subgraph_dfs(cfg1, cfg2) -> list:
    '''
    The algorithm computes all possible dfs paths for cfg1 and cfg2 by calling cfg.get_all_dfs_paths(). 
    Each path from cfg1 is compared to cfg2, where the maximum length DFS path between the two graphs is
    returned.
    '''
    cfg1_dfs_paths = get_all_dfs_paths(cfg1)
    cfg2_dfs_paths = get_all_dfs_paths(cfg2)
    max_path = []
    for path1 in cfg1_dfs_paths:
        for path2 in cfg2_dfs_paths:
            i = 0; curr_max_path = []
            while i < min(len(path1), len(path2)):
                if path1[i] == path2[i]:
                    curr_max_path.append(path1[i])
                i += 1
            if len(max_path) < len(curr_max_path):
                max_path = curr_max_path
    return max_path
            


def node_to_string(node1):
    '''helper function used by CFG_to_hash_graph() and is_subgraph()'''
    node_str = ""
    node_str += node1.i.opname
    # if node1.parent: print(f"NODE ABOVE'S PARENTS{[p.i.opname for p in node1.parent]}")
    for ins in node1.ins:
        node_str += ins.i.opname
    return node_str

def CFG_to_hash_graph(cfg1):
    '''helper function used by is_subgraph() to take in a cfg and return a set of its nodes and edges as hashes'''
    nodes = []; edges = []
    #use iterative DFS to make a list of hashes of nodes and edges
    visited = {}
    stack = []
    stack.append(0) #assumes there will always be a block wit nid 0
    while len(stack) > 0:
        cnode_nid = stack.pop(-1)
        visited[cnode_nid] = 1
        cnode = cfg1.opcodes[cnode_nid]
        cnode_str_hash = hash(node_to_string(cnode))
        nodes.append(cnode_str_hash)
        for node in cnode.children:
            edges.append((cnode_str_hash, hash(node_to_string(node))))
            if visited.get(node.nid) == None:
                stack.append(node.nid)
    return nodes, edges

# def get_node_mappings(cfg) -> list:
#     # TODO: Finish implementation
#     '''
#     - Goal is to create a list of python dictionaries that contains all possible node mappings by mapping node data to unique hashes
#     - In each dictionary: Key = nid; Value = Unique Hash
#     - Multiple mappings with be produced ONLY IF the data we are leveraging (Ex: last instruction) is the same for two+ nodes
#     '''


#     node_maps = [] #an array of hashmaps
#     visited = {}; stack = []; id_to_hash_map = {}
#     stack.append(0) #assumes there will always be a block wit nid 0
#     while len(stack) > 0:
#         cnode_nid = stack.pop(-1)
#         visited[cnode_nid] = 1
#         cnode = cfg1.opcodes[cnode_nid]
#         cnode_str_hash = hash(node_to_string(cnode)) 
#     return

def CFG_to_hash_graph_list(cfg1) -> list:
    '''helper function used by is_subgraph() to take in a cfg and return a set of its nodes and edges as hashes
    conflict resolution for nodes with the same instructions/data within the same graph
    '''
    used_hashes = {}
    #use iterative DFS to make a list of hashes of nodes and edges
    done = False
    while done == False:
        done = True
        nodes = []; edges = []
        visited = {}; stack = []; id_to_hash_map = {}
        stack.append(0) #assumes there will always be a block wit nid 0
        while len(stack) > 0:
            cnode_nid = stack.pop(-1)
            visited[cnode_nid] = 1
            cnode = cfg1.opcodes[cnode_nid]
            cnode_str_hash = hash(node_to_string(cnode))
            if used_hashes.get(cnode_str_hash) != None:
                done = False
                #then resolve the conflict
                cnode_str_hash += used_hashes[cnode_str_hash]
                nodes.append(cnode_str_hash)
                used_hashes[cnode_str_hash] += 1
            else:
                used_hashes[cnode_str_hash] = 1
                nodes.append(cnode_str_hash)
            id_to_hash_map[cnode_nid] = cnode_str_hash
            for node in cnode.children:
                edges.append((cnode_str_hash, hash(node_to_string(node))))
                if visited.get(node.nid) == None:
                    stack.append(node.nid)
    return nodes, edges

def is_subgraph(cfg1, cfg2):
    '''takes in two CFG objects, returns whether either one is a subgraph of the other'''
    cfg1_sub_cfg2 = True
    cfg2_sub_cfg1 = True
    nodes1, edges1 = CFG_to_hash_graph(cfg1)
    nodes2, edges2 = CFG_to_hash_graph(cfg2)
    #first check if graph 1 is a subgraph of graph 2:
    for n1 in nodes1:
        if n1 not in nodes2:
            cfg1_sub_cfg2 = False
    for e in edges1:
        if e not in edges2:
            cfg1_sub_cfg2 = False
    #first check if graph 2 is a subgraph of graph 1:
    for n2 in nodes2:
        if n2 not in nodes1:
            cfg2_sub_cfg1 = False
    for e in edges2:
        if e not in edges1:
            cfg2_sub_cfg1 = False
    return cfg1_sub_cfg2 & cfg2_sub_cfg1

def is_subgraph_last_instruction(cfg1, cfg2):
    '''
    need a new method:
    do an exponential recursive dfs that adds a recursive call IFF a
    valid, no-visited adjacent node is present
    - in theory this would be a dangerously recursive exponential algorithm
    - in practice, it is assumed that it will behave fine in almost all test
      cases through its use of heuristics
    Approach: DFS-sweep algorithm as the base that uses a helper function
    - will likely
    '''
    '''takes in two CFG objects, returns whether either one is a subgraph of the other'''
    cfg1_sub_cfg2 = True
    cfg2_sub_cfg1 = True
    nodes1, edges1 = CFG_to_hash_graph(cfg1)
    nodes2, edges2 = CFG_to_hash_graph(cfg2)
    #first check if graph 1 is a subgraph of graph 2:
    for n1 in nodes1:
        if n1 not in nodes2:
            cfg1_sub_cfg2 = False
    for e in edges1:
        if e not in edges2:
            cfg1_sub_cfg2 = False
    #first check if graph 2 is a subgraph of graph 1:
    for n2 in nodes2:
        if n2 not in nodes1:
            cfg2_sub_cfg1 = False
    for e in edges2:
        if e not in edges1:
            cfg2_sub_cfg1 = False
    return cfg1_sub_cfg2 & cfg2_sub_cfg1
     
def max_subgraph(cfg1, cfg2) -> list:
    '''takes in two CFG objects, returns the nid's of the maximum subgraph'''
    nodes1, edges1 = CFG_to_hash_graph(cfg1)
    nodes2, edges2 = CFG_to_hash_graph(cfg2)
    #first check if graph 1 is a subgraph of graph 2:
    node_map = {}; edge_map = {}; 
    for n in nodes1:
        node_map[n] = node_map[n] + 1 if node_map.get(n) else 1
    for n in nodes2:
        node_map[n] = node_map[n] + 1 if node_map.get(n) else 1
    for e in edges1:
        edge_map[e] = edge_map[e] + 1 if edge_map.get(e) else 1
    for e in edges2:
        edge_map[e] = edge_map[e] + 1 if edge_map.get(e) else 1
    common_node_set = []; common_edge_set = []
    for node, count in node_map.items():
        if count == 2: #assuming a node will appear at most twice (once in each graph)
            common_node_set.append(node)
    for edge, count in edge_map.items():
        if count == 2: #assuming an edge will appear at most twice (once in each graph)
            common_edge_set.append(edge)
    if len(common_node_set) < 1: return None 

    #dfs-sweep should be performed to find the largest connected component.
    dfs_stack = []; max_subgraph_node = None; max_subgraph_size = 0; subgraph_nid_set = []
    dfs_stack.append(common_node_set[0])
    visited = {}
    for node in common_node_set: #this for-loop is probably close to O(n^5) if not worse :
        curr_subgraph_size = 0
        curr_path = []
        while(len(dfs_stack)) > 0:
            curr_node = dfs_stack.pop(0)
            if not visited.get(curr_node):
                curr_path.append(curr_node)
                visited[curr_node] = 1
                #append neighbors and increment curr_subgraph_size
                for edge in common_edge_set:
                    if curr_node == edge[0] and not visited.get(edge[1]):
                        dfs_stack.append(edge[1])
                curr_subgraph_size += 1
        if max_subgraph_size < curr_subgraph_size:
            max_subgraph_node = node
            max_subgraph_size = curr_subgraph_size
            subgraph_nid_set = curr_path

    #return the nid's of the graph
    return_path = []
    for node in subgraph_nid_set:
        for cfg_node in cfg1.opcodes: 
        # print(type(cfg1.opcodes[node]))
            if node == hash(node_to_string(cfg1.opcodes[cfg_node])):
                return_path.append(cfg_node)
    return return_path



            
class CFG:
                
        

    def __init__(self, codeobject, target_offset=-1, source_code=None):
        def lstadd(hmap, key, val):
            if key not in hmap:
                hmap[key] = [val]
            else:
                hmap[key].append(val)
        self.name = codeobject.co_name
        self.source_code = source_code
        enter = CFGNode(dis.Instruction('NOP', opcode=dis.opmap['NOP'], arg=0, argval=0, argrepr=0, offset=0,starts_line=0, is_jump_target=False), 0)
        last = enter
        self.target_offset = target_offset
        self.jump_to = {}
        self.opcodes = {}
        return_nodes = []
        #for i,ins in enumerate(dis.get_instructions(myfn)):
        curr_line_num = None
        for i,ins in enumerate(dis.get_instructions(codeobject)):
            # byte => offset
            byte = ins.offset #i * 2
            curr_line_num = ins.starts_line if ins.starts_line != None else curr_line_num
            node = CFGNode(ins, byte, line_no = curr_line_num)
            node.nid = byte
            self.opcodes[byte] = node
            #print(i,ins)
            # print(ins, ins.offset, type(ins.offset))
            # Target offset is for only marking a certain instruction
            if ins.offset == target_offset:
                node.isTarget = True
            if ins.opname in ops_list:
                last.add_child(node)
                last = node
                if ins.opname == 'RETURN_VALUE':
                    return_nodes.append(node)
            elif ins.opname in ops_jumps:
                lstadd(self.jump_to, ins.arg, node)
                node.props['jmp'] = True
                last.add_child(node)
                last = node
            elif ins.opname in ops_relative_jumps:
                node.props['jmp'] = True
                lstadd(self.jump_to, (i+1)*2 + ins.arg, node)
                last.add_child(node)
                last = node
            elif ins.opname == 'SETUP_LOOP':
                last.add_child(node)
                last = node
            elif ins.opname in ops_setups:
                lstadd(self.jump_to, (i+1)*2 + ins.arg, node)
                node.props['jmp'] = True
                last.add_child(node)
                last = node
            else:
                print( ins )
                print( ins.opname )
                assert False
        # print ("last inst:", byte)
        if target_offset == -1: 
            self.target_offset = last.bid
            last.isTarget = True
        for byte in self.opcodes:
            if  byte in self.jump_to:
                node = self.opcodes[byte]
                assert node.i.is_jump_target
                for b in self.jump_to[byte]:
                    b.add_child(node)
                    
        # Empty return nodes   
        for rnodes in return_nodes:
            rnodes.children.clear()
        # Setup basic blocks
        block_nodes = []
        for nid, cnode in self.opcodes.items():
            #if len(cnode.children) == 1 and len(cnode.parent) == 1:
            if len(cnode.children) == 1: # Only one child node
                if len(cnode.children[0].parent) == 1: # only one parent node of child
                    cnode.block = True            
                    block_nodes.append(cnode)
                    # print (nid)
            # if cnode.i.opname in ops_setups and cnode not in block_nodes:
            #     cnode.block = True            
            #     block_nodes.append(cnode)
        
        for cnode in block_nodes:
            first_node = cnode
            last_node = cnode
            
            traversed_node = []
            
            if cnode.first_node_id == -1:
                while True:
                    if len(last_node.parent) > 0 and first_node.parent[0].block == True:
                        first_node = first_node.parent[0]
                        traversed_node.append(first_node)
                    else:
                        break
                cnode.first_node_id = first_node.nid;
            if cnode.last_node_id == -1:
                while True:
                    if len(last_node.children) > 0 and last_node.children[0].block == True:
                        last_node = last_node.children[0]
                        traversed_node.append(last_node)
                    else:
                        break
                cnode.last_node_id = last_node.nid;
            for node in traversed_node:
                node.first_node_id = first_node.nid;
                node.last_node_id = last_node.nid;
            
            
        node_to_del = []
        for cnode in block_nodes:
            if cnode.children[0].block == True:
                # merge                
                merging_to = self.opcodes[cnode.first_node_id];
                merged_children = self.opcodes[cnode.last_node_id].children;
                node_to_del.append(cnode.children[0].nid);
                merging_to.ins.append(cnode.children[0])
                merging_to.children = merged_children
                
        for node in node_to_del:
            del self.opcodes[node];
        
        
        node_to_del.clear()
        for nid, cnode in self.opcodes.items():
            if len(cnode.children) == 1:
                if len(cnode.children[0].parent) == 1 and cnode.children[0].i.opname not in ops_setups:
                    merging_to = self.opcodes[nid];
                    merged_children = self.opcodes[cnode.children[0].nid].children;
                    node_to_del.append(cnode.children[0].nid);
                    merging_to.ins.append(cnode.children[0])
                    for n in cnode.children[0].ins:
                        merging_to.ins.append( n )
                    merging_to.children = merged_children

        
        for node in node_to_del:
            del self.opcodes[node];
    
    def get_adjacent_blocks(self, nid):
        if nid not in self.opcodes: return None
    def get_source_line(self, line_no):
        if self.source_code == None: return None
        lines = self.source_code.split('\n')
        if len(lines) < line_no: return None
        return lines[line_no -1]
    
    def to_graph(self, benign = False):
        # create graph
        G = pygraphviz.AGraph(directed=True)
        # Iterade cnodes
        for nid, cnode in self.opcodes.items():
            G.add_node(cnode.bid)
            n = G.get_node(cnode.bid)
            s = ""
            if len(cnode.ins) > 0:
                for node in cnode.ins:
                    ## Mark NODE that is troublesome
                    if not benign and cnode.isTarget and node.nid == self.target_offset:
                        t = "%d: %s <-- \n" % (node.nid, node.i.opname)
                    else:
                        t = "%d: %s\n" % (node.nid, node.i.opname)
                    s = s + t
                ## Mark node if starting node
                if not benign and cnode.isTarget and nid == self.target_offset:
                    n.attr['label'] = "%d: %s  <--\n%s" % (nid, cnode.i.opname, s)
                else:
                    n.attr['label'] = "%d: %s \n%s" % (nid, cnode.i.opname, s)
            else:
                ## Mark node if only starting node
                if not benign and cnode.isTarget and nid == self.target_offset:
                    n.attr['label'] = "%d: %s  <--" % (nid, cnode.i.opname)
                else:
                    n.attr['label'] = "%d: %s" % (nid, cnode.i.opname)        
                # n.attr['label'] = "%d: %s" % (nid, cnode.i.opname)
            n.attr["shape"] = "box"
            if not benign and cnode.isTarget:
                n.attr['style'] = "filled"
                n.attr['color'] = "lightcoral"
            for cn in cnode.children:
                G.add_edge(cnode.bid, cn.bid)
        return G

    def dfs(self, func):
        visited = {}
        stack = []
        stack.append(0) #assumes there will always be a block wit nid 0
        while len(stack) > 0:
            cnode_nid = stack.pop(-1)
            visited[cnode_nid] = 1
            cnode = self.opcodes[cnode_nid]
            func(cnode)
            for node in cnode.children:
                if visited.get(node.nid) == None:
                    stack.append(node.nid)
        return

    def bfs(self, func):
        visited = {}
        queue = []
        queue.append(0) #assumes there will always be a block wit nid 0
        while len(queue) > 0:
            cnode_nid = queue.pop(0)
            visited[cnode_nid] = 1
            cnode = self.opcodes[cnode_nid]
            func(cnode_nid)
            for node in cnode.children:
                if visited.get(node.nid) == None:
                    queue.append(node.nid)
        return
    
    # def is_subgraph(self, cfg2):
    


def analyze_cfg(cfg):
    v = cfg
    print('Function name:', v.name)
    print('Basic blocks:', len(v.opcodes), [v.opcodes[node].i.opname for node in v.opcodes])
    # Get edges
    for nid, cnode in v.opcodes.items():
        for cn in cnode.children:
            print(cnode.i.opname,len(cnode.ins), cnode.bid,'-->>>' ,cn.bid, len(cn.ins),cn.i.opname)
            print(
                cnode.ins[0].i.opname if len(cnode.ins) else "[]", 
                cnode.ins[-1].i.opname if len(cnode.ins) else "[]", 
                '---->>>',
                cn.ins[0].i.opname if len(cn.ins) else "[]", 
                cn.ins[-1].i.opname if len(cn.ins) else "[]", 
                cn.ins[-1].block if len(cn.ins) else "[]"
            )
            # print(cnode.i.opcode, '')
    v.to_graph().draw('output/cfg/'+ v.name + '.out.png', prog='dot')
    

def draw_graph(cfg, f_name=None):
    if f_name == None:
        filename = 'output/cfg/'+ cfg.name + '.out.png'
    else:
        filename = 'output/cfg/curr_'+ cfg.name + '.out.png'
    for i in range(1000):
        if exists(filename):
            filename = 'output/cfg/'+ cfg.name+str(i) + '.out.png'
    cfg.to_graph().draw(filename, prog='dot')

def draw_graph_from_bc(bc, opt=None):
    v1 = CFG(bc, -1) # target
    draw_graph(v1, opt)

def get_edges_of_node(cnode):
    # Gets the outgoing edges of the current basic block
    edges_v1 = []
    for cn in cnode.children:
            edges_v1.append("{}:{}".format(cnode.i.opname if len(cnode.ins) == 0 else cnode.ins[-1].i.opname, cn.i.opname))
    edges_v1 = [replace_common_strings(i) for i in edges_v1]
    return edges_v1


