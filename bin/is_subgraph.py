import sys,dis, os
sys.path.insert(0, './lib')
import readpyc, cfg, manipulate_pyc
import argparse
import py_compile

from cfg import *

func_test= 'check'
test_1 = 'test/original.pyc'
test_2 = 'test/initial.pyc'

def main(argv):
    parser = argparse.ArgumentParser(description= "Specify 2 filenames")
    parser.add_argument('filename_1', type = str, help=".py or .pyc file", default = 'test/initial.pyc')
    parser.add_argument('filename_2', type = str, help = ".py or .pyc file", default = 'test/initial.pyc')
    args = parser.parse_args()
    try:
        target_bc_1 = args.filename_1
        if args.filename_1.endswith('.py'):
            target_bc_1 = py_compile.compile(args.filename_1)
        if not os.path.isfile(target_bc_1) and not target_bc_1.endswith('.pyc'): raise
    except:
        print('Invalid file_1 or file_1 does not exist')
        exit()
    try:
        target_bc_2 = args.filename_2
        if args.filename_2.endswith('.py'):
            target_bc_2 = py_compile.compile(args.filename_2)
        if not os.path.isfile(target_bc_2) and not target_bc_2.endswith('.pyc'): raise
    except:
        print('Invalid file_2 or file_2 does not exist')
        exit()

   
    target_func = '<module>'
    co_1 = readpyc.read_file_get_object(target_bc_1)
    co_1_fn = manipulate_pyc.get_co_fn(co_1, target_func)
    test_cfg_1 = cfg.CFG(co_1_fn, -1)
    co_2 = readpyc.read_file_get_object(target_bc_2)
    co_2_fn = manipulate_pyc.get_co_fn(co_2, target_func)
    test_cfg_2 = cfg.CFG(co_2_fn, -1)

    print(f" All graph 1's DFS paths: {cfg.get_all_dfs_paths(test_cfg_1, nid = False)}")

    print(f" All graph 2's DFS paths: {cfg.get_all_dfs_paths(test_cfg_2, nid = False)}")

    print(f" Testing max_subgraph_dfs function:\n Output: {cfg.max_subgraph_dfs(test_cfg_1, test_cfg_2)}")

if __name__ == '__main__':
    # main(sys.argv[1:])
    main(None)




