import sys,dis, os
sys.path.insert(0, './lib')
import readpyc, cfg, manipulate_pyc

from cfg import *

func_test= 'check'
test_1 = 'test/original.pyc'
test_2 = 'test/initial.pyc'

def main(argv):
    #print (argv[0])
    try:
        target_bc = sys.argv[1]
        if not os.path.isfile(target_bc) and not target_bc.endswith('.pyc'): raise
    except:
        print('Invalid file or file does not exist')
        exit()
    try:
        target_func = sys.argv[2]
    except:
        target_func = '<module>'
    co = readpyc.read_file_get_object(target_bc)
    co_fn = manipulate_pyc.get_co_fn(co, target_func)
    cfg.draw_graph_from_bc(co_fn, None)
    

if __name__ == '__main__':
    # main(sys.argv[1:])
    main(None)