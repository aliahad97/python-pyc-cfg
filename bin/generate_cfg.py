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
    parser = argparse.ArgumentParser(description= "Specify a filename, function (optional), output directory")
    parser.add_argument('filename', type = str, help=".py or .pyc file", default = 'test/initial.pyc')
    parser.add_argument('--function', type = str, default = '<module>', help = "Function Name")
    parser.add_argument('--output_dir', type = str, default = 'output/cfg', help = "Output Directory")
    parser.add_argument('--output_filename', type = str, default = '', help = "Output file")
    args = parser.parse_args()
    try:
        target_bc = args.filename
        if args.filename.endswith('.py'):
            target_bc = py_compile.compile(args.filename)
        if not os.path.isfile(target_bc) and not target_bc.endswith('.pyc'): raise
    except:
        print('Invalid file or file does not exist')
        exit()
    try:
        target_func = args.function
    except:
        target_func = '<module>'
    co = readpyc.read_file_get_object(target_bc)
    co_fn = manipulate_pyc.get_co_fn(co, target_func)
    output_filename = args.output_filename if args.output_filename != '' else None
    cfg.draw_graph_from_bc(co_fn, output_filename)
    test_cfg = cfg.CFG(co_fn, -1)

if __name__ == '__main__':
    # main(sys.argv[1:])
    main(None)