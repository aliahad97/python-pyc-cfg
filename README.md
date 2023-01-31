# Overview

This repo creates a small application to generate CFG from python pyc files. The current version is tested on Python 3.8. 

# Requirements

- Python version 3.8
- Graphviz

# Helper scripts

## Compiling Python scripts to pyc files

Run the following command to compile `initial.py` in `test` directory:

```
source compile.sh
```

Output:
- The output of the script is a pyc file and the human readable format of the disassembly. 

# Usage:

Run below command to generate CFG.

```
source generate_cfg.sh <pyc_file_name> <function name>
```

Output:
The above command will generate a CFG in the `output/cfg` directory. 

Sample command:

```
source generate_cfg.sh test/initial.pyc
```