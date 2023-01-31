import sys, types, dis, traceback, copy
import readpyc
import opcode, re
# try:
#     input_f = sys.argv[1]
#     output_f = sys.argv[2]
# except:
#     print("no input and output file")

jmp_op_boolean_chain = [
    111, # Boolean expression "AND" JUMP_IF_FALSE_OR_POP
    112, # Boolean expression "OR" JUMP_IF_TRUE_OR_POP
    # 113, # Absolute jump - Can skip since no boolean
    114, # Conditional POP_JUMP_IF_FALSE
    115, # Conditional POP_JUMP_IF_TRUE
    # 121, # No idea where this is used - jabs_op('JUMP_IF_NOT_EXC_MATCH', 121)
]
terminate_boolean_chain = [
    90, # name_op('STORE_NAME', 90) 
    95, # name_op('STORE_ATTR', 95) # Index in name list
    97, # name_op('STORE_GLOBAL', 97)     # ""
    125, # def_op('STORE_FAST', 125)       # Local variable number
    83, # def_op('RETURN_VALUE', 83)
    86, # def_op('YIELD_VALUE', 86)
]
 # a and b and/or c # goes on

'''
jrel_op('JUMP_FORWARD', 110)    # Number of bytes to skip
jabs_op('JUMP_IF_FALSE_OR_POP', 111) # Target byte offset from beginning of code
jabs_op('JUMP_IF_TRUE_OR_POP', 112)  # ""
jabs_op('JUMP_ABSOLUTE', 113)        # ""
jabs_op('POP_JUMP_IF_FALSE', 114)    # ""
jabs_op('POP_JUMP_IF_TRUE', 115)     # ""

Finer-grained -> less generic
More generic -> 

whitelist 
regex `*`
'''

def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll-1



def print_co_info(co):
    print(co)
    print("argcount:", co.co_argcount)
    if sys.version_info[1] > 7:
        print("co_posonlyargcount:", co.co_posonlyargcount) #Add this in Python 3.8+
    print("co_kwonlyargcount:", co.co_kwonlyargcount)  #Add this in Python3
    print("co_nlocals:",co.co_nlocals)
    print("co_stacksize:",co.co_stacksize)
    print("co_flags:",co.co_flags)
    print("code:",co.co_code)
    print("co_consts:",co.co_consts)
    print("co_names:",co.co_names)
    print("co_varnames:",co.co_varnames)
    print("co_filename:",co.co_filename)
    print("co_name:",co.co_name)
    print("co_firstlineno:",co.co_firstlineno)
    print("co_lnotab:",co.co_lnotab)   # In general, You should adjust this
    print("co_freevars:",co.co_freevars)
    print("co_cellvars:",co.co_cellvars)
    print("Dis:")
    print("="*5)
    dis.dis(co.co_code)
    print("="*5)
    # )

# Get the Code object with function name 'fn_name' in 'co'
def get_co_fn(co, fn_name):
    # print(co.co_name, fn_name)
    # print(type(co.co_name), type(fn_name))
    # print(len(co.co_name), len(fn_name))
    if fn_name == co.co_name: 
        return co
    for const in co.co_consts:
        if isinstance(const, types.CodeType):
            temp = get_co_fn(const, fn_name)
            if temp is not None: return temp
    return None
