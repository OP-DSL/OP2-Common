#!/usr/bin/env python

"""
OP2 source code transformation tool

This tool parses the user's original source code to produce
target-specific code to execute the user's kernel functions.

This prototype is written in Python and is directly based on the
parsing and code generation of the matlab source code transformation code

usage: ./op2.py 'file1','file2',...

This takes as input

file1.cpp, file2.cpp, ...

and produces as output modified versions

file1_op.cpp, file2_op.cpp, ...

then calls a number of target-specific code generators
to produce individual kernel files of the form

xxx_kernel.cpp  -- for OpenMP x86 execution
xxx_kernel.cu   -- for CUDA execution

plus a master kernel file of the form

file1_kernels.cpp  -- for OpenMP x86 execution
file1_kernels.cu   -- for CUDA execution
"""

import sys
import re
import datetime
import os

# Import MPI+SEQ and MPI+autovectorised SEQ
from op2_gen_seq import op2_gen_seq
from op2_gen_mpi_vec import op2_gen_mpi_vec

# import OpenMP and CUDA code generation functions
from op2_gen_openmp_simple import op2_gen_openmp_simple

from op2_gen_openacc import op2_gen_openacc

from op2_gen_cuda import op2_gen_cuda
from op2_gen_cuda_simple import op2_gen_cuda_simple
from op2_gen_cuda_simple_hyb import op2_gen_cuda_simple_hyb

# from http://stackoverflow.com/a/241506/396967
def comment_remover(text):
    """Remove comments from text"""

    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return ""
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)


def op_parse_calls(text):
    """Parsing for op_init/op_exit/op_partition/op_hdf5 calls"""

    # remove comments just for this call
    text = comment_remover(text)

    inits = len(re.findall('op_init', text))
    exits = len(re.findall('op_exit', text))
    parts = len(re.findall('op_partition', text))
    hdf5s = len(re.findall('hdf5', text))

    return (inits, exits, parts, hdf5s)


def op_decl_set_parse(text):
    """Parsing for op_decl_set calls"""

    sets = []
    for m in re.finditer('op_decl_set\((.*)\)', text):
        args = m.group(1).split(',')

        # check for syntax errors
        if len(args) != 2:
            print 'Error in op_decl_set : must have three arguments'
            return

        sets.append({
            'name': args[1].strip()
            })
    for m in re.finditer('op_decl_set_hdf5\((.*)\)', text):
        args = m.group(1).split(',')

        # check for syntax errors
        if len(args) != 2:
            print 'Error in op_decl_set : must have three arguments'
            return

        sets.append({
            'name': args[1].strip()[1:-1]
            })

    return sets


def op_decl_const_parse(text):
    """Parsing for op_decl_const calls"""

    consts = []
    for m in re.finditer('op_decl_const\((.*)\)', text):
        args = m.group(1).split(',')

        # check for syntax errors
        if len(args) != 3:
            print 'Error in op_decl_const : must have three arguments'
            return

        consts.append({
            'loc': m.start(),
            'dim': args[0].strip(),
            'type': args[1].strip(),
            'name': args[2].strip(),
            'name2': args[2].strip()
            })

    return consts


def arg_parse(text, j):
    """Parsing arguments in op_par_loop to find the correct closing brace"""

    depth = 0
    loc2 = j
    while 1:
        if text[loc2] == '(':
            depth = depth + 1

        elif text[loc2] == ')':
            depth = depth - 1
            if depth == 0:
                return loc2
        loc2 = loc2 + 1


def get_arg_dat(arg_string, j):
    loc = arg_parse(arg_string, j + 1)
    dat_args_string = arg_string[arg_string.find('(', j) + 1:loc]

    # remove comments
    dat_args_string = comment_remover(dat_args_string)

    # check for syntax errors
    if len(dat_args_string.split(',')) != 6:
        print 'Error parsing op_arg_dat(%s): must have six arguments' \
              % dat_args_string
        return

    # split the dat_args_string into  6 and create a struct with the elements
    # and type as op_arg_dat
    temp_dat = {'type': 'op_arg_dat',
                'dat': dat_args_string.split(',')[0].strip(),
                'idx': dat_args_string.split(',')[1].strip(),
                'map': dat_args_string.split(',')[2].strip(),
                'dim': dat_args_string.split(',')[3].strip(),
                'typ': dat_args_string.split(',')[4].strip(),
                'acc': dat_args_string.split(',')[5].strip()}

    return temp_dat


def get_arg_gbl(arg_string, k):
    loc = arg_parse(arg_string, k + 1)
    gbl_args_string = arg_string[arg_string.find('(', k) + 1:loc]

    # remove comments
    gbl_args_string = comment_remover(gbl_args_string)

    # check for syntax errors
    if len(gbl_args_string.split(',')) != 4:
        print 'Error parsing op_arg_gbl(%s): must have four arguments' \
              % gbl_args_string
        return

    # split the gbl_args_string into  4 and create a struct with the elements
    # and type as op_arg_gbl
    temp_gbl = {'type': 'op_arg_gbl',
                'data': gbl_args_string.split(',')[0].strip(),
                'dim': gbl_args_string.split(',')[1].strip(),
                'typ': gbl_args_string.split(',')[2].strip(),
                'acc': gbl_args_string.split(',')[3].strip()}

    return temp_gbl

def append_init_soa(text):
  text = re.sub('\\bop_init\\b\\s*\((.*)\)','op_init_soa(\\1,1)', text)
  text = re.sub('\\bop_mpi_init\\b\\s*\((.*)\)','op_mpi_init_soa(\\1,1)', text)
  return text

def op_par_loop_parse(text):
    """Parsing for op_par_loop calls"""

    loop_args = []

    search = "op_par_loop"
    i = text.find(search)
    while i > -1:
        arg_string = text[text.find('(', i) + 1:text.find(';', i + 11)]

        # parse arguments in par loop
        temp_args = []
        num_args = 0

        # parse each op_arg_dat
        search2 = "op_arg_dat"
        search3 = "op_arg_gbl"
        j = arg_string.find(search2)
        k = arg_string.find(search3)

        while j > -1 or k > -1:
            if k <= -1:
                temp_dat = get_arg_dat(arg_string, j)
                # append this struct to a temporary list/array
                temp_args.append(temp_dat)
                num_args = num_args + 1
                j = arg_string.find(search2, j + 11)

            elif j <= -1:
                temp_gbl = get_arg_gbl(arg_string, k)
                # append this struct to a temporary list/array
                temp_args.append(temp_gbl)
                num_args = num_args + 1
                k = arg_string.find(search3, k + 11)

            elif j < k:
                temp_dat = get_arg_dat(arg_string, j)
                # append this struct to a temporary list/array
                temp_args.append(temp_dat)
                num_args = num_args + 1
                j = arg_string.find(search2, j + 11)

            else:
                temp_gbl = get_arg_gbl(arg_string, k)
                # append this struct to a temporary list/array
                temp_args.append(temp_gbl)
                num_args = num_args + 1
                k = arg_string.find(search3, k + 11)

        temp = {'loc': i,
                'name1': arg_string.split(',')[0].strip(),
                'name2': arg_string.split(',')[1].strip(),
                'set': arg_string.split(',')[2].strip(),
                'args': temp_args,
                'nargs': num_args}

        loop_args.append(temp)
        i = text.find(search, i + 10)
    print '\n\n'
    return (loop_args)


def main():

    # declare constants

    ninit = 0
    nexit = 0
    npart = 0
    nhdf5 = 0
    nconsts = 0
    nkernels = 0
    consts = []
    kernels = []
    sets = []
    kernels_in_files = []

    OP_ID = 1
    OP_GBL = 2
    OP_MAP = 3

    OP_READ = 1
    OP_WRITE = 2
    OP_RW = 3
    OP_INC = 4
    OP_MAX = 5
    OP_MIN = 6

    auto_soa=os.getenv('OP_AUTO_SOA','0')

    OP_accs_labels = ['OP_READ', 'OP_WRITE', 'OP_RW', 'OP_INC',
                      'OP_MAX', 'OP_MIN']

    #  loop over all input source files

    kernels_in_files = [[] for _ in range(len(sys.argv) - 1)]
    for a in range(1, len(sys.argv)):
        print 'processing file ' + str(a) + ' of ' + str(len(sys.argv) - 1) + \
              ' ' + str(sys.argv[a])

        src_file = str(sys.argv[a])
        f = open(src_file, 'r')
        text = f.read()
        any_soa = 0

        # check for op_init/op_exit/op_partition/op_hdf5 calls

        inits, exits, parts, hdf5s = op_parse_calls(text)

        if inits + exits + parts + hdf5s > 0:
            print ' '
        if inits > 0:
            print'contains op_init call'
            if auto_soa<>'0':
              text = append_init_soa(text)
        if exits > 0:
            print'contains op_exit call'
        if parts > 0:
            print'contains op_partition call'
        if hdf5s > 0:
            print'contains op_hdf5 calls'

        ninit = ninit + inits
        nexit = nexit + exits
        npart = npart + parts
        nhdf5 = nhdf5 + hdf5s

        # parse and process constants

        const_args = op_decl_const_parse(text)
        set_list = op_decl_set_parse(text)
        for i in range(0,len(set_list)):
          sets.append(set_list[i])

        # cleanup '&' symbols from name and convert dim to integer
        for i in range(0, len(const_args)):
            if const_args[i]['name'][0] == '&':
                const_args[i]['name'] = const_args[i]['name'][1:]
                const_args[i]['dim'] = int(const_args[i]['dim'])

        # check for repeats
        nconsts = 0
        for i in range(0, len(const_args)):
            repeat = 0
            name = const_args[i]['name']
            for c in range(0, nconsts):
                if const_args[i]['name'] == consts[c]['name']:
                    repeat = 1
                    if const_args[i]['type'] != consts[c]['type']:
                        print 'type mismatch in repeated op_decl_const'
                    if const_args[i]['dim'] != consts[c]['dim']:
                        print 'size mismatch in repeated op_decl_const'

            if repeat > 0:
                print 'repeated global constant ' + const_args[i]['name']
            else:
                print '\nglobal constant (' + const_args[i]['name'].strip() \
                      + ') of size ' + str(const_args[i]['dim'])

            # store away in master list
            if repeat == 0:
                nconsts = nconsts + 1
                temp = {'dim': const_args[i]['dim'],
                        'type': const_args[i]['type'].strip(),
                        'name': const_args[i]['name'].strip()}
                consts.append(temp)

        # parse and process op_par_loop calls

        loop_args = op_par_loop_parse(text)
        for i in range(0, len(loop_args)):
            name = loop_args[i]['name1']
            nargs = loop_args[i]['nargs']
            print '\nprocessing kernel ' + name + ' with ' + str(nargs) + ' arguments',

            # process arguments

            var = [''] * nargs
            idxs = [0] * nargs
            dims = [''] * nargs
            maps = [0] * nargs
            mapnames = ['']*nargs
            typs = [''] * nargs
            accs = [0] * nargs
            soaflags = [0] * nargs

            for m in range(0, nargs):
                arg_type = loop_args[i]['args'][m]['type']
                args = loop_args[i]['args'][m]

                if arg_type.strip() == 'op_arg_dat':
                    var[m] = args['dat']
                    idxs[m] = args['idx']

                    if str(args['map']).strip() == 'OP_ID':
                        maps[m] = OP_ID
                        if int(idxs[m]) != -1:
                            print 'invalid index for argument' + str(m)
                    else:
                        maps[m] = OP_MAP
                        mapnames[m] = str(args['map']).strip()

                    dims[m] = args['dim']
                    soa_loc = args['typ'].find(':soa')
                    if ((auto_soa=='1') and (((not dims[m].isdigit()) or int(dims[m])>1)) and (soa_loc < 0)):
                        soa_loc = len(args['typ'])-1

                    if soa_loc > 0:
                        soaflags[m] = 1
                        any_soa = 1
                        typs[m] = args['typ'][1:soa_loc]
                    else:
                        typs[m] = args['typ'][1:-1]


                    l = -1
                    for l in range(0, len(OP_accs_labels)):
                        if args['acc'].strip() == OP_accs_labels[l].strip():
                            break

                    if l == -1:
                        print 'unknown access type for argument ' + str(m)
                    else:
                        accs[m] = l + 1

                if arg_type.strip() == 'op_arg_gbl':
                    maps[m] = OP_GBL
                    var[m] = args['data']
                    dims[m] = args['dim']
                    typs[m] = args['typ'][1:-1]

                    l = -1
                    for l in range(0, len(OP_accs_labels)):
                        if args['acc'].strip() == OP_accs_labels[l].strip():
                            break

                    if l == -1:
                        print 'unknown access type for argument ' + str(m)
                    else:
                        accs[m] = l + 1

                if (maps[m] == OP_GBL) and (accs[m] == OP_WRITE or accs[m] == OP_RW):
                    print 'invalid access type for argument ' + str(m)

                if (maps[m] != OP_GBL) and (accs[m] == OP_MIN or accs[m] == OP_MAX):
                    print 'invalid access type for argument ' + str(m)


            print ' '

            # identify indirect datasets

            ninds = 0
            inds = [0] * nargs
            invinds = [0] * nargs
            indtyps = [''] * nargs
            inddims = [''] * nargs
            indaccs = [0] * nargs
            invmapinds = [0]*nargs
            mapinds = [0]*nargs

            j = [i for i, x in enumerate(maps) if x == OP_MAP]

            while len(j) > 0:

                indtyps[ninds] = typs[j[0]]
                inddims[ninds] = dims[j[0]]
                indaccs[ninds] = accs[j[0]]
                invinds[ninds] = j[0]  # inverse mapping
                ninds = ninds + 1
                for i in range(0, len(j)):
                    if var[j[0]] == var[j[i]] and typs[j[0]] == typs[j[i]] \
                            and accs[j[0]] == accs[j[i]] and mapnames[j[0]] == mapnames[j[i]]:  # same variable
                        inds[j[i]] = ninds

                k = []
                for i in range(0, len(j)):
                    if not (var[j[0]] == var[j[i]] and typs[j[0]] == typs[j[i]]
                            and accs[j[0]] == accs[j[i]] and mapnames[j[0]] == mapnames[j[i]]):  # same variable
                        k = k + [j[i]]
                j = k

            if ninds > 0:
              invmapinds = invinds[:]
              for i in range(0,ninds):
                for j in range(0,i):
                  if (mapnames[invinds[i]] == mapnames[invinds[j]]):
                    invmapinds[i] = invmapinds[j]

              for i in range(0,nargs):
                mapinds[i] = i
                for j in range(0,i):
                  if (maps[i] == OP_MAP) and (mapnames[i] == mapnames[j]) and (idxs[i] == idxs[j]):
                    mapinds[i] = mapinds[j]

            # check for repeats

            repeat = False
            rep1 = False
            rep2 = False
            which_file = -1
            for nk in range(0, nkernels):
                rep1 = kernels[nk]['name'] == name and \
                    kernels[nk]['nargs'] == nargs and \
                    kernels[nk]['ninds'] == ninds
                if rep1:
                    rep2 = True
                    for arg in range(0, nargs):
                        rep2 = rep2 and \
                            kernels[nk]['dims'][arg] == dims[arg] and \
                            kernels[nk]['maps'][arg] == maps[arg] and \
                            kernels[nk]['typs'][arg] == typs[arg] and \
                            kernels[nk]['accs'][arg] == accs[arg] and \
                            kernels[nk]['idxs'][arg] == idxs[arg] and \
                            kernels[nk]['soaflags'][arg] == soaflags[arg] and \
                            kernels[nk]['inds'][arg] == inds[arg]

                    for arg in range(0, ninds):
                        rep2 = rep2 and \
                            kernels[nk]['inddims'][arg] == inddims[arg] and \
                            kernels[nk]['indaccs'][arg] == indaccs[arg] and \
                            kernels[nk]['indtyps'][arg] == indtyps[arg] and \
                            kernels[nk]['invinds'][arg] == invinds[arg]
                    if rep2:
                        print 'repeated kernel with compatible arguments: ' + \
                              kernels[nk]['name'],
                        repeat = True
                        which_file = nk
                    else:
                        print 'repeated kernel with incompatible arguments: ERROR'
                        break

            # output various diagnostics

            if not repeat:
                print '  local constants:',
                for arg in range(0, nargs):
                    if maps[arg] == OP_GBL and accs[arg] == OP_READ:
                        print str(arg),
                print '\n  global reductions:',
                for arg in range(0, nargs):
                    if maps[arg] == OP_GBL and accs[arg] != OP_READ:
                        print str(arg),
                print '\n  direct arguments:',
                for arg in range(0, nargs):
                    if maps[arg] == OP_ID:
                        print str(arg),
                print '\n  indirect arguments:',
                for arg in range(0, nargs):
                    if maps[arg] == OP_MAP:
                        print str(arg),
                if ninds > 0:
                    print '\n  number of indirect datasets: ' + str(ninds),

                print '\n'

            # store away in master list

            if not repeat:
                nkernels = nkernels + 1
                temp = {'name': name,
                        'nargs': nargs,
                        'dims': dims,
                        'maps': maps,
                        'var': var,
                        'typs': typs,
                        'accs': accs,
                        'idxs': idxs,
                        'inds': inds,
                        'soaflags': soaflags,

                        'ninds': ninds,
                        'inddims': inddims,
                        'indaccs': indaccs,
                        'indtyps': indtyps,
                        'invinds': invinds,
                        'mapnames' : mapnames,
                        'mapinds': mapinds,
                        'invmapinds' : invmapinds}
                kernels.append(temp)
                (kernels_in_files[a - 1]).append(nkernels - 1)
            else:
                append = 1
                for in_file in range(0, len(kernels_in_files[a - 1])):
                    if kernels_in_files[a - 1][in_file] == which_file:
                        append = 0
                if append == 1:
                    (kernels_in_files[a - 1]).append(which_file)

        # output new source file

        fid = open(src_file.split('.')[0] + '_op.cpp', 'w')
        date = datetime.datetime.now()
        #fid.write('//\n// auto-generated by op2.py on ' +
        #          date.strftime("%Y-%m-%d %H:%M") + '\n//\n\n')
        fid.write('//\n// auto-generated by op2.py\n//\n\n')

        loc_old = 0

        # read original file and locate header location
        header_len = 11
        loc_header = [text.find("op_seq.h")]
        if loc_header[0] == -1:
          header_len = 13
          loc_header = [text.find("op_lib_cpp.h")]

        # get locations of all op_decl_consts
        n_consts = len(const_args)
        loc_consts = [0] * n_consts
        for n in range(0, n_consts):
            loc_consts[n] = const_args[n]['loc']

        # get locations of all op_par_loops
        n_loops = len(loop_args)
        loc_loops = [0] * n_loops
        for n in range(0, n_loops):
            loc_loops[n] = loop_args[n]['loc']

        locs = sorted(loc_header + loc_consts + loc_loops)

        # process header, loops and constants
        for loc in range(0, len(locs)):
            if locs[loc] != -1:
                fid.write(text[loc_old:locs[loc] - 1])
                loc_old = locs[loc] - 1

            indent = ''
            ind = 0
            while 1:
                if text[locs[loc] - ind] == '\n':
                    break
                indent = indent + ' '
                ind = ind + 1

            if (locs[loc] in loc_header) and (locs[loc] != -1):
                fid.write(' "op_lib_cpp.h"\n\n')
                fid.write('//\n// op_par_loop declarations\n//\n')
                fid.write('#ifdef OPENACC\n#ifdef __cplusplus\nextern "C" {\n#endif\n#endif\n')
                for k_iter in range(0, len(kernels_in_files[a - 1])):
                    k = kernels_in_files[a - 1][k_iter]
                    line = '\nvoid op_par_loop_' + \
                        kernels[k]['name'] + '(char const *, op_set,\n'
                    for n in range(1, kernels[k]['nargs']):
                        line = line + '  op_arg,\n'
                    line = line + '  op_arg );\n'
                    fid.write(line)

                fid.write('#ifdef OPENACC\n#ifdef __cplusplus\n}\n#endif\n#endif\n')
                fid.write('\n')
                loc_old = locs[loc] + header_len-1
                continue

            if locs[loc] in loc_loops:
                indent = indent + ' ' * len('op_par_loop')
                endofcall = text.find(';', locs[loc])
                curr_loop = loc_loops.index(locs[loc])
                name = loop_args[curr_loop]['name1']
                line = str(' op_par_loop_' + name + '(' +
                           loop_args[curr_loop]['name2'] + ',' +
                           loop_args[curr_loop]['set'] + ',\n' + indent)

                for arguments in range(0, loop_args[curr_loop]['nargs']):
                    elem = loop_args[curr_loop]['args'][arguments]
                    if elem['type'] == 'op_arg_dat':
                        line = line + elem['type'] + '(' + elem['dat'] + \
                            ',' + elem['idx'] + ',' + elem['map'] + \
                            ',' + elem['dim'] + ',' + elem['typ'] + \
                            ',' + elem['acc'] + '),\n' + indent
                    elif elem['type'] == 'op_arg_gbl':
                        line = line + elem['type'] + '(' + elem['data'] + \
                            ',' + elem['dim'] + ',' + elem['typ'] + \
                            ',' + elem['acc'] + '),\n' + indent

                fid.write(line[0:-len(indent) - 2] + ');')

                loc_old = endofcall + 1
                continue

            if locs[loc] in loc_consts:
                curr_const = loc_consts.index(locs[loc])
                endofcall = text.find(';', locs[loc])
                name = const_args[curr_const]['name']
                fid.write(indent[0:-2] + 'op_decl_const2("' + name.strip() +
                          '",' + str(const_args[curr_const]['dim']) + ',' +
                          const_args[curr_const]['type'] + ',' +
                          const_args[curr_const]['name2'].strip() + ');')
                loc_old = endofcall + 1
                continue

        fid.write(text[loc_old:])
        fid.close()

        f.close()
    # end of loop over input source files

    #  errors and warnings

    if ninit == 0:
        print' '
        print'-----------------------------'
        print'  WARNING: no call to op_init'
        if auto_soa==1:
          print'  WARNING: code generated with OP_AUTO_SOA,\n but couldn\'t modify op_init to pass\n an additional parameter of 1.\n Please make sure OP_AUTO_SOA is set when executing'
        print'-----------------------------'

    if nexit == 0:
        print' '
        print'-------------------------------'
        print'  WARNING: no call to op_exit  '
        print'-------------------------------'

    if npart == 0 and nhdf5 > 0:
        print' '
        print'---------------------------------------------------'
        print'  WARNING: hdf5 calls without call to op_partition '
        print'---------------------------------------------------'

    #
    #  finally, generate target-specific kernel files
    #


    op2_gen_seq(str(sys.argv[1]), date, consts, kernels) # MPI+GENSEQ version - initial version, no vectorisation
    op2_gen_mpi_vec(str(sys.argv[1]), date, consts, kernels) # MPI+GENSEQ with code that gets auto vectorised with intel compiler (version 15.0 and above)

    #code generators for OpenMP parallelisation with MPI
    #op2_gen_openmp(str(sys.argv[1]), date, consts, kernels) # Initial OpenMP code generator
    op2_gen_openmp_simple(str(sys.argv[1]), date, consts, kernels) # Simplified and Optimized OpenMP code generator
    op2_gen_openacc(str(sys.argv[1]), date, consts, kernels) # Simplified and Optimized OpenMP code generator

    #code generators for NVIDIA GPUs with CUDA
    #op2_gen_cuda(str(sys.argv[1]), date, consts, kernels,sets) # Optimized for Fermi GPUs
    op2_gen_cuda_simple(str(sys.argv[1]), date, consts, kernels,sets) # Optimized for Kepler GPUs

    # generates openmp code as well as cuda code into the same file
    #op2_gen_cuda_simple_hyb(str(sys.argv[1]), date, consts, kernels,sets) # CPU and GPU will then do comutations as a hybrid application

    import subprocess
    retcode = subprocess.call("which clang-format > /dev/null", shell=True)
    if retcode == 0:
      retcode = subprocess.call("$OP2_INSTALL_PATH/../translator/c/python/format.sh", shell=True)
    else:
      print 'Cannot find clang-format in PATH'
      print 'Install and add clang-format to PATH to format generated code to conform to code formatting guidelines'

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main()
    # Print usage message if no arguments given
    else:
        print __doc__
        sys.exit(1)

