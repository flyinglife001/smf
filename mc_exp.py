#coding:utf-8
import random
import math
import sys
from subprocess import Popen, PIPE

def run(command):
    process = Popen(command, stdout=PIPE, shell=True)
    while True:
        line = process.stdout.readline()
        if not line:
            break
        yield line.strip().decode('utf-8')

def grid_generate(valid_dict):
    """
    :param valid_dict: it's form is params with str items.
    :return: paras dict with str items.
    """

    #idx is reaching to the top of list or not.
    #print 'idx',idx_L

    idx_L = [0]*len(valid_dict)

    while True:
        is_finished = True
        comb_tuple = {}
        max_add_idx = -1

        i = 0
        for k,S in valid_dict.items():
            if idx_L[i] == len(S):
                max_add_idx = i
            elif idx_L[i] != len(S) - 1:
                is_finished = False
                comb_tuple[k] = S[idx_L[i]]
            else:
                comb_tuple[k] = S[idx_L[i]]

            i += 1

        #compute the next combination
        for i in range(max_add_idx + 1):
            idx_L[i] = 0
        idx_L[max_add_idx + 1] += 1

        if len(comb_tuple) == len(valid_dict):
            yield comb_tuple

        if is_finished:
            break

#generate the parameter config

def rel_rmse(dname):
    #print l_value
    fname = './%s-rmse.txt' %(dname[0])
    result_f = open(fname,'w')

    cfg_dict = {}

    config = {
    'lambda': [10, 50, 100, 150, 200, 250],
    'seed': [43],
    'epoches': [100],
    'normalize': [1]
    }

    cfg_dict['softimpute'] = config.copy()
    config['epsilon'] = [1e-2]
    config['k'] = [10,50,100]

    cfg_dict['nmc'] = config.copy()
    cfg_dict['fmc'] = config.copy()
    cfg_dict['svt'] = config.copy()
    cfg_dict['fpca'] = config.copy()

    for name,config in cfg_dict.items():
        for input in grid_generate(config):
            input['method'] = name
            input['dt'] = dname[0]
            input['delimiter'] = dname[1]
            input['train'] = './%s.train'%dname[0]
            input['test'] = './%s.test'%dname[0]
            input['r'] = dname[2]
            input['c'] = dname[3]

            print(input)
            cmd_str = 'julia ./main.jl '
            for k,v in input.items():
                cmd_str += str(k) + ' ' + str(v) + ' '

            print(cmd_str)

            result = run(cmd_str)
            last_l = ''
            for l in result:
                print(l)
                last_l = l
            #read the running results
            result_f.write(last_l + '\n')
    result_f.close()

def simulate(dname):
    #print l_value
    fname = './%s.txt' %(dname[0])
    #fname = 'test.txt'
    result_f = open(fname,'w')

    cfg_dict = {}

    config = {
        'lambda': [10, 50, 100, 150, 200, 250],
        'seed': [43],
        'epoches': [60],
        'normalize': [1]
    }
    cfg_dict['softimpute'] = config.copy()
    config['epsilon'] = [1e-2]
    config['k'] = [5,10,15]

    cfg_dict['fmc'] = config.copy()
    cfg_dict['nmc'] = config.copy()
    cfg_dict['svt'] = config.copy()
    cfg_dict['fpca'] = config.copy()

    for name,config in cfg_dict.items():
        for input in grid_generate(config):
            input['method'] = name
            input['dt'] = dname[0]
            input['delimiter'] = dname[1]
            input['train'] = './%s.train'%dname[0]
            input['test'] = './%s.test'%dname[0]
            input['r'] = dname[2]
            input['c'] = dname[3]

            print(input)
            cmd_str = 'julia ./main.jl '
            for k,v in input.items():
                cmd_str += str(k) + ' ' + str(v) + ' '

            print(cmd_str)

            result = run(cmd_str)
            last_l = ''
            for l in result:
                print(l)
                last_l = l
            #read the running results
            result_f.write(last_l + '\n')
    result_f.close()

if __name__ == '__main__':

    datasets = [('sim-0.0-150-300','::',150,300),
                ('sim-0.8-150-300', '::', 150, 300),
                ('sim-3.0-150-300', '::', 150, 300),
                ]
    for d in datasets:
        simulate(d)

    datasets = [
        ("yahoo_music", "::", 3000, 3000),
        ("douban", "::", 3000, 3000),
        ("flixster", "::", 3000, 3000),
        ("movielens", "::", 943, 1682),
    ]
    for d in datasets:
        rel_rmse(d)

