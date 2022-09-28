import re
import numpy as np

def get_x_y_axis(filename : str, aggreg=np.mean) :
    """Parses the output of an AutoTVM run to get x and y axis for plotting
    througput depending on batch size

    Parameters:
    filename (str): The location of the file to parse
    aggreg (function): The aggregation function to use (e.g. np.min, np.max)

    Returns:
    x (list): The x axis, corresponding to the batch size of the model
    y (list): The y axis corresponding to the throughput (min, max or average)

   """
    file1 = open(filename, 'r')
    Lines = file1.readlines()
    secDict = {}

    currentSize=-1

    for i, line in enumerate(Lines):
        if line.startswith('                AUTOTUNING'):

            size = [int(s) for s in line.split() if s.isdigit()][0]
            secDict[str(size)] = [[],[],[],[],[],[]]
            currentSize=str(size)
            #  print(size)

        if line.startswith('[Task  '):
            #7th char of the line Gives the number of the task
            secDict[currentSize][int(line[7])-1].append(float(re.sub("[^\d\.]", "", line.split('|')[0].split('/')[3])))

    for elem in secDict :
        for i, e in enumerate(secDict[elem]):
            secDict[elem][i] = max(e)

    for elem in secDict :
        secDict[elem] = aggreg(secDict[elem])

    x = [int(k) for k in secDict.keys()]
    y = secDict.values()

    return (x,list(y))


def get_x_y_axis_duration(filename : str) :
    """Parses the output of an AutoTVM run to get x and y axis for plotting the
    duration time depending on the batch size

    Parameters:
    filename (str): The location of the file to parse

    Returns:
    secDict (dictionary of dictionnaries): The dict containing the unoptimized
    time, the optimized time, and the speedup for each batch size. Dict struct
    is as follow : my_dict[my_batch_size]['unoptimized']. 3 possible values
    for the inner dict, which are : 'unoptimized', 'optimized' and 'speedup'
   """

    file1 = open(filename, 'r')
    Lines = file1.readlines()
    secDict = {}

    currentSize=-1

    for i, line in enumerate(Lines):
        if line.startswith('                AUTOTUNING'):

            size = [int(s) for s in line.split() if s.isdigit()][0]
            secDict[str(size)] = {}
            currentSize=str(size)

        if line.startswith('optimized:'):
            #7th char of the line Gives the number of the task
            secDict[currentSize]['optimized']   = float(line.split(',')[0].split(':')[2])
        if line.startswith('unoptimized:'):
            secDict[currentSize]['unoptimized'] = float(line.split(',')[0].split(':')[2])


    for key_bs in secDict :
        unopt = secDict[key_bs]['unoptimized']
        opt = secDict[key_bs]['optimized']

        secDict[key_bs]['speedup'] = unopt / opt

    return secDict