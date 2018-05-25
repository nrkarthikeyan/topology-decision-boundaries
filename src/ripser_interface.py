import numpy as np

def ripser(dm, pdname, 
           ripser_path = '../ripser/ripser', 
           maxdim = 1):
    """
    Calls the ripser code to obtain the persistence diagram
    # dims start from 0 in this program
    # 0 - number of simply connected components
    # 1 - number of holes
    """

    import subprocess
    import os
    # import sys
    # import numpy as np

    ripser_path = os.path.abspath(ripser_path)
    distmat = os.path.abspath(dm)
    # persdiag = os.path.abspath(pdname)

    #assert subprocess.call([ripser_path, distmat, "--format", "dipha"]) == 0
    print("Using program ripser to compute persistence diagrams")
    
    p = subprocess.Popen([ripser_path, distmat, "--format", "dipha",
                         "--dim", str(maxdim)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = p.communicate()


    if p.returncode != 0:
        raise Exception("Error with Ripser program - " + str(output[1]))
    else:
        dims, birth_values, death_values = parse_ripser_output(output)

    return dims, birth_values, death_values

#     dims, birth_values, death_values = load_persistence_diagram(pdname)

#     death_values[dims < 0] = np.inf
#     dims[dims < 0] = 0

def parse_ripser_output(output):
    """ Parse the output file of the ripser program
        if it succeeds, we will get three arrays:
        dims - dimension of each homoogy group
        birth_values - the birth times of each group
        death_values - the death times of each group
        
        else these arrays will be of length 0
        
    """
    
    import re
    p1 = re.compile("persistence intervals in dim ([0-9]+):")
    p2 = re.compile("\[([0-9]+),([0-9 ]+)\)")

    ll = output[0].split("\n")
    #print(ll)

    cnt = 0
    dims = np.zeros(len(ll))
    birth_values = np.zeros(len(ll))
    death_values = np.zeros(len(ll))

    dim = 0
    for l in ll:

        dims[cnt] = dim

        # Find dimension
        m = p1.match(l.strip())
        if m:
            dim = float(m.groups()[0])
            dims[cnt] = dim
            continue

        # Find birth and death times
        m = p2.match(l.strip())
        if m:
            v1 = float(m.groups()[0])
            v2 = m.groups()[1]
            if v2 == " ":
                v2 = np.Inf
            else:
                v2 = float(v2)

            birth_values[cnt] = v1
            death_values[cnt] = v2

            cnt += 1

    dims = dims[0:cnt]
    birth_values = birth_values[0:cnt]
    death_values = death_values[0:cnt]

    return dims, birth_values, death_values


