cimport cython
@cython.boundscheck(False) 
@cython.wraparound(False) 

cpdef compute_C_closest(int [:,::1] C, int [:,::1] S, int [::1] inds, 
                      double [:,::1] D, double s, double [::1] sigarr):
    """
    Compute a sparse representation for C:
    Each row of C will have its diagonal index and the closest N
    nearest neighbors from the opposing class
    S - Closest neighbor indices from opposing classes (sorted)
    D - Closest neighbor distances from opposing classes (sorted)
    inds - number of elements in each row of C
    s - multiplier for local scale
    sigarr - local scale
    """
    
    cdef int m = C.shape[0]
    cdef int N = C.shape[1] # This N is actually N+1
    cdef int i1, i2, indsi1, indc
    cdef double s1, yi1, s2
    
    # Update C with neighbors
    for i1 in range(m):
        # temp variables
        indsi1 = inds[i1]
        s1 = s*sigarr[i1]
        
        # Go through the actual "N" neighbors 
        for i2 in range(N-1):
            indc = S[i1,i2]
            if indc == -1:
                break
            elif D[i1, i2] <= s1*sigarr[indc]:
                C[i1, indsi1] = indc
                indsi1 += 1
        inds[i1] = indsi1
        
cpdef compute_C_first(int [:,::1] C, int [::1] inds, double [:,::1] D, 
                double s, double [::1] sigarr, double [::1] y):
    """
    Compute a sparse representation for C using the first (not closest) N 
    nearest neighbors from the opposing class
    """
    
    cdef int m = C.shape[0]
    cdef int N = C.shape[1]
    cdef int i1, i2, indsi1
    cdef double s1, yi1, s2
    
    # Update C with neighbors
    for i1 in range(m):
        # temp variables
        indsi1 = inds[i1]
        s1 = s*sigarr[i1]
        yi1 = y[i1]

        for i2 in range(m):
            # Choose only the first N neighbors 
            # (this is actually the first occuring, not the closest)
            if indsi1 >= N:
                break
            else:
                s2 = s1*sigarr[i2]
                if (D[i1, i2] <= s2) and (yi1 != y[i2]):
                    C[i1, indsi1] = i2
                    indsi1 += 1
        inds[i1] = indsi1
        
cpdef init_C_inds(int [:,::1] C, int [::1] inds):
    
    cdef int m = C.shape[0]
    
    # Initialize C
    for i1 in range(m):
        C[i1,0] = i1
        inds[i1] += 1

cpdef update_P(int [:,::1] C, int [::1] inds, int [:,::1] P, int mval):
    
    cdef int m = C.shape[0]
    cdef int N = C.shape[1]
    cdef int i1, i2, indsi1, indsh1, h1, h1i3
    cdef int* Crow
    cdef int* Prow
    cdef int* H1row
    #cdef int mval = 1
    
    # update all P values for 0- and 1-hop neighbors
    for i1 in range(m):
        indsi1 = inds[i1]
        Crow = &C[i1,0]
        Prow = &P[i1,0]
        for i2 in range(indsi1):
            Prow[Crow[i2]] += 1
            
    # Now do 2 hop neighbor update
    for i1 in range(m):
        Prow = &P[i1,0]
        Crow = &C[i1,0]
        indsi1 = inds[i1]
        for i2 in range(1,indsi1):
            h1 = Crow[i2]
            H1row = &C[h1,0]
            indsh1 = inds[h1]
            for i3 in range(1,indsh1):
                h1i3 = H1row[i3]
                if Prow[h1i3] < mval and h1i3 != i1:
                # if h1i3 != i1:
                    Prow[h1i3] += 1