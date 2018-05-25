# Topology of decision boundaries

This repository implements Plain- and Local-Scaled Labeled Vietoris-Rips (P-LVR and LS-LVR) complexes to analyze the topology of decision boundaries.

Details available at: 


**Key Notebooks**

- simple\_2\_class.ipynb - Analyzes the decision boundary topology of the simple two class data using the P-LVR and LS-LVR complexes
    - The results are saved in the "results" folder

**Compiling Cython code for Labeled VR Complexes**

cd src

python setup.py build_ext --inplace

### External Dependencies

https://github.com/Ripser/ripser
