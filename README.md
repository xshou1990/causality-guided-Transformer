# Pairwise Causality Guided Transformers for Event Sequences

Source code for Pairwise Causality Guided Transformers for Event Sequences

# Run the code for  Pairwise Causality Guided Transformers for Event Sequences

### Dependencies
* Python 3.7.
* [Anaconda](https://www.anaconda.com/) contains all the required packages.
* [PyTorch](https://pytorch.org/) version 1.8.0. above

### Instructions
1. folders:
a. data. The **data** folder is already placed inside the root folder. Each dataset contains train dev test pickled files. 
b.code. We used summs_utils_v4.py to generate BSuMM-1; summs_utils_v3.py to generate BSuMM-2. Please change the window accordingly.  
c.preprocess. We used to convert data into standard inpt for our model.
d.transformer. This folder contains our main models, modules that supports the training of our models.
e.prior. This folder contains prior distribution for our experiments for PAIN.

2. To run. simply place "python run.py" on commandline.  One can modify the parameters in the .py before run.

3. Appendix: Pairwise Causality Guided Transformers for Event Sequences is attached. 