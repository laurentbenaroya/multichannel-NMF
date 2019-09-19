The aim of this project is to perform Multichannel Source Separation with the Projected Gradient algorithm.

The cost function is the likelihood of the standard complex Gaussian model formulation [1].
It can be generalized to any cost function, such as the quadratic cost. This is the strength of the approach 
compared to model based approaches.
The gradients are not straightforward to obtain "by hand".

Though, it seems that the algorithm depends on the initialisation, similar to the Expectation-Maximization approach.
Currently only the instantaneous case is implemented. 
The case of a complex, frequency dependent, mixing matrix is less straightforward 
(related to the gradient descent of a complex matrix).

A jupyter notebook with a demo is in demo/demo_C_Mae.ipynb

ELB - 09/2019

[1] A. Ozerov and C. Févotte, 
"Multichannel nonnegative matrix factorization in convolutive mixtures for audio source separation," 
IEEE Trans. on Audio, Speech and Lang, 2010.
