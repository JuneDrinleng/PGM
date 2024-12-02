# PGM
This is my repo to realize the probabilistic graph model‘s algorithm.   
It mainly includes the inference and learning part 

## Inference
1. Variable Elimination and Belief Propagation  
we use clique tree to realize the inference algorithm. however, the algorithm is written in a bad way. we even can not be sure it is correct or not.  
2. Mean Field Inference
we use mean field inference to realize the inference algorithm. The result of it is quite near to the result we use pypgm package.
3. Gibbs Sampling
we use Gibbs Sampling to realize the inference algorithm. we at first try to find out the reason of not convergence. And finally we realize that with the iteration number increase, we are not making it converge, but continuing to sample.
4. Metropolis-Hastings algorithm
we use Metropolis-Hastings algorithm to realize the inference algorithm. The result is near the Mean Field Inference. Attention: the algorithm is also the sampling algorithm.
## Learning

## copyright
The code is written by myself. If you want to use it, please contact me. It is not allowed to use directly in your project for PGM course.   
