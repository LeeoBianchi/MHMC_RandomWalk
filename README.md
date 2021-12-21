# MHMC_RandomWalk
**A Python-based, parallelized (multi-core) implementation of the Metropolis-Hastings random walk algorithm for Bayesian inference of the parameters through their posterior distributions.**

The task was to analyze a dataset regarding cases of Malaria in Gambia, in order to determine the influential factors among a serie of parameters.

The code is divided in three parts:
 - "MC.py" includes all the useful functions, including the model's Likelihood and the random-scan Metropolis-Hastings algorithm implementing the random walk for sampling the parameters' posterior distributions.
 - "mhmc.py" constitutes the code's core. It reads the data and recalling the functions in "MC.py" starts as many markov chains as I set, or as the number of available cores the machine has. In the end it saves the sampling obtained in .gz files.
 - The Main script reads the .gz output files and computes Gelman-Rubin convergence analysis, as well as plotting the posterior distributions.
