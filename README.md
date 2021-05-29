# EE364B Final Project: Python Library for Phase Retrieval Methods
This is my final project submission for EE364B. I developed a set of python-based methods to generated and test verious phase retrieval algorithms. The motivation behind this project was seeing a lot of different papers in recent years focused on convex formulations for phase retrieval, but not many python-based implementations of these methods in a way that can be readily used to benchmark several algorithms for a given signal. This is very much a "prototype" and any/all contributions are welcome.

## Required Libraries
* Numpy
* CVXPY & CVXOPT
* MOSEK
* Scipy
* Pyplot

## Parameters
* n, signal vector dimension
* m, measurement vector dimension
* alpha, noise paramter (Gaussian on interval [0,alpha])
* noise, boolean selection

## Methods
* fienup - alternating projections
* phasecut - alternative implementation for PhaseLift via trace minimization and "lifting"
* phaselift - original phaselift implementation but untested due solver limitations
* phasemax - non-lifting method via basis pursuit to solve dual problem


