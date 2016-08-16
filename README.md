Bayesian A/B testing

* MCMC (slowest. Recommended for small numbers <1e5) (see see plots/output.png)
![img] (https://raw.githubusercontent.com/grafke/AB_testing/master/plots/output.png)
* Numerical integrating over joint distribution (see plots/output_joint.png)
![img] (https://raw.githubusercontent.com/grafke/AB_testing/master/plots/output_joint.png)
* Closed form solution
   -Output: "P(Test group is better that Control group) = 90%"
* Approximate closed form solution
   -Output: "P(Test group is better that Control group) = 90%"

G test

Which method to choose depends on a particular problem. Some problems do not have closed form solutions,
some have strict time|computational constraints, some require high precision.