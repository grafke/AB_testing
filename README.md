Bayesian A/B testing

* MCMC (slowest. Recommended for small numbers <1e5) (see see plots/output.png)
![img] (https://raw.githubusercontent.com/grafke/AB_testing/master/plots/output.png)
* Numerical integrating over joint distribution (see plots/output_joint.png)
  *Output: "P(Test group is better that Control group) = 97.9587576712%"
![img] (https://raw.githubusercontent.com/grafke/AB_testing/master/plots/output_joint.png)
* Closed form solution
  **Output: "P(Test group is better that Control group) = 97.9943832157%"
* Approximate closed form solution
  **Output: "P(Test group is better that Control group) = 97.9963126837%"

G test
  *Output "Test statistic = 95.99%"
  
  
Which method to choose depends on a particular problem. Some problems do not have closed form solutions,
some have strict time|computational constraints, some require high precision.