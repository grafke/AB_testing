Bayesian A/B testing methods
=====================

Quick and dirty Python code for evaluating AB tests

* MCMC (slowest. Recommended for small numbers <1e5) (see see plots/output.png)

````
>>> p_success_uni, p_failure_uni, p_success_beta, p_failure_beta = analyze_mcmc(base_data, variant_data)
>>> print "P(Test group is better than Control group) = %s%%" % (100 * p_success_beta)
"P(Test group is better that Control group) = 70.47%"
````

![img] (https://raw.githubusercontent.com/grafke/AB_testing/master/plots/output.png)

* Numerical integrating over joint distribution (see plots/output_joint.png)

````
>>> p_success, p_failure = analyze_joint(base_pos, base_neg, variant_pos, variant_neg)
>>> print "P(Test group is better than Control group) = %s%%" % (100 * p_success)
"P(Test group is better that Control group) = 97.9587576712%"
````

![img] (https://raw.githubusercontent.com/grafke/AB_testing/master/plots/output_joint.png)
  
![img] (https://raw.githubusercontent.com/grafke/AB_testing/master/plots/AB_animation.gif)
  
* Closed form solution

````
>>> p_success = analyze_closed_form(base_pos, base_neg, variant_pos, variant_neg)
>>> print "P(Test group is better than Control group) = %s%%" % (100 * p_success)
"P(Test group is better that Control group) = 97.9943832157%"
````

* Approximate closed form solution


````
>>> p_success = analyze_closed_form(base_pos, base_neg, variant_pos, variant_neg)
>>> print "P(Test group is better than Control group) = %s%%" % (100 * p_success)
"P(Test group is better that Control group) = 97.9963126837%"
````

G test

````
>>> g = g_test(base_pos, base_neg, variant_pos, variant_neg)
>>> print('Test statistic = %s%%' % round(g_to_p(g, 1) * 100, 2)) 
"Test statistic = 28.09%"
````
  
  
Which method to choose depends on a particular problem. Some problems do not have closed form solutions,
some have strict time|computational constraints, some require high precision.