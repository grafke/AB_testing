from __future__ import division
import numpy as np
from ab_testing_methods import g_test, g_to_p, analyze_mcmc, analyze_closed_form, analyze_approx, analyze_joint


base_pos = 2643146
base_neg = 13980139
variant_pos = 2646705
variant_neg = 13971736

# Bayesian AB test with big numbers

print "P(Test group is better that Control group) = %s%%" % (
100 * analyze_closed_form(base_pos, base_neg, variant_pos, variant_neg))

print "P(Test group is better that Control group) = %s%%" % (
100 * analyze_approx(base_pos, base_neg, variant_pos, variant_neg))

print "P(Test group is better than Control group) = %s%%" % (
100 * analyze_joint(base_pos, base_neg, variant_pos, variant_neg)[0])
print "P(Test group is worse than Control group) = %s%%" % (
100 * analyze_joint(base_pos, base_neg, variant_pos, variant_neg)[1])


# Bayesian AB test with small numbers

base_pos = 5559
base_neg = 32659
variant_pos = 5583
variant_neg = 32559

base = np.r_[[0] * base_neg + [1] * base_pos]
variant = np.r_[[0] * variant_neg + [1] * variant_pos]

analyze_mcmc(base, variant)

# G test

g = g_test(base_pos, base_neg, variant_pos, variant_neg)
print('Test statistic = %s%%' % round(g_to_p(g, 1) * 100, 2))
