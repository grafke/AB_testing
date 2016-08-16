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

print "P(Test group is better that Control group) = %s%%" % (
100 * analyze_joint(base_pos, base_neg, variant_pos, variant_neg, minp=.158, maxp=.16))

# Bayesian AB test with small numbers

base = np.loadtxt('data/base.data')
variant = np.loadtxt('data/variant.data')

analyze_mcmc(base, variant)

# G test

g = g_test(base_pos, base_neg, variant_pos, variant_neg)
print('Test statistic = %s%%' % round(g_to_p(g, 1) * 100, 2))
