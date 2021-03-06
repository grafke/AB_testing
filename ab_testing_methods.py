from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from scipy.special._ufuncs import betaln
from scipy.stats.distributions import chi2
from scipy.stats import norm, beta

def ci_interval(base_reach, base_visitors, variant_reach, variant_visitors, CI=0.95, interval="two-sided", e=0.5):
    """
    Confidence interval of the lift of the two campaigns
    http://ilpubs.stanford.edu:8090/993/2/displayadinfluenceTR.pdf

    base_reach is the total reach for campaign A,
    base_visitors is the total visitors for campaign A,
    etc.
    """
    # lift of B over A
    if interval not in ("two-sided", "upper", "lower"):
        raise ValueError("Interval must be either 'two-sided', 'upper', or 'lower'.")
    # add e to each number to keep weird stuff from happening when Pa or Pb are close to 0
    base_reach += e
    base_visitors += e
    variant_reach += e
    variant_visitors += e
    log_lift_mean = math.log((float(variant_visitors) / variant_reach) / (float(base_visitors) / base_reach))
    # if the interval is two-sided then the tail probabilities are cut in half
    pval = (1.0 - CI) / 2 if interval == "two-sided" else (1.0 - CI)
    zval = norm.ppf(1.0 - pval)
    try:
        se = math.sqrt((1.0 / variant_visitors) - (1.0 / variant_reach) + (1.0 / base_visitors) - (1.0 / base_reach))
    except ValueError as e:
        sys.stderr.write('Invalid reach: %s, %s, %s, %s' % (base_reach, base_visitors, variant_reach, variant_visitors))
        return (-1.0, float("inf"))
    # return a tuple of (lower_limit, upper_limit)
    return (
        -1.0 if interval == "lower" else math.exp(
            log_lift_mean - zval * se) - 1,
        float("inf") if interval == "upper" else math.exp(
            log_lift_mean + zval * se) - 1
    )
  
  
def visualize_mcmc(normal_base_samples, normal_variant_samples, normal_delta_samples, beta_base_samples,
                   beta_variant_samples, beta_delta_samples, output='plots/output.png'):
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(321)

    plt.hist(normal_base_samples, histtype='stepfilled', bins=25, alpha=0.85,
             label="Uniform posterior of $p_A$", color="#A60628", normed=True,
             edgecolor="none")
    plt.legend(loc="upper right")
    plt.title("Normal posterior distributions of \n$p_A$, $p_B$, and delta unknowns\n\n")

    ax = fig.add_subplot(323)

    plt.hist(normal_variant_samples, histtype='stepfilled', bins=25, alpha=0.85,
             label="Uniform posterior of $p_B$", color="#467821", normed=True,
             edgecolor="none")
    plt.legend(loc="upper right")

    ax = fig.add_subplot(325)

    plt.hist(normal_delta_samples, histtype='stepfilled', bins=50, alpha=0.85,
             label="Uniform posterior of $p_B$ - $p_A$", color="#7A68A6", normed=True,
             edgecolor="none")
    plt.legend(loc="upper right")
    plt.vlines(0, 0, 500, color="black", alpha=.5)
    plt.xlabel("P(Base is BETTER than variant): %.3f\n" % (normal_delta_samples < 0).mean() +
               "P(Base is WORSE than variant): %.3f" % (normal_delta_samples > 0).mean())

    ax = fig.add_subplot(322)

    plt.hist(beta_base_samples, histtype='stepfilled', bins=25, alpha=0.85,
             label="Beta posterior of $p_A$", color="#A60628", normed=True,
             edgecolor="none")
    plt.legend(loc="upper right")
    plt.title("Beta posterior distributions of \n$p_A$, $p_B$, and delta unknowns\n\n")

    ax = fig.add_subplot(324)

    plt.hist(beta_variant_samples, histtype='stepfilled', bins=25, alpha=0.85,
             label="Beta posterior of $p_B$", color="#467821", normed=True,
             edgecolor="none")
    plt.legend(loc="upper right")

    ax = fig.add_subplot(326)

    plt.hist(beta_delta_samples, histtype='stepfilled', bins=50, alpha=0.85,
             label="Beta posterior of $p_B$ - $p_A$", color="#7A68A6", normed=True,
             edgecolor="none")
    plt.legend(loc="upper right")
    plt.vlines(0, 0, 500, color="black", alpha=.5)
    plt.xlabel("P(Base is WORSE than variant): %.3f\n" % (beta_delta_samples > 0).mean() +
               "P(Base is BETTER than variant): %.3f" % (beta_delta_samples < 0).mean())

    # plt.show()
    plt.savefig(output)


def __analyze_mcmc(base=[], variant=[], num_samples=20000, burn=1000, visualize=True, output='plots/output.png'):
    """
    Bayesian AB test using MCMC (PYMC2)
    Slow with big numbers
    Read more: http://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter3_MCMC/Chapter3.ipynb

    ||Use analyze_mcmc()||

    :param base:
    :param variant:
    :param output:
    :param num_samples:
    :param burn:
    :return:
    """
    import pymc as pm
    base_pos = np.count_nonzero(base)
    base_all = len(base)

    variant_pos = np.count_nonzero(variant)
    variant_all = len(variant)

    p_A_n = pm.Uniform('p_A_n', lower=0, upper=1)
    p_B_n = pm.Uniform('p_B_n', lower=0, upper=1)

    p_A_b = pm.Beta('p_A_b', base_pos + 1, base_all - base_pos + 1)
    p_B_b = pm.Beta('p_B_b', variant_pos + 1, variant_all - variant_pos + 1)

    @pm.deterministic
    def delta(p_A=p_A_n, p_B=p_B_n):
        return p_B - p_A

    @pm.deterministic
    def beta_delta(p_A=p_A_b, p_B=p_B_b):
        return p_B - p_A

    obs_A = pm.Bernoulli("obs_A_n", p_A_n, value=base, observed=True)
    obs_B = pm.Bernoulli("obs_B_n", p_B_n, value=variant, observed=True)

    obs_A_b = pm.Bernoulli("obs_A_b", p_A_b, value=base, observed=True)
    obs_B_b = pm.Bernoulli("obs_B_b", p_B_b, value=variant, observed=True)

    mcmc = pm.MCMC([p_A_n, p_B_n, delta, obs_A, obs_B])
    mcmc.sample(num_samples, burn)

    mcmc_beta = pm.MCMC([p_A_b, p_B_b, beta_delta, obs_A_b, obs_B_b])
    mcmc_beta.sample(num_samples, burn)

    p_A_samples = mcmc.trace("p_A_n")[:]
    p_B_samples = mcmc.trace("p_B_n")[:]
    delta_samples = mcmc.trace("delta")[:]

    p_A_b_samples = mcmc_beta.trace("p_A_b")[:]
    p_B_b_samples = mcmc_beta.trace("p_B_b")[:]
    beta_delta_samples = mcmc_beta.trace("beta_delta")[:]

    if visualize:
        visualize_mcmc(p_A_samples, p_B_samples, delta_samples, p_A_b_samples, p_B_b_samples, beta_delta_samples,
                       output)

    p_success_uniform = (delta_samples > 0).mean()
    p_failure_uniform = (delta_samples < 0).mean()
    p_success_beta = (beta_delta_samples > 0).mean()
    p_failure_beta = (beta_delta_samples < 0).mean()

    return p_success_uniform, p_failure_uniform, p_success_beta, p_failure_beta


def analyze_mcmc(base=[], variant=[], num_samples=20000, output='plots/output.png', visualize=True):
    """
    Bayesian AB test using MCMC (PYMC3)
    Slow with big numbers
    Read more: http://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter3_MCMC/Chapter3.ipynb

    :param base: list
        control group outcomes [0, 1, 0, 0]
    :param variant: list
        test group outcomes [0, 1, 0, 0]
    :param num_samples: int, optional
        number of samples to draw using MCMC, default: 20000
    :param output: string, optional
        filename to save the plots, default: 'plots/output.png'
    :param visualize: boolean, optional
        default: True
    :return list
        [p_success_uniform, p_failure_uniform, p_success_beta, p_failure_beta]
    """
    import pymc3 as pm

    base_pos = np.count_nonzero(base)  # number of positive events in the control group
    base_all = len(base)  # number of events in the control group

    variant_pos = np.count_nonzero(variant)  # number of positive events in the test group
    variant_all = len(variant)  # number of events in the test group

    # Simple mode assuming uniform distributions
    with pm.Model() as normal_model:
        p_A_n = pm.Uniform('Uniform Base', lower=0, upper=1)
        p_B_n = pm.Uniform('Uniform Variant', lower=0, upper=1)

        delta = pm.Deterministic('delta', p_B_n - p_A_n)

        obs_A = pm.Bernoulli("obs_A_n", p_A_n, observed=base)
        obs_B = pm.Bernoulli("obs_B_n", p_B_n, observed=variant)

        start = pm.find_MAP()
        step = pm.Metropolis([p_A_n, p_B_n, delta, obs_A, obs_B])
        trace = pm.sample(num_samples, step, start=start)

        normal_base_data = trace.get_values('Uniform Base')
        normal_variant_data = trace.get_values('Uniform Variant')
        delta_data = trace.get_values('delta')

        # axn = pm.traceplot(trace)

    # Model assuming Beta distributions
    with pm.Model() as beta_model:
        p_A_b = pm.Beta('Beta Base', base_pos + 1, base_all - base_pos + 1)
        p_B_b = pm.Beta('Beta Variant', variant_pos + 1, variant_all - variant_pos + 1)

        beta_delta = pm.Deterministic('Beta delta', p_B_b - p_A_b)

        obs_A_b = pm.Bernoulli("obs_A_b", p_A_b, observed=base)
        obs_B_b = pm.Bernoulli("obs_B_b", p_B_b, observed=variant)

        start = pm.find_MAP()
        step = pm.Metropolis([p_A_b, p_B_b, beta_delta, obs_A_b, obs_B_b])
        trace = pm.sample(num_samples, step, start=start)

        beta_base_data = trace.get_values('Beta Base')
        beta_variant_data = trace.get_values('Beta Variant')
        beta_delta_data = trace.get_values('Beta delta')

        # axb = pm.traceplot(trace)

    p_success_uniform = (delta_data > 0).mean()
    p_failure_uniform = (delta_data < 0).mean()
    p_success_beta = (beta_delta_data > 0).mean()
    p_failure_beta = (beta_delta_data < 0).mean()

    if visualize:
        visualize_mcmc(normal_base_data, normal_variant_data, delta_data, beta_base_data, beta_variant_data,
                       beta_delta_data,
                       output)

    return p_success_uniform, p_failure_uniform, p_success_beta, p_failure_beta


def analyze_closed_form(base_pos, base_neg, variant_pos, variant_neg):
    """
    Closed form solution for Bayesian AB test.
    Read more: http://www.evanmiller.org/bayesian-ab-testing.html#binary_ab_derivation
    Can be slow with large numbers

    :param base_pos: int
        number of positive events in the control group
    :param base_neg: int
        number of negative events in the control group
    :param variant_pos: int
        number of positive events in the test group
    :param variant_neg: int
        nunmber of negative events in the test group
    :return: float
        probability that Test group is better that Control group.
    """
    base_pos += 1
    base_neg += 1
    variant_pos += 1
    variant_neg += 1
    return np.sum([np.exp(
        betaln(base_pos + i, variant_neg + base_neg) - np.log(variant_neg + i) - betaln(1 + i, variant_neg) - betaln(
            base_pos, base_neg)) for i in range(0, variant_pos - 1)])


def analyze_approx(base_pos, base_neg, variant_pos, variant_neg):
    """
    Approximate solution for a Bayesian AB test.
    Source: http://www.johndcook.com/fast_beta_inequality.pdf
    Very fast

    :param base_pos: int
        number of positive events in the control group
    :param base_neg: int
        number of negative events in the control group
    :param variant_pos: int
        number of positive events in the test group
    :param variant_neg: int
        nunmber of negative events in the test group
    :return: float
        probability that Test group is better that Control group.
    """
    u1 = base_pos / (base_pos + base_neg)
    u2 = variant_pos / (variant_pos + variant_neg)
    var1 = base_pos * base_neg / (((base_pos + base_neg) ** 2) * (base_pos + base_neg + 1))
    var2 = variant_pos * variant_neg / (((variant_pos + variant_neg) ** 2) * (variant_pos + variant_neg + 1))
    return norm.cdf((u2 - u1) / np.sqrt(var1 + var2))


def analyze_sampl(base_pos, base_neg, variant_pos, variant_neg, N=1e6):
    """
    Bayesian AB test by sampling from posterior

    :param base_pos: int
        number of positive events in the control group
    :param base_neg: int
        number of negative events in the control group
    :param variant_pos: int
        number of positive events in the test group
    :param variant_neg: int
        nunmber of negative events in the test group
    :param N: int or tuple of ints, optional
                Number of samples to consider
    :return: float
        probability that Test group is better that Control group.
    """
    base_sim = np.random.beta(base_pos + 1, base_neg + 1, N)
    variant_sim = np.random.beta(variant_pos + 1, variant_neg + 1, N)
    return np.average(base_sim < variant_sim)


def visualize_joint(joint_data, p_success, p_failure, plot_range, output='plots/output_joint.png'):
    import matplotlib.colors as colors
    fig, ax = plt.subplots(1, 1)

    cmap = plt.cm.RdBu_r
    plt.imshow(joint_data, cmap=cmap, aspect='auto', interpolation='gaussian', alpha=0.2, extent=plot_range,
               norm=colors.PowerNorm(gamma=0.1))
    ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    ax.set_xlabel('Test group, p')
    ax.set_ylabel('Control group, p')

    plt.title('P(Test is better than Control) = %s%%' % (p_success * 100))
    plt.savefig(output)


def analyze_joint(base_pos, base_neg, variant_pos, variant_neg, N=1024, make_plot=True, output='plots/output_joint.png',
                  minp=None, maxp=None):
    """
    Bayesian AB test
    Read more: https://en.wikipedia.org/wiki/Joint_probability_distribution
    http://cdn2.hubspot.net/hubfs/310840/VWO_SmartStats_technical_whitepaper.pdf

    Adjust N, minp, maxp.

    :param base_pos: int
        number of positive events in the control group
    :param base_neg: int
        number of negative events in the control group
    :param variant_pos: int
        number of positive events in the test group
    :param variant_neg: int
        nunmber of negative events in the test group
    :param N: int or tuple of ints, optional
                Number of samples to consider
    :param output: str, optional
        file name to save the plot
    :param minp: float, optional
        min p
    :param maxp: float, optional
        max p
    :return: list(float, float)
        [probability that Test group is better that Control group,
        probability that Test group is worse that Control group]
    """
    if minp is None or maxp is None:
        m = beta.mean(base_pos + 1, base_neg + 1)
        minp = round(m - (0.01 * m), 3)
        maxp = round(m + (0.01 * m), 3)

    base_x = [beta.pdf(i, base_pos + 1, base_neg + 1) for i in np.linspace(minp, maxp, N)]
    variant_y = [beta.pdf(i, variant_pos + 1, variant_neg + 1) for i in np.linspace(minp, maxp, N)]

    joint = np.array([[i * j for i in variant_y] for j in base_x])
    joint_sum = np.sum(joint)

    sum_below_diagonal = np.sum([joint[x][y] for x in range(N) for y in range(N) if x < y])
    sum_above_diagonal = np.sum([joint[x][y] for x in range(N) for y in range(N) if x > y])
    p_success = sum_below_diagonal / joint_sum
    p_failure = sum_above_diagonal / joint_sum

    if make_plot:
        visualize_joint(joint, p_success, p_failure, (minp, maxp, minp, maxp), output=output)

    return p_success, p_failure, joint, (minp, maxp, minp, maxp)


def g_test(a, b, c, d):
    '''
    G-test for 2x2 contingency table
    contingency table
            failures    successes
    base       a            b
    variant    c            d
    '''

    def flnf(f):
        return f * np.log(f) if f > 0.5 else 0

    row1 = a + b
    row2 = c + d
    col1 = a + c
    col2 = b + d

    total = flnf(a + b + c + d)
    cell_totals = flnf(a) + flnf(b) + flnf(c) + flnf(d)
    row_totals = flnf(row1) + flnf(row2)
    col_totals = flnf(col1) + flnf(col2)

    return 2 * (cell_totals + total - (row_totals + col_totals))


def g_to_p(g, df):
    assert g >= 0, g
    return float(1 - chi2.sf(g, df))
