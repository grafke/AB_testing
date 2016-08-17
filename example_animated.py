from __future__ import division
import numpy as np
import sys
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
from ab_testing_methods import analyze_joint


def explode_ct(p, n, b):
    """
    [Helper function used in simulation]
    Simulate data by creating an array of observations.
        [[1, 0, 1, 0],
        [1, 0, 1, 0]]

    :param p: int
        number of successes
    :param n: int
        number of failures
    :param b: int
        number of buckets
    :return:
        array
    """
    return np.array_split(np.random.permutation(np.r_[[0] * n + [1] * p]), b)


def pos_neg_counts(x):
    """
    [Helper function used in simulation]
    Counts 1 and 0 in x

    :param x: array
        observations
    :return: list
        [number of positive events, number of negative events]
    """
    return np.count_nonzero(x), len(x) - np.count_nonzero(x)


def save_animation(ani, output_format='gif', output='plots/AB_animation.', author='Me'):
    """
    :param animation: animation
    :param output_format: str, optional
        gif or mp4
    :param output: str, optional
    :param author: str, optional
    """
    supported_formats = {'gif': 'imagemagick',
                         'mp4': 'ffmpeg'}

    Writer = animation.writers[supported_formats.get(output_format, 'gif')]
    writer = Writer(metadata=dict(artist=author))
    ani.save(output + output_format, writer=writer)


def create_animation(verbose=True):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)  # plot joint
    ax2 = fig.add_subplot(1, 2, 2)  # plot p(success)

    list_of_p_success = []
    ims = []
    for i in range(1, buckets + 1):
        # Create synthetic data
        base_observed_pos, base_observed_neg = pos_neg_counts(np.concatenate(base_data[:i]).ravel())
        variant_observed_pos, variant_observed_neg = pos_neg_counts(np.concatenate(variant_data[:i]).ravel())

        # Run AB test
        p_success, p_failure, joint_dist_data, plot_lim = analyze_joint(base_observed_pos, base_observed_neg,
                                                                        variant_observed_pos, variant_observed_neg,
                                                                        make_plot=False)

        list_of_p_success.append(p_success)

        # Plot joint distribution
        im1 = ax1.imshow(joint_dist_data, cmap=plt.cm.RdBu_r, aspect='auto', interpolation='gaussian', alpha=0.2,
                         extent=plot_lim, norm=colors.PowerNorm(gamma=0.1), animated=True)
        ax1.plot([0, 1], [0, 1], transform=ax1.transAxes)
        ax1.set_xlabel('Test group, p')
        ax1.set_ylabel('Control group, p')

        # Plot p(success)
        im2, = ax2.plot(range(1, i + 1), list_of_p_success)
        ax2.set_ylim(0, 1)
        ax2.set_title('P(Test is better than Control) = %s%%' % (p_success * 100))

        ims.append([im1, im2])

        if verbose:
            sys.stdout.write('%s%% complete\n' % round(100 * i / (buckets + 1), 2))
            sys.stdout.flush()

    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=False, repeat_delay=3000)

    return ani


if __name__ == '__main__':
    base_pos = 2643146
    base_neg = 13980139
    variant_pos = 2646705
    variant_neg = 13971736

    # experiment duration seconds
    t = 1209600

    # Observation interval in seconds
    # i.e. how often to peek in data
    delta_t = 86400

    # data_buckets
    buckets = int(t / delta_t)

    # sample data
    base_data = explode_ct(base_pos, base_neg, buckets)
    variant_data = explode_ct(variant_pos, variant_neg, buckets)

    anim = create_animation()

    save_animation(anim)
