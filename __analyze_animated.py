"""
Work in progress
"""
from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from scipy.stats import beta
import matplotlib.animation as animation

# Outcome
base_pos = 2643146
base_neg = 13980139
variant_pos = 2646705
variant_neg = 13971736

# experiment duration seconds
t = 1209600

# interval seconds
delta_t = 86400 / 1

# data_buckets
buckets = int(t / delta_t)


def explode_ct(p, n, b):
    return np.array_split(np.random.permutation(np.r_[[0] * n + [1] * p]), b)


def pos_neg_counts(x):
    return np.count_nonzero(x), len(x) - np.count_nonzero(x)


def plot_range(p, n):
    m = beta.mean(p + 1, n + 1)
    return round(m - (0.01 * m), 3), round(m + (0.01 * m), 3)


base_data = explode_ct(base_pos, base_neg, buckets)
variant_data = explode_ct(variant_pos, variant_neg, buckets)


def analyze_joint((base_pos, base_neg), (variant_pos, variant_neg), N=1024, minp=None, maxp=None):
    if minp is None or maxp is None:
        minp, maxp = plot_range(base_pos, base_neg)

    base_x = [beta.pdf(i, base_pos + 1, base_neg + 1) for i in np.linspace(minp, maxp, N)]
    variant_y = [beta.pdf(i, variant_pos + 1, variant_neg + 1) for i in np.linspace(minp, maxp, N)]

    joint = np.array([[i * j for i in variant_y] for j in base_x])
    joint_sum = np.sum(joint)

    sum_below_diagonal = np.sum([joint[x][y] for x in range(N) for y in range(N) if x < y])
    sum_above_diagonal = np.sum([joint[x][y] for x in range(N) for y in range(N) if x > y])
    p_success = sum_below_diagonal / joint_sum
    p_failure = sum_above_diagonal / joint_sum

    return p_failure, p_success, joint, (minp, maxp, minp, maxp)


def main(output_format='gif'):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    list_of_p_success = []
    ims = []
    for i in range(1, buckets + 1):
        bp, bn = pos_neg_counts(np.concatenate(base_data[:i]).ravel())
        vp, vn = pos_neg_counts(np.concatenate(variant_data[:i]).ravel())
        p_f, p_s, j, r = analyze_joint((bp, bn), (vp, vn))
        list_of_p_success.append(p_s)
        im1 = ax1.imshow(j, cmap=plt.cm.RdBu_r, aspect='auto', interpolation='gaussian', alpha=0.2,
                         extent=r, norm=colors.PowerNorm(gamma=0.1), animated=True)
        ax1.plot([0, 1], [0, 1], transform=ax1.transAxes)
        ax1.set_xlabel('Test group, p')
        ax1.set_ylabel('Control group, p')
        im2, = ax2.plot(range(1, i + 1), list_of_p_success)
        ax2.set_title('P(Test is better than Control) = %s%%' % (p_s * 100))
        ims.append([im1, im2])
        print('%s%% complete' % round(100 * i / (buckets + 1), 2))

    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=False, repeat_delay=3000)

    if output_format == 'gif':
        Writer = animation.writers['imagemagick']
        writer = Writer(metadata=dict(artist='Me'))
        ani.save('plots/AB_animation.gif', writer=writer)
    elif output_format == 'mp4':
        Writer = animation.writers['ffmpeg']
        writer = Writer(metadata=dict(artist='Me'))
        ani.save('plots/AB_animation.mp4', writer=writer)
    else:
        plt.show()


if __name__ == '__main__':
    main()
