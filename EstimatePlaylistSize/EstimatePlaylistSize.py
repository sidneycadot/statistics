#! /usr/bin/env python3

# When monitoring an Internet Radio station, we notice after a while that we hear some
# songs that we heard before. Can this information be used to assess the total size of the playlist?
#
# We model this situation as a vase (the playlist) containing balls (the songs). The balls are not sequentially
# numbered, but they are uniquely identifiable; perhaps by color, or by a unique text written on their outside
# (a combination of artist and song name, for example?). Each identification occurs only on one ball in the vase.
#
# We model the process of song selection as random, memoryless draws from the vase, with replacement.
#
# It is useful to realize that this model is not an accurate representation of the way song selection is
# performed on an actual radio stations. There are at least three important differences:
#
# (1) No balls enter or leave the vase. On actual radio stations, there is a certain 'turnaround' of songs
#     that enter or leave the playlist. This is most prominent on stations that specialize in popular hits,
#     but even stations with a relatively stable poopulation of songs (such as 'Golden Oldie' stations) will
#     have their playlists updated once in a while, albeit much less frequent.
#
# (2) All balls have an identical probability of being drawn.
#
# (3) The drawing process is memoryless.
#
# Our model states that all balls have an identical probability of being drawn.
#
# "dd" vectors
# ------------


import itertools, collections, time
import numpy as np
import scipy.special
import math
import scipy.optimize
from matplotlib import pyplot as plt

# *** num_balls = 5, num_draws = 8
#
#     100800 --> (1, 2, 1)
#      67200 --> (3, 1, 1)
#      50400 --> (2, 3)
#      50400 --> (2, 1, 0, 1)
#      33600 --> (2, 0, 2)
#      16800 --> (1, 0, 1, 1)
#      16800 --> (0, 1, 2)
#      12600 --> (0, 4)
#      12600 --> (0, 2, 0, 1)
#      10080 --> (1, 1, 0, 0, 1)
#       8400 --> (4, 0, 0, 1)
#       6720 --> (3, 0, 0, 0, 1)
#       1680 --> (2, 0, 0, 0, 0, 1)
#       1120 --> (0, 0, 1, 0, 1)
#        700 --> (0, 0, 0, 2)
#        560 --> (0, 1, 0, 0, 0, 1)
#        160 --> (1, 0, 0, 0, 0, 0, 1)
#          5 --> (0, 0, 0, 0, 0, 0, 0, 1)
# ---------------
#     390625 total

def brute_force(num_balls, num_draws):
    """ Enumerate all possible sequences of 'num_draws' draws, from a vase with 'num_balls' balls,
        with replacement.
        Summarize each sequence as a 'dd' vector; and count the different vectors as they occur.

        A 'dd' vector is defined as follows: dd[i] counts the number of balls that were seen
        precisely (i + 1) times. E.g., in an experiment with 8 draws, a dd vector of [2, 1, 0, 1]
        means that 2 balls were seen once, 1 ball was seen twice, and 1 ball was seen four times.
    """
    balls = list(range(num_balls))

    distributions = collections.Counter()

    for draw in itertools.product(*[balls for d in range(num_draws)]):
        count_duplicates = collections.Counter(collections.Counter(draw).values())
        distribution = tuple(count_duplicates[i] for i in range(1, 1 + max(count_duplicates)))
        distributions[distribution] += 1

    return distributions

def calc_count(dd, n):
    """ Given a particular 'dd' vector and a value 'n' equal to the number of different balls in the vase,
        This function returns the number of different draws that result in the 'dd' vectors given.
    """

    us = np.sum(dd)
    ts = np.dot(dd, np.arange(1, len(dd) + 1))

    return math.factorial(ts) // np.product([math.factorial(d) * math.factorial(i) ** d for (i, d) in enumerate(dd, 1)]) * (math.factorial(n) // math.factorial(n - us))

def test_calc_count():
    """ Verify that the result of calc_count() is identical to the count we obtain from
        a brute-force enumeration of all draw sequences.
    """
    for maxval in range(1, 101):
        for num_balls in range(0, maxval + 1):
            for num_draws in range(1, maxval + 1):

                if not ((num_balls == maxval) or (num_draws == maxval)):
                    continue

                print("*** num_balls = {}, num_draws = {}".format(num_balls, num_draws))
                print()

                t1 = time.time()
                bf = brute_force(num_balls, num_draws)
                t2 = time.time()

                duration = (t2 - t1)

                total_count = 0
                for (dd, count) in bf.most_common():
                    calc = calc_count(dd, num_balls)
                    print("{:10} --> {}".format(count, dd))
                    assert (count == calc)
                    total_count += count

                print("---------------")
                print("{:10} total ; {} distributions ({:.3f} s)".format(total_count, len(bf), duration))
                print()

                assert total_count == (num_balls ** num_draws)


def enumerate_dd(sofar, remaining):
    """ This code is incorrect!
        It generates solutions even when the number of different balls is too low.
    """
    if remaining == 0:
        yield sofar
    curr = len(sofar) + 1
    if remaining < curr:
        return
    p = 0
    while remaining >= 0:
        yield from enumerate_dd(sofar + [p], remaining)
        p += 1
        remaining -= curr

def montecarlo(dd, num_balls, num_repeats):

    num_draws = np.dot(dd, np.arange(1, len(dd) + 1))

    dd_count = calc_count(dd, num_balls)

    smaller = 0.0

    for rep in range(num_repeats):
        mc_draw = np.random.choice(num_balls, num_draws)
        mc_count_duplicates = collections.Counter(collections.Counter(mc_draw).values())
        mc_dd = tuple(mc_count_duplicates[i] for i in range(1, 1 + max(mc_count_duplicates)))
        #print(mc_dd)
        mc_count = calc_count(mc_dd, num_balls)
        if mc_count < dd_count:
            smaller += 1.0
        elif mc_count == dd_count:
            smaller += 0.5

    score = smaller / num_repeats

    return score

def main2():
    dd = [1974, 295, 17, 2]
    xy = []
    for num_balls in np.arange(9000, 9601, 5):
        score = montecarlo(dd, num_balls, 500)
        print(num_balls, score)
        xy.append((num_balls, score))
    xy = np.array(xy)
    plt.plot(xy[:,0], xy[:,1], "*-")
    plt.show()

def calc_ts_counts(ts, n):
    for dd in enumerate_dd([], ts):
        count = calc_count(dd, n)
        print(count, dd)

def main():
    #test_calc_count()
    for num_draws in [9]:
        for dd in enumerate_dd([], num_draws):
            print(dd)


if __name__ == "__main__":
    main()
    #main()
    #print(calc_count([2,1,0,1], 5))
    #print(H(11.3))
    #MaximumLikelihoodEstimator([1974,295,17,2])
    #for n in range(1, 20):
    #    count = len(list(enumerate_dd([], n)))
    #    print(n, count)
    #calc_ts_counts(10, 30)
