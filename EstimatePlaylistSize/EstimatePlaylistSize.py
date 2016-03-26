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
# dd-tuples
# ---------
#
# In the code below, have use is made of 'dd-tuples'. "dd" stands for "duplicate distribution".
# Given a sequence of draws from a vase, the dd-tuple summarizes the observed balls in terms of duplicates seen.
#
# As a definition: the i'th element of the tuple counts the number of balls that were seen (i + 1) times during
# the sequence of draws.
#
# For example, suppose we draw the following sequence of balls: [ C, A, C, C, D, B, C, B ]
#
#   We observed 2 balls only once: A and D.
#   We observed 1 ball twice: B
#   We observed 0 balls three times.
#   We observed 1 ball four times: C.
#
# These observations are summarized using the dd-tuple (2, 1, 0, 1).
#
# Note that adding zeroes to the end of a dd-tuple does not alter its meaning. By convention, we end the tuple
#   at its last non-zero entry.
#
# From a given dd-tuple, we can extract two important pieces of information:
#
#   (1) sum(dd)                      equals the number of different balls observed, a.k.a. num_unique.
#   (2) dd . [1, 2, 3, 4, 5, 6, ...] i.e, the sum of dd values multiplied by the increasing integer range starting at 1,
#                                         equals the number of balls observed in total (including duplicates), a.k.a. num_draws.
#
# dd-distributions
# ----------------
#
# For a given experimental setup where we know the number of balls in the vase (num_balls) and the number of balls to be drawn (num_draws),
# there are (num_balls ** num_draws) possible equiprobable sequences of drawn balls.
#
# If we determine the d-tuple for each of the possible sequences, we will see that any given dd-tuple can be obtained in multiple ways.
# The number of draw sequences that give rise to a certain dd-tuple is directly proportional to its probability.

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
    """ Enumerate all possible sequences of 'num_draws' draws, from a vase containing 'num_balls' balls,
        with replacement, by brute force.
        Summarize each draw sequence as a dd-tuple, and count the different dd-tuples as they occur.
    """
    balls = list(range(num_balls))

    dd_distribution = collections.Counter()

    for draw in itertools.product(*[balls for d in range(num_draws)]):
        count_duplicates = collections.Counter(collections.Counter(draw).values())
        dd = tuple(count_duplicates[i] for i in range(1, 1 + max(count_duplicates)))
        dd_distribution[dd] += 1

    return dd_distribution

def calc_dd_count(dd, num_balls):
    """ Given a particular dd-tuple and num_balls (the number of different balls in the vase),
          return the number of different draws that result in the dd-tuple given.
    """

    num_unique = np.sum(dd)                             # number of unique balls seen, according to the dd-tuple.
    num_draws  = np.dot(dd, np.arange(1, len(dd) + 1))  # number of balls seen (a.k.a., number of draws), according to the dd-tuple.

    return math.factorial(num_draws) // np.product([math.factorial(d) * math.factorial(i) ** d for (i, d) in enumerate(dd, 1)]) * \
                  (math.factorial(num_balls) // math.factorial(num_balls - num_unique))

def test_calc_dd_count():
    """ Verify that the result of calc_dd_count() is identical to the count we obtain from
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
                    calc = calc_dd_count(dd, num_balls)
                    print("{:10} --> {}".format(count, dd))
                    assert (count == calc)
                    total_count += count

                print("---------------")
                print("{:10} total ; {} distributions ({:.3f} s)".format(total_count, len(bf), duration))
                print()

                assert total_count == (num_balls ** num_draws)

def enumerate_dd_rec(sofar, balls_unseen, draws_remaining):
    """ Generate all possible dd-tuples recursively.
    """
    if balls_unseen >= 0 and draws_remaining == 0:
        yield tuple(sofar)

    curr = len(sofar) + 1

    if draws_remaining < curr:
        return

    p = 0
    while balls_unseen >= 0 and draws_remaining >= 0:
        yield from enumerate_dd_rec(sofar + [p], balls_unseen, draws_remaining)
        p += 1
        balls_unseen -= 1
        draws_remaining -= curr

def enumerate_dd(num_balls, num_draws):
    """ Generate all possible dd-tuples.
    """
    yield from enumerate_dd_rec([], num_balls, num_draws)

def fast_enumerate(num_balls, num_draws):
    """ This is functionally equivalent to br
    """
    dd_distribution = collections.Counter()
    for dd in enumerate_dd(num_balls, num_draws):
        dd_distribution[dd] = calc_dd_count(dd, num_balls)
    return dd_distribution

def test_fast_enumerate():
    for num_balls in range(0, 8):
        for num_draws in range(1, 8):
            bf = brute_force(num_balls, num_draws)
            fe = fast_enumerate(num_balls, num_draws)
            print(num_balls, num_draws)
            assert bf == fe

def examine():
    for num_balls in range(30):
        dd_distribution = fast_enumerate(num_balls, 3)
        cLess = 0
        cEqual = 0
        cMore = 0
        refcount = dd_distribution[(3, )]
        for dd in dd_distribution:
            if dd_distribution[dd] < refcount:
                cLess += calc_dd_count(dd, num_balls)
        print(num_balls, cLess)

def check_tree():
    dd_distribution = fast_enumerate(6, 9)
    for dd in dd_distribution:
        print(dd, dd_distribution[dd])

def montecarlo(dd, num_balls, num_repeats):

    num_draws = np.dot(dd, np.arange(1, len(dd) + 1))

    dd_count = calc_dd_count(dd, num_balls)

    smaller = 0.0

    for rep in range(num_repeats):
        mc_draw = np.random.choice(num_balls, num_draws)
        mc_count_duplicates = collections.Counter(collections.Counter(mc_draw).values())
        mc_dd = tuple(mc_count_duplicates[i] for i in range(1, 1 + max(mc_count_duplicates)))
        #print(mc_dd)
        mc_count = calc_dd_count(mc_dd, num_balls)
        if mc_count < dd_count:
            smaller += 1.0
        elif mc_count == dd_count:
            smaller += 0.5

    score = smaller / num_repeats

    return score

def drive_monte_carlo():
    dd = [1639, 859, 69, 20, 1]
    xy = []
    for num_balls in np.arange(4980, 5000, 10):
        score = montecarlo(dd, num_balls, 10000)
        print(num_balls, score)
        xy.append((num_balls, score))
    xy = np.array(xy)
    plt.plot(xy[:,0], xy[:,1], "*")
    plt.show()

def calc_ts_counts(ts, n):
    for dd in enumerate_dd([], 8, 10):
        count = calc_dd_count(dd, n)
        print(count, dd)

def main():

    #for dd in enumerate_dd([], ts):
    #    count = calc_count(dd, n)

    #test_calc_count()
    #test_fast_enumerate()

    check_tree()

if __name__ == "__main__":
    main()
