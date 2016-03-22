#! /usr/bin/env python3

import itertools, collections
import numpy as np
import scipy.special
import math
import scipy.optimize
from matplotlib import pyplot as plt

#*** num_balls = 5, num_draws = 8
#
#    100800 --> (1, 2, 1) [100800]
#     67200 --> (3, 1, 1) [67200]
#     50400 --> (2, 3) [50400]
#     50400 --> (2, 1, 0, 1) [50400]
#     33600 --> (2, 0, 2) [33600]
#     16800 --> (1, 0, 1, 1) [16800]
#     16800 --> (0, 1, 2) [16800]
#     12600 --> (0, 4) [12600]
#     12600 --> (0, 2, 0, 1) [12600]
#     10080 --> (1, 1, 0, 0, 1) [10080]
#      8400 --> (4, 0, 0, 1) [8400]
#      6720 --> (3, 0, 0, 0, 1) [6720]
#      1680 --> (2, 0, 0, 0, 0, 1) [1680]
#      1120 --> (0, 0, 1, 0, 1) [1120]
#       700 --> (0, 0, 0, 2) [700]
#       560 --> (0, 1, 0, 0, 0, 1) [560]
#       160 --> (1, 0, 0, 0, 0, 0, 1) [160]
#         5 --> (0, 0, 0, 0, 0, 0, 0, 1) [5]
#---------------
#    390625 total

def brute_force(num_balls, num_draws):

    balls = list(range(num_balls))

    distributions = collections.Counter()

    for draw in itertools.product(*[balls for d in range(num_draws)]):
        count_duplicates = collections.Counter(collections.Counter(draw).values())
        distribution = tuple(count_duplicates[i] for i in range(1, 1 + max(count_duplicates)))
        distributions[distribution] += 1

    return distributions

def calc_count(dd, n):
    us = np.sum(dd)
    ts = np.dot(dd, np.arange(1, len(dd) + 1))

    count = math.factorial(ts) // np.product([math.factorial(d) * math.factorial(i) ** d for (i, d) in enumerate(dd, 1)]) * (math.factorial(n) // math.factorial(n - us))

    return count

def test_count():

    err = 0

    for num_balls in range(0, 8 + 1):
        for num_draws in range(1, 8 + 1):

            print("*** num_balls = {}, num_draws = {}".format(num_balls, num_draws))
            print()

            bf = brute_force(num_balls, num_draws)

            total_count = 0
            for (dd, count) in bf.most_common():
                calc = calc_count(dd, num_balls)
                print("{:10} --> {} [{}]".format(count, dd, calc))
                if (count != calc):
                    print("YIKES!")
                    err += 1
                total_count += count

            print("---------------")
            print("{:10} total".format(total_count))
            print()

            assert total_count == (num_balls ** num_draws)

    assert err == 0

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

#1 1963
#2 291
#3 17
#4 2

def main():
    dd = [1974, 295, 17, 2]
    xy = []
    for num_balls in np.arange(9000, 9601, 5):
        score = montecarlo(dd, num_balls, 500)
        print(num_balls, score)
        xy.append((num_balls, score))
    xy = np.array(xy)
    plt.plot(xy[:,0], xy[:,1], "*-")
    plt.show()

def enumerate_dd(sofar, remaining):
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

def calc_ts_counts(ts, n):
    for dd in enumerate_dd([], ts):
        count = calc_count(dd, n)
        print(count, dd)


if __name__ == "__main__":
    #main()
    #print(calc_count([2,1,0,1], 5))
    #print(H(11.3))
    #MaximumLikelihoodEstimator([1974,295,17,2])
    #for n in range(1, 20):
    #    count = len(list(enumerate_dd([], n)))
    #    print(n, count)

    calc_ts_counts(10, 30)
