
#include <cassert>
#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <iomanip>

double calculate_log_probability(const std::vector<unsigned> & dd, unsigned num_balls)
{
    unsigned num_unique = 0;
    unsigned num_draws = 0;
    double log_denom = 0;

    for (unsigned i = 1; i <= dd.size(); ++i)
    {
        const double dd_i = dd[i - 1];

        num_unique += dd_i;
        num_draws += dd_i * i;

        log_denom += lgamma(1 + dd_i) + dd_i * lgamma(1 + i);
    }

    assert(num_balls >= num_unique);

    return lgamma(1 + num_draws) - log_denom + lgamma(1 + num_balls) - lgamma(1 + num_balls - num_unique) - num_draws * log(num_balls);
}

class BallDrawer
{
    public:

        BallDrawer(const unsigned num_balls, const unsigned seed) : rgen(seed), distribution(0, num_balls - 1)
        {
            // Empty body.
        }

        void draw_balls(std::vector<unsigned> & dd, const unsigned num_draws)
        {
            dd.clear(); // Reset size to 0.

            for (unsigned draw = 0; draw < num_draws; ++draw)
            {
                const unsigned ball = distribution(rgen);

                unsigned bin; // The selected bin to draw a song from.
                unsigned sum = 0; // Cumulative sum of dd entries up to dd[bin].

                for (bin = 0; bin < dd.size(); ++bin)
                {
                    sum += dd[bin];

                    if (ball < sum)
                    {
                        break;
                    }
                }

                if (bin == dd.size())
                {
                    bin = 0; // Add ball to the first dd bin.
                }
                else
                {
                    --dd[bin]; // Remove ball from selected bin
                    ++bin; // Add ball to the next bin.
                }

                if (bin == dd.size())
                {
                    dd.resize(bin + 1);
                }

                ++dd[bin];

            }
        }

    private:

        std::mt19937 rgen;
        std::uniform_int_distribution<unsigned> distribution;
};

void monte_carlo(const unsigned num_balls, const unsigned num_draws, const unsigned num_repeats, const unsigned seed)
{
    BallDrawer drawer(num_balls, seed);

    std::vector<unsigned> dd; // Vector where we simulate draws in. 
    dd.resize(num_draws);     // The 'num_draws' size is worst case.

    for (unsigned rep = 0; rep < num_repeats; ++rep)
    {
        drawer.draw_balls(dd, num_draws);

        const double log_probability = calculate_log_probability(dd, num_balls);

        if (1)
        {

            std::cout << "dd: num_balls " << num_balls << " num_draws " << num_draws << " log_probability " << log_probability << " --> ";
            for (unsigned i = 0; i < dd.size(); ++i)
            {
                if (i != 0)
                {
                    std::cout << ", ";
                }
                std::cout << dd[i];
            }
            std::cout << std::endl;
        }
    }
}

double monte_carlo_pvalue(const std::vector<unsigned> & ref_dd, const unsigned num_balls, const unsigned num_repeats, const unsigned seed)
{
    BallDrawer drawer(num_balls, seed);

    const double ref_dd_log_probability = calculate_log_probability(ref_dd, num_balls);

    unsigned num_draws = 0;

    for (unsigned i = 1; i <= ref_dd.size(); ++i)
    {
        const double ref_dd_i = ref_dd[i - 1];

        num_draws += ref_dd_i * i;
    }

    std::vector<unsigned> dd; // Vector where we simulate draws in. 
    dd.resize(num_draws);     // The 'num_draws' size is worst case.

    double score = 0.0;

    for (unsigned rep = 0; rep < num_repeats; ++rep)
    {
        drawer.draw_balls(dd, num_draws);

        const double dd_log_probability = calculate_log_probability(dd, num_balls);

        const double log_probability_difference = ref_dd_log_probability - dd_log_probability;

        if (fabs(log_probability_difference) < 1e-10)
        {
            // ref_dd and mc_dd are equally probable
            score += 0.5;
        }
        else if (log_probability_difference > 0)
        {
            // ref_dd is more probably than mc_dd
            score += 1.0;
        }
    }

    return score / num_repeats; // estimated p-value
}

int main()
{
    unsigned seed = 0;

    //const unsigned dd[15] = {0, 0, 0, 0, 1, 1, 0, 1, 2, 1, 0, 1, 2, 0, 1};
    //const unsigned dd[5] = {1157, 1415, 281, 32, 6};
    //const unsigned dd[5] = {0,1,0,0,1};
    //const unsigned dd[7] = {1439, 901, 379, 104, 29, 4, 2};

    //for (unsigned num_balls = 10; num_balls <= 15; num_balls += 1)
    //{
    //    const double p_value = monte_carlo(dd, 15, num_balls, 10000, ++seed);
    //    std::cout << num_balls << "  " << p_value << std::endl;
    //}

    const std::vector<unsigned> dd_100_200({27, 22, 25, 8, 2, 2});

    //monte_carlo(100, 200, 100000, seed);
    //double p_value = monte_carlo_pvalue(dd_100_200, 100, 100000, seed);
    std::cout << p_value << std::endl;
    return 0;
}
