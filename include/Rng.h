#ifndef RNG_H
#define RNG_H

#include <random>
#include <algorithm>
#include <vector>

class Rng {
public:
    static Rng& getInstance() {
        static Rng instance;
        return instance;
    }

    void setSeed(unsigned int seed) {
        engine.seed(seed);
    }

    // [0, 1) real distribution
    double uniform01() {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(engine);
    }

    // [min, max] real distribution
    double uniformReal(double min, double max) {
        std::uniform_real_distribution<double> dist(min, max);
        return dist(engine);
    }

    // [min, max] integer distribution
    int uniformInt(int min, int max) {
        if (min > max) return min;
        std::uniform_int_distribution<int> dist(min, max);
        return dist(engine);
    }

    // Normal distribution
    double normal(double mean, double stddev) {
        std::normal_distribution<double> dist(mean, stddev);
        return dist(engine);
    }

    // Shuffle a vector
    template<typename T>
    void shuffle(std::vector<T>& vec) {
        std::shuffle(vec.begin(), vec.end(), engine);
    }

    std::mt19937& getEngine() {
        return engine;
    }

private:
    Rng() : engine(std::random_device{}()) {}
    std::mt19937 engine;

    // Disallow copy/assignment
    Rng(const Rng&) = delete;
    Rng& operator=(const Rng&) = delete;
};

#endif // RNG_H
