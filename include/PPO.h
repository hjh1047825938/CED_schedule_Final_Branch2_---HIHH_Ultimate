#ifndef PPO_H
#define PPO_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

class MultiMet;

struct PPOConfig {
    int state_dim = 7;
    int hidden_dim = 128;
    int episodes = 10000;
    int update_every = 64;
    int ppo_epochs = 4;
    int minibatch_size = 64;
    double gamma = 0.99;
    double gae_lambda = 0.95;
    double clip_eps = 0.2;
    double actor_lr = 3e-4;
    double critic_lr = 3e-4;
    double entropy_coef = 0.01;
    double value_coef = 0.5;
    double max_grad_norm = 0.5;
    double beta_min = 1e-3;
};

struct PPORunResult {
    std::vector<double> best_curve;
    double final_best = 0.0;
};

class PPOScheduler {
public:
    PPOScheduler(MultiMet* solver, const PPOConfig& cfg, uint32_t seed);
    PPORunResult Train();

private:
    struct Transition {
        std::array<double, 7> state{};
        std::vector<double> action;
        std::vector<double> alpha;
        std::vector<double> beta;
        double log_prob = 0.0;
        double reward = 0.0;
        double value = 0.0;
        bool done = true;
    };

    struct Layer {
        int in_dim = 0;
        int out_dim = 0;
        std::vector<double> w;
        std::vector<double> b;
        std::vector<double> gw;
        std::vector<double> gb;
        std::vector<double> mw;
        std::vector<double> vw;
        std::vector<double> mb;
        std::vector<double> vb;
    };

    struct ForwardCache {
        std::vector<double> x;
        std::vector<double> z1;
        std::vector<double> a1;
        std::vector<double> z2;
        std::vector<double> a2;
        std::vector<double> z3;
    };

    struct ActorOutput {
        std::vector<double> alpha;
        std::vector<double> beta;
        std::vector<double> alpha_raw;
        std::vector<double> beta_raw;
        double log_prob = 0.0;
        double entropy = 0.0;
    };

    struct RunningStats {
        int count = 0;
        double sum_fit = 0.0;
        double last_fit = 0.0;
        double best_fit = 0.0;
        double last_reward = 0.0;
        double last_best_improve = 0.0;
        std::vector<double> last_action;
        std::vector<double> prev_action;
        std::vector<double> best_action;
        std::vector<int> success_window;
        int success_ptr = 0;
        int success_sum = 0;
    };

    MultiMet* solver_ = nullptr;
    PPOConfig cfg_{};
    uint32_t seed_ = 1;
    int action_dim_ = 0;

    Layer actor_l1_{};
    Layer actor_l2_{};
    Layer actor_l3_{};

    Layer critic_l1_{};
    Layer critic_l2_{};
    Layer critic_l3_{};

    std::mt19937 rng_{};
    uint64_t adam_step_actor_ = 0;
    uint64_t adam_step_critic_ = 0;

    std::vector<Transition> buffer_;
    RunningStats stats_{};

    void InitNetwork();
    void InitLayer(Layer& layer, int in_dim, int out_dim);

    ForwardCache ForwardMLP(const std::array<double, 7>& state,
                            const Layer& l1,
                            const Layer& l2,
                            const Layer& l3) const;

    ActorOutput ActorForward(const std::array<double, 7>& state,
                             const std::vector<double>* fixed_action,
                             ForwardCache* cache) const;

    double CriticForward(const std::array<double, 7>& state, ForwardCache* cache) const;

    std::array<double, 7> BuildState(int episode_idx) const;
    double ComputeActionDiversity() const;

    void UpdatePPO();
    void ComputeGAE(std::vector<double>& advantages, std::vector<double>& returns) const;

    static double Softplus(double x);
    static double Sigmoid(double x);
    static double Digamma(double x);
    static double Trigamma(double x);
    static double BetaLogProb(double x, double a, double b);
    static double BetaEntropy(double a, double b);

    double SampleBeta(double a, double b);
    double Random01();

    void ZeroGradLayer(Layer& layer);
    void BackwardLayer(const Layer& layer,
                       Layer& grad_layer,
                       const std::vector<double>& input,
                       const std::vector<double>& grad_out,
                       std::vector<double>& grad_in);

    void AdamStep(Layer& layer, double lr, uint64_t step);
    double GlobalGradNorm(const std::vector<Layer*>& layers) const;
    void ScaleGrad(const std::vector<Layer*>& layers, double scale);
};

#endif
