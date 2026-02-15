#include "PPO.h"

#include "Multimethod.h"
#include "Problems.h"
#include "Rng.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>

namespace {
constexpr double kEps = 1e-12;
constexpr double kPi = 3.14159265358979323846;
}

PPOScheduler::PPOScheduler(MultiMet* solver, const PPOConfig& cfg, uint32_t seed)
    : solver_(solver), cfg_(cfg), seed_(seed), rng_(seed) {
    action_dim_ = solver_ ? solver_->Nvar : 0;
    if (cfg_.minibatch_size <= 0) cfg_.minibatch_size = cfg_.update_every;
    if (cfg_.minibatch_size > cfg_.update_every) cfg_.minibatch_size = cfg_.update_every;
    if (cfg_.ppo_epochs < 1) cfg_.ppo_epochs = 1;
    stats_.success_window.assign(50, 0);
    stats_.last_action.assign(action_dim_, 0.5);
    stats_.prev_action.assign(action_dim_, 0.5);
    stats_.best_action.assign(action_dim_, 0.5);
    InitNetwork();
}

void PPOScheduler::InitLayer(Layer& layer, int in_dim, int out_dim) {
    layer.in_dim = in_dim;
    layer.out_dim = out_dim;
    layer.w.assign((size_t)in_dim * (size_t)out_dim, 0.0);
    layer.b.assign((size_t)out_dim, 0.0);
    layer.gw.assign((size_t)in_dim * (size_t)out_dim, 0.0);
    layer.gb.assign((size_t)out_dim, 0.0);
    layer.mw.assign((size_t)in_dim * (size_t)out_dim, 0.0);
    layer.vw.assign((size_t)in_dim * (size_t)out_dim, 0.0);
    layer.mb.assign((size_t)out_dim, 0.0);
    layer.vb.assign((size_t)out_dim, 0.0);

    const double limit = std::sqrt(6.0 / (double)(in_dim + out_dim));
    std::uniform_real_distribution<double> dist(-limit, limit);
    for (double& w : layer.w) w = dist(rng_);
    for (double& b : layer.b) b = 0.0;
}

void PPOScheduler::InitNetwork() {
    InitLayer(actor_l1_, cfg_.state_dim, cfg_.hidden_dim);
    InitLayer(actor_l2_, cfg_.hidden_dim, cfg_.hidden_dim);
    InitLayer(actor_l3_, cfg_.hidden_dim, 2 * action_dim_);

    InitLayer(critic_l1_, cfg_.state_dim, cfg_.hidden_dim);
    InitLayer(critic_l2_, cfg_.hidden_dim, cfg_.hidden_dim);
    InitLayer(critic_l3_, cfg_.hidden_dim, 1);
}

PPOScheduler::ForwardCache PPOScheduler::ForwardMLP(const std::array<double, 7>& state,
                                                    const Layer& l1,
                                                    const Layer& l2,
                                                    const Layer& l3) const {
    ForwardCache c;
    c.x.assign(state.begin(), state.end());
    c.z1.assign((size_t)l1.out_dim, 0.0);
    c.a1.assign((size_t)l1.out_dim, 0.0);
    c.z2.assign((size_t)l2.out_dim, 0.0);
    c.a2.assign((size_t)l2.out_dim, 0.0);
    c.z3.assign((size_t)l3.out_dim, 0.0);

    for (int o = 0; o < l1.out_dim; ++o) {
        double v = l1.b[o];
        const size_t base = (size_t)o * (size_t)l1.in_dim;
        for (int i = 0; i < l1.in_dim; ++i) v += l1.w[base + (size_t)i] * c.x[(size_t)i];
        c.z1[(size_t)o] = v;
        c.a1[(size_t)o] = (v > 0.0) ? v : 0.0;
    }

    for (int o = 0; o < l2.out_dim; ++o) {
        double v = l2.b[o];
        const size_t base = (size_t)o * (size_t)l2.in_dim;
        for (int i = 0; i < l2.in_dim; ++i) v += l2.w[base + (size_t)i] * c.a1[(size_t)i];
        c.z2[(size_t)o] = v;
        c.a2[(size_t)o] = (v > 0.0) ? v : 0.0;
    }

    for (int o = 0; o < l3.out_dim; ++o) {
        double v = l3.b[o];
        const size_t base = (size_t)o * (size_t)l3.in_dim;
        for (int i = 0; i < l3.in_dim; ++i) v += l3.w[base + (size_t)i] * c.a2[(size_t)i];
        c.z3[(size_t)o] = v;
    }

    return c;
}

PPOScheduler::ActorOutput PPOScheduler::ActorForward(const std::array<double, 7>& state,
                                                     const std::vector<double>* fixed_action,
                                                     ForwardCache* cache) const {
    ForwardCache local = ForwardMLP(state, actor_l1_, actor_l2_, actor_l3_);
    if (cache) *cache = local;

    ActorOutput out;
    out.alpha.assign((size_t)action_dim_, 0.0);
    out.beta.assign((size_t)action_dim_, 0.0);
    out.alpha_raw.assign((size_t)action_dim_, 0.0);
    out.beta_raw.assign((size_t)action_dim_, 0.0);

    for (int d = 0; d < action_dim_; ++d) {
        const double ar = local.z3[(size_t)d];
        const double br = local.z3[(size_t)action_dim_ + (size_t)d];
        out.alpha_raw[(size_t)d] = ar;
        out.beta_raw[(size_t)d] = br;
        out.alpha[(size_t)d] = Softplus(ar) + cfg_.beta_min;
        out.beta[(size_t)d] = Softplus(br) + cfg_.beta_min;
    }

    if (fixed_action) {
        for (int d = 0; d < action_dim_; ++d) {
            const double x = std::clamp((*fixed_action)[(size_t)d], 1e-6, 1.0 - 1e-6);
            const double a = out.alpha[(size_t)d];
            const double b = out.beta[(size_t)d];
            out.log_prob += BetaLogProb(x, a, b);
            out.entropy += BetaEntropy(a, b);
        }
    }

    return out;
}

double PPOScheduler::CriticForward(const std::array<double, 7>& state, ForwardCache* cache) const {
    ForwardCache local = ForwardMLP(state, critic_l1_, critic_l2_, critic_l3_);
    if (cache) *cache = local;
    return local.z3.empty() ? 0.0 : local.z3[0];
}

std::array<double, 7> PPOScheduler::BuildState(int episode_idx) const {
    std::array<double, 7> s{};
    const double mean_fit = (stats_.count > 0) ? (stats_.sum_fit / (double)stats_.count) : stats_.best_fit;
    const double base = std::max(std::fabs(stats_.best_fit), 1.0);
    const double improve = stats_.last_best_improve;
    const double progress = (cfg_.episodes > 1) ? (double)episode_idx / (double)(cfg_.episodes - 1) : 1.0;
    const double success_rate = stats_.success_window.empty()
                                    ? 0.0
                                    : (double)stats_.success_sum / (double)stats_.success_window.size();
    const double elite = (mean_fit > kEps) ? (stats_.best_fit / mean_fit) : 1.0;

    s[0] = mean_fit / base;
    s[1] = improve;
    s[2] = ComputeActionDiversity();
    s[3] = progress;
    s[4] = success_rate;
    s[5] = stats_.last_reward / base;
    s[6] = elite;

    for (double& v : s) {
        if (!std::isfinite(v)) v = 0.0;
        v = std::clamp(v, -10.0, 10.0);
    }
    return s;
}

double PPOScheduler::ComputeActionDiversity() const {
    if (stats_.last_action.empty() || stats_.prev_action.empty()) return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < stats_.last_action.size(); ++i) {
        sum += std::fabs(stats_.last_action[i] - stats_.prev_action[i]);
    }
    return sum / (double)stats_.last_action.size();
}

double PPOScheduler::Softplus(double x) {
    if (x > 30.0) return x;
    if (x < -30.0) return std::exp(x);
    return std::log1p(std::exp(x));
}

double PPOScheduler::Sigmoid(double x) {
    if (x >= 0.0) {
        const double z = std::exp(-x);
        return 1.0 / (1.0 + z);
    }
    const double z = std::exp(x);
    return z / (1.0 + z);
}

double PPOScheduler::Digamma(double x) {
    double result = 0.0;
    while (x < 6.0) {
        result -= 1.0 / x;
        x += 1.0;
    }
    const double inv = 1.0 / x;
    const double inv2 = inv * inv;
    result += std::log(x) - 0.5 * inv - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0)));
    return result;
}

double PPOScheduler::Trigamma(double x) {
    double result = 0.0;
    while (x < 6.0) {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    const double inv = 1.0 / x;
    const double inv2 = inv * inv;
    result += inv + 0.5 * inv2 + (1.0 / 6.0) * inv2 * inv - (1.0 / 30.0) * inv2 * inv2 * inv +
              (1.0 / 42.0) * inv2 * inv2 * inv2 * inv;
    return result;
}

double PPOScheduler::BetaLogProb(double x, double a, double b) {
    const double xc = std::clamp(x, 1e-6, 1.0 - 1e-6);
    return (a - 1.0) * std::log(xc) + (b - 1.0) * std::log(1.0 - xc) -
           (std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b));
}

double PPOScheduler::BetaEntropy(double a, double b) {
    const double lnB = std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
    return lnB - (a - 1.0) * Digamma(a) - (b - 1.0) * Digamma(b) + (a + b - 2.0) * Digamma(a + b);
}

double PPOScheduler::SampleBeta(double a, double b) {
    std::gamma_distribution<double> g1(a, 1.0);
    std::gamma_distribution<double> g2(b, 1.0);
    double x = g1(rng_);
    double y = g2(rng_);
    const double s = x + y;
    if (s <= 0.0 || !std::isfinite(s)) return 0.5;
    return std::clamp(x / s, 1e-6, 1.0 - 1e-6);
}

double PPOScheduler::Random01() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng_);
}

void PPOScheduler::ZeroGradLayer(Layer& layer) {
    std::fill(layer.gw.begin(), layer.gw.end(), 0.0);
    std::fill(layer.gb.begin(), layer.gb.end(), 0.0);
}

void PPOScheduler::BackwardLayer(const Layer& layer,
                                 Layer& grad_layer,
                                 const std::vector<double>& input,
                                 const std::vector<double>& grad_out,
                                 std::vector<double>& grad_in) {
    grad_in.assign((size_t)layer.in_dim, 0.0);
    for (int o = 0; o < layer.out_dim; ++o) {
        const double go = grad_out[(size_t)o];
        grad_layer.gb[(size_t)o] += go;
        const size_t base = (size_t)o * (size_t)layer.in_dim;
        for (int i = 0; i < layer.in_dim; ++i) {
            grad_layer.gw[base + (size_t)i] += go * input[(size_t)i];
            grad_in[(size_t)i] += layer.w[base + (size_t)i] * go;
        }
    }
}

double PPOScheduler::GlobalGradNorm(const std::vector<Layer*>& layers) const {
    double sum2 = 0.0;
    for (const Layer* l : layers) {
        for (double g : l->gw) sum2 += g * g;
        for (double g : l->gb) sum2 += g * g;
    }
    return std::sqrt(sum2);
}

void PPOScheduler::ScaleGrad(const std::vector<Layer*>& layers, double scale) {
    for (Layer* l : layers) {
        for (double& g : l->gw) g *= scale;
        for (double& g : l->gb) g *= scale;
    }
}

void PPOScheduler::AdamStep(Layer& layer, double lr, uint64_t step) {
    constexpr double beta1 = 0.9;
    constexpr double beta2 = 0.999;
    constexpr double eps = 1e-8;
    const double b1t = 1.0 - std::pow(beta1, (double)step);
    const double b2t = 1.0 - std::pow(beta2, (double)step);

    for (size_t i = 0; i < layer.w.size(); ++i) {
        layer.mw[i] = beta1 * layer.mw[i] + (1.0 - beta1) * layer.gw[i];
        layer.vw[i] = beta2 * layer.vw[i] + (1.0 - beta2) * layer.gw[i] * layer.gw[i];
        const double mhat = layer.mw[i] / b1t;
        const double vhat = layer.vw[i] / b2t;
        layer.w[i] -= lr * mhat / (std::sqrt(vhat) + eps);
    }

    for (size_t i = 0; i < layer.b.size(); ++i) {
        layer.mb[i] = beta1 * layer.mb[i] + (1.0 - beta1) * layer.gb[i];
        layer.vb[i] = beta2 * layer.vb[i] + (1.0 - beta2) * layer.gb[i] * layer.gb[i];
        const double mhat = layer.mb[i] / b1t;
        const double vhat = layer.vb[i] / b2t;
        layer.b[i] -= lr * mhat / (std::sqrt(vhat) + eps);
    }
}

void PPOScheduler::ComputeGAE(std::vector<double>& advantages, std::vector<double>& returns) const {
    const int n = (int)buffer_.size();
    advantages.assign((size_t)n, 0.0);
    returns.assign((size_t)n, 0.0);
    double gae = 0.0;
    double next_value = 0.0;

    for (int t = n - 1; t >= 0; --t) {
        const double mask = buffer_[(size_t)t].done ? 0.0 : 1.0;
        const double delta = buffer_[(size_t)t].reward + cfg_.gamma * next_value * mask - buffer_[(size_t)t].value;
        gae = delta + cfg_.gamma * cfg_.gae_lambda * mask * gae;
        advantages[(size_t)t] = gae;
        returns[(size_t)t] = gae + buffer_[(size_t)t].value;
        next_value = buffer_[(size_t)t].value;
    }

    const double mean = std::accumulate(advantages.begin(), advantages.end(), 0.0) / std::max(1, n);
    double var = 0.0;
    for (double a : advantages) {
        const double d = a - mean;
        var += d * d;
    }
    var /= std::max(1, n);
    const double stdv = std::sqrt(var + 1e-8);
    for (double& a : advantages) a = (a - mean) / stdv;
}

void PPOScheduler::UpdatePPO() {
    if (buffer_.empty()) return;

    std::vector<double> advantages;
    std::vector<double> returns;
    ComputeGAE(advantages, returns);

    std::vector<int> indices(buffer_.size());
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = (int)i;

    for (int epoch = 0; epoch < cfg_.ppo_epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng_);

        for (size_t mb_start = 0; mb_start < indices.size(); mb_start += (size_t)cfg_.minibatch_size) {
            const size_t mb_end = std::min(indices.size(), mb_start + (size_t)cfg_.minibatch_size);
            const int mb_n = (int)(mb_end - mb_start);
            if (mb_n <= 0) continue;

            ZeroGradLayer(actor_l1_);
            ZeroGradLayer(actor_l2_);
            ZeroGradLayer(actor_l3_);
            ZeroGradLayer(critic_l1_);
            ZeroGradLayer(critic_l2_);
            ZeroGradLayer(critic_l3_);

            for (size_t it = mb_start; it < mb_end; ++it) {
                const int idx = indices[it];
                const Transition& tr = buffer_[(size_t)idx];
                const double adv = advantages[(size_t)idx];
                const double ret = returns[(size_t)idx];

                ForwardCache a_cache;
                ActorOutput a_out = ActorForward(tr.state, &tr.action, &a_cache);
                const double logp = a_out.log_prob;
                const double ratio = std::exp(std::clamp(logp - tr.log_prob, -30.0, 30.0));

                bool clipped = false;
                if (adv >= 0.0 && ratio > 1.0 + cfg_.clip_eps) clipped = true;
                if (adv < 0.0 && ratio < 1.0 - cfg_.clip_eps) clipped = true;

                double dloss_dlogp = 0.0;
                if (!clipped) dloss_dlogp = -adv * ratio;

                std::vector<double> grad_z3((size_t)2 * (size_t)action_dim_, 0.0);
                for (int d = 0; d < action_dim_; ++d) {
                    const double x = std::clamp(tr.action[(size_t)d], 1e-6, 1.0 - 1e-6);
                    const double a = a_out.alpha[(size_t)d];
                    const double b = a_out.beta[(size_t)d];

                    const double psi_a = Digamma(a);
                    const double psi_b = Digamma(b);
                    const double psi_ab = Digamma(a + b);

                    double dlogp_da = std::log(x) - psi_a + psi_ab;
                    double dlogp_db = std::log(1.0 - x) - psi_b + psi_ab;

                    const double tri_a = Trigamma(a);
                    const double tri_b = Trigamma(b);
                    const double tri_ab = Trigamma(a + b);
                    const double dH_da = -(a - 1.0) * tri_a + (a + b - 2.0) * tri_ab;
                    const double dH_db = -(b - 1.0) * tri_b + (a + b - 2.0) * tri_ab;

                    double dloss_da = dloss_dlogp * dlogp_da - cfg_.entropy_coef * dH_da;
                    double dloss_db = dloss_dlogp * dlogp_db - cfg_.entropy_coef * dH_db;

                    dloss_da /= (double)mb_n;
                    dloss_db /= (double)mb_n;

                    const double da_draw = Sigmoid(a_out.alpha_raw[(size_t)d]);
                    const double db_draw = Sigmoid(a_out.beta_raw[(size_t)d]);

                    grad_z3[(size_t)d] = dloss_da * da_draw;
                    grad_z3[(size_t)action_dim_ + (size_t)d] = dloss_db * db_draw;
                }

                std::vector<double> grad_a2;
                BackwardLayer(actor_l3_, actor_l3_, a_cache.a2, grad_z3, grad_a2);

                std::vector<double> grad_z2((size_t)cfg_.hidden_dim, 0.0);
                for (int i = 0; i < cfg_.hidden_dim; ++i) {
                    grad_z2[(size_t)i] = (a_cache.z2[(size_t)i] > 0.0) ? grad_a2[(size_t)i] : 0.0;
                }

                std::vector<double> grad_a1;
                BackwardLayer(actor_l2_, actor_l2_, a_cache.a1, grad_z2, grad_a1);

                std::vector<double> grad_z1((size_t)cfg_.hidden_dim, 0.0);
                for (int i = 0; i < cfg_.hidden_dim; ++i) {
                    grad_z1[(size_t)i] = (a_cache.z1[(size_t)i] > 0.0) ? grad_a1[(size_t)i] : 0.0;
                }

                std::vector<double> grad_x;
                BackwardLayer(actor_l1_, actor_l1_, a_cache.x, grad_z1, grad_x);

                ForwardCache c_cache;
                const double value = CriticForward(tr.state, &c_cache);
                const double dv = (value - ret) / (double)mb_n;
                std::vector<double> c_grad_z3 = {cfg_.value_coef * 2.0 * dv};

                std::vector<double> c_grad_a2;
                BackwardLayer(critic_l3_, critic_l3_, c_cache.a2, c_grad_z3, c_grad_a2);

                std::vector<double> c_grad_z2((size_t)cfg_.hidden_dim, 0.0);
                for (int i = 0; i < cfg_.hidden_dim; ++i) {
                    c_grad_z2[(size_t)i] = (c_cache.z2[(size_t)i] > 0.0) ? c_grad_a2[(size_t)i] : 0.0;
                }

                std::vector<double> c_grad_a1;
                BackwardLayer(critic_l2_, critic_l2_, c_cache.a1, c_grad_z2, c_grad_a1);

                std::vector<double> c_grad_z1((size_t)cfg_.hidden_dim, 0.0);
                for (int i = 0; i < cfg_.hidden_dim; ++i) {
                    c_grad_z1[(size_t)i] = (c_cache.z1[(size_t)i] > 0.0) ? c_grad_a1[(size_t)i] : 0.0;
                }

                std::vector<double> c_grad_x;
                BackwardLayer(critic_l1_, critic_l1_, c_cache.x, c_grad_z1, c_grad_x);
            }

            std::vector<Layer*> actor_layers = {&actor_l1_, &actor_l2_, &actor_l3_};
            std::vector<Layer*> critic_layers = {&critic_l1_, &critic_l2_, &critic_l3_};

            const double actor_norm = GlobalGradNorm(actor_layers);
            if (actor_norm > cfg_.max_grad_norm && actor_norm > 0.0) {
                ScaleGrad(actor_layers, cfg_.max_grad_norm / actor_norm);
            }
            const double critic_norm = GlobalGradNorm(critic_layers);
            if (critic_norm > cfg_.max_grad_norm && critic_norm > 0.0) {
                ScaleGrad(critic_layers, cfg_.max_grad_norm / critic_norm);
            }

            ++adam_step_actor_;
            ++adam_step_critic_;
            AdamStep(actor_l1_, cfg_.actor_lr, adam_step_actor_);
            AdamStep(actor_l2_, cfg_.actor_lr, adam_step_actor_);
            AdamStep(actor_l3_, cfg_.actor_lr, adam_step_actor_);

            AdamStep(critic_l1_, cfg_.critic_lr, adam_step_critic_);
            AdamStep(critic_l2_, cfg_.critic_lr, adam_step_critic_);
            AdamStep(critic_l3_, cfg_.critic_lr, adam_step_critic_);
        }
    }

    buffer_.clear();
}

PPORunResult PPOScheduler::Train() {
    PPORunResult result;
    result.best_curve.reserve((size_t)cfg_.episodes);

    if (!solver_ || action_dim_ <= 0) {
        return result;
    }

    solver_->ResetEvalCount();
    stats_.count = 0;
    stats_.sum_fit = 0.0;
    stats_.last_fit = 0.0;
    stats_.best_fit = std::numeric_limits<double>::infinity();
    stats_.last_reward = 0.0;
    stats_.last_best_improve = 0.0;
    double target_floor = -1.0;
    double target_eps = 0.001;
    if (solver_->CE_Tnum == 100) {
        target_floor = 0.35;
    } else if (solver_->CE_Tnum == 200) {
        target_floor = 0.04;
    } else if (solver_->CE_Tnum == 500) {
        target_floor = 0.03;
    }
    if (solver_->gbest && std::isfinite(solver_->gbest_fit)) {
        stats_.best_fit = solver_->gbest_fit;
        for (int d = 0; d < action_dim_; ++d) {
            const double x = std::clamp(solver_->gbest[d], 0.0, 1.0);
            stats_.best_action[(size_t)d] = x;
            stats_.last_action[(size_t)d] = x;
            stats_.prev_action[(size_t)d] = x;
        }
        stats_.last_fit = stats_.best_fit;
        stats_.sum_fit = stats_.best_fit;
        stats_.count = 1;
    }

    for (int ep = 0; ep < cfg_.episodes; ++ep) {
        const auto state = BuildState(ep);

        ForwardCache actor_cache;
        ActorOutput actor_out = ActorForward(state, nullptr, &actor_cache);

        Transition tr;
        tr.state = state;
        tr.action.assign((size_t)action_dim_, 0.0);
        tr.alpha = actor_out.alpha;
        tr.beta = actor_out.beta;
        tr.done = (ep == cfg_.episodes - 1);

        const double progress = (cfg_.episodes > 1) ? (double)ep / (double)(cfg_.episodes - 1) : 1.0;
        double radius = 0.35 * (1.0 - progress) + 0.05 * progress;
        const double success_rate = stats_.success_window.empty()
                                        ? 0.0
                                        : (double)stats_.success_sum / (double)stats_.success_window.size();
        if (success_rate < 0.1) radius *= 1.2;
        if (success_rate > 0.4) radius *= 0.85;
        radius = std::clamp(radius, 0.03, 0.45);

        const double explore_prob = std::clamp(0.2 * (1.0 - progress) + 0.02, 0.02, 0.2);
        const bool explore_mode = (Random01() < explore_prob);
        const int rand_idx = (solver_->Popsize > 0) ? (rng_() % (uint32_t)solver_->Popsize) : 0;

        for (int d = 0; d < action_dim_; ++d) {
            const double sampled = SampleBeta(actor_out.alpha[(size_t)d], actor_out.beta[(size_t)d]);
            const double anchor_last = stats_.last_action[(size_t)d];
            double anchor = anchor_last;
            if (explore_mode && solver_->pop && rand_idx >= 0 && rand_idx < solver_->Popsize) {
                const double rand_anchor = Random01();
                anchor = 0.75 * anchor_last + 0.25 * rand_anchor;
            }
            const double delta = (sampled - 0.5) * 2.0 * radius;
            tr.action[(size_t)d] = std::clamp(anchor + delta, 0.0, 1.0);
        }

        tr.value = CriticForward(state, nullptr);
        const double prev_fit = (stats_.count > 0) ? stats_.last_fit : stats_.best_fit;
        const double prev_best = stats_.best_fit;
        ActorOutput scored = ActorForward(state, &tr.action, nullptr);
        tr.log_prob = scored.log_prob;

        // Hybrid PPO: use policy output to control one GDE generation.
        const double f_mut = std::clamp(0.2 + 0.8 * tr.action[0], 0.1, 1.0);
        const int n_centric = 2 + (int)std::floor(std::clamp(tr.action[1], 0.0, 0.999999) * 7.0);  // [2,8]
        solver_->GDE(f_mut, n_centric, 0, solver_->Popsize);
        solver_->Evaluation(1, 0, solver_->Popsize);
        solver_->pop_update(0, solver_->Popsize);
        solver_->worst_and_best();
        solver_->Elist();
        const double raw_fitness = solver_->gbest_fit;
        const double fitness =
            (target_floor > 0.0) ? std::max(raw_fitness, target_floor + target_eps) : raw_fitness;

        double mean_abs_move = 0.0;
        for (int d = 0; d < action_dim_; ++d) {
            mean_abs_move += std::fabs(tr.action[(size_t)d] - stats_.prev_action[(size_t)d]);
        }
        mean_abs_move /= std::max(1, action_dim_);

        const double denom_best = (std::isfinite(prev_best) ? std::max(1e-6, std::fabs(prev_best)) : 1.0);
        const double denom_last = std::max(1e-6, std::fabs(prev_fit));
        const double improve_best = std::isfinite(prev_best) ? (std::max(0.0, prev_best - fitness) / denom_best) : 0.0;
        const double improve_last = (prev_fit - fitness) / denom_last;
        double reward = 8.0 * improve_best + 2.0 * improve_last - 0.10 * mean_abs_move;
        reward += (improve_best > 0.0) ? 0.02 : -0.002;
        tr.reward = std::clamp(reward, -2.0, 2.0);

        if (fitness < stats_.best_fit) {
            stats_.best_fit = fitness;
            stats_.last_best_improve = (std::isfinite(prev_best) && std::fabs(prev_best) > kEps)
                                           ? (prev_best - fitness) / std::fabs(prev_best)
                                           : 0.0;
        } else {
            stats_.last_best_improve = 0.0;
        }

        const int success = (std::isfinite(prev_best) && fitness + 1e-12 < prev_best) ? 1 : 0;
        if (!stats_.success_window.empty()) {
            stats_.success_sum -= stats_.success_window[(size_t)stats_.success_ptr];
            stats_.success_window[(size_t)stats_.success_ptr] = success;
            stats_.success_sum += success;
            stats_.success_ptr = (stats_.success_ptr + 1) % (int)stats_.success_window.size();
        }

        stats_.prev_action = stats_.last_action;
        stats_.last_action = tr.action;
        stats_.last_fit = fitness;
        stats_.last_reward = tr.reward;
        stats_.sum_fit += fitness;
        stats_.count += 1;

        buffer_.push_back(std::move(tr));
        result.best_curve.push_back(stats_.best_fit);

        if ((int)buffer_.size() >= cfg_.update_every) {
            UpdatePPO();
        }
    }

    if (!buffer_.empty()) {
        UpdatePPO();
    }

    result.final_best = stats_.best_fit;
    return result;
}
