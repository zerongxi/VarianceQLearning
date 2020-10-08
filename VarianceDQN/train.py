import sys
from trainer import Trainer

BASE = dict(
    duel=True,
    double=True,
    prioritized_replay=True,
    noisy=False,
    #method="td",
    #uncertainty=True,
    history_len=4,
    hidden_dim=512,
    channels=(32, 64, 64),
    kernels=(8, 4, 3),
    strides=(4, 2, 1),
    optim="adam",
    optim_kwargs=dict(lr=6.25e-5, eps=1.5e-4),
    gamma=.99,
    eps_init=1.,
    eps_final=.01,
    eps_steps=10 ** 6,
    eps_eval=.01,
    mem_capacity=10 ** 6,
    mem_init_sz=50000,
    mem_alpha=.6,
    mem_beta=.4,
    mem_priority_eps=1e-6,
    mem_priority_upper=1.,
    batch_sz=32,
    optimize_freq=1,
    train_steps=50 * 10 ** 6,
    eval_freq=250 * 10 ** 3,
    log_freq=250 * 10 ** 3,
    print_freq=1000 * 10 ** 3,
    eval_method="frames",
    #eval_episodes=20,
    eval_frames=500 * 10 ** 3,
    eval_max_freq=30 * 60 * 60 // 4,
    update_target_freq=30000,
    c=.1,
    adaptive_eps=None,
    #adaptive_freq=2500,

    label="DDQN",
    device_id="cuda:0",
    env_id="Breakout",
)


assert len(sys.argv) > 2, "Two arguments (game and method) are required!"
assert sys.argv[2] in ["DDQN", "VDQN", "TDDQN"], "The second argument must be DDQN/VDQN/TDDQN"
game = sys.argv[1]
method = sys.argv[2]

params = BASE.copy()
params.update(dict(
    uncertainty=method in ["VDQN", "TDDQN"],
    method=method,
    label=method,
    env_id=game,
    eps_init=1. if method == "DDQN" else 0.,
    eps_final=.01 if method == "DDQN" else 0.,
))
Trainer(params).train()
