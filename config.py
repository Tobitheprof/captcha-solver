import argparse
import sys

import dnnlib
from dnnlib.submission.submit import submit_run
from dnnlib.EasyDict import EasyDict

# Submit config
submit_config = dnnlib.SubmitConfig()
submit_config.run_dir_root = 'results'
submit_config.run_dir_ignore += ['datasets', 'results']

desc = "autoencoder"

# Network config
net_config = EasyDict(func_name="network.autoencoder")

# Optimizer config
optimizer_config = EasyDict(beta1=0.9, beta2=0.99, epsilon=1e-8)

# Noise augmentation config
gaussian_noise_config = {
    'func_name': 'train.AugmentGaussian',
    'train_stddev_rng_range': (0.0, 50.0),
    'validation_stddev': 25.0
}
poisson_noise_config = {
    'func_name': 'train.AugmentPoisson',
    'lam_max': 50.0
}

# Train config
train_config = EasyDict(
    iteration_count=300000,
    eval_interval=1000,
    minibatch_size=4,
    run_func_name="train.train",
    learning_rate=0.0003,
    ramp_down_perc=0.3,
    noise=gaussian_noise_config,
    noise2noise=True,
    train_tfrecords='datasets/imagenet_val_raw.tfrecords'
)

# Validation run config
validate_config = EasyDict(
    run_func_name="validation.validate",
    dataset=None,
    network_snapshot=None,
    noise=gaussian_noise_config
)

# jhellsten quota group

def train(args):
    if args.noise2noise is not None:
        train_config.noise2noise = args.noise2noise
    if args.long_train:
        train_config.iteration_count = 500000
        train_config.eval_interval = 5000
        train_config.ramp_down_perc = 0.5
    if args.noise is not None:
        noise_type = args.noise.lower()
        if noise_type == 'gaussian':
            train_config.noise = gaussian_noise_config
        elif noise_type == 'poisson':
            train_config.noise = poisson_noise_config
        else:
            print('Unknown noise type', args.noise)
            sys.exit(1)

    submit_config.run_desc += "-n2n" if train_config.noise2noise
