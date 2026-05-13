# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np

from agmm.iv_dgp_generate_data import generate_data
from agmm.agmm_trainer import (
    train_agmm,
    train_kernellayergmm,
    train_centroidmmdgmm,
    train_kernellossagmm
)
from agmm.agmm_utilities import eval_performance
from agmm.rbflayer import gaussian, inverse_multiquadric


def dgp_to_bools(dgp_str):
    x = False
    z = False

    if dgp_str == "x_image":
        x = True
    elif dgp_str == "z_image":
        z = True
    elif dgp_str == "xz_image":
        x = True
        z = True

    return x, z


def arch_hyperparam_select(est, dgp):
    dropout_p = 0.1
    n_hidden = 200
    return dropout_p, n_hidden


def kernel_hyperparam_select(est, dgp, g_features_override=None, n_centers_override=None):
    g_features = 10
    n_centers = 25
    kernel_fn = gaussian

    # --------------------------------------------------
    # Estimator-specific defaults
    # --------------------------------------------------
    if est == "KernelLayerMMDGMM":
        if dgp == "x_image":
            g_features = 70
            n_centers = 25
        elif dgp == "xz_image":
            g_features = 100
            n_centers = 100

    elif est == "CentroidMMDGMM":
        # Default setting for thesis unless overridden
        g_features = 10
        n_centers = 25

    elif est == "KernelLossAGMM":
        pass

    # --------------------------------------------------
    # Apply overrides after estimator-specific defaults
    # --------------------------------------------------
    if g_features_override is not None:
        g_features = g_features_override

    if n_centers_override is not None:
        n_centers = n_centers_override

    # --------------------------------------------------
    # Kernel parameters
    # --------------------------------------------------
    centers = np.random.uniform(-4, 4, size=(n_centers, g_features))
    sigma = 2.0 / g_features
    sigmas = np.ones((n_centers,)) * sigma

    return g_features, n_centers, kernel_fn, centers, sigmas, sigma

def train_hyperparam_select(est, dgp):
    learner_lr = 1e-4
    adversary_lr = 5e-5
    learner_l2 = 1e-4
    adversary_l2 = 1e-4
    adversary_norm_reg = 1e-4
    n_epochs = 200
    bs = 100
    train_learner_every = 1
    train_adversary_every = 2

    if est == "AGMM":
        if dgp == "z_image":
            learner_lr = 2e-4
        elif dgp == "x_image":
            learner_l2 = 1e-8
            adversary_l2 = 1e-8

    elif est == "KernelLayerMMDGMM":
        if dgp == "z_image":
            learner_l2 = 1e-10
            adversary_l2 = 1e-10
        elif dgp == "x_image":
            learner_l2 = 1e-10
            adversary_l2 = 1e-10
        elif dgp == "xz_image":
            learner_lr = 1e-5

    elif est == "CentroidMMDGMM":
        learner_lr = 5e-5
        adversary_lr = 2e-5
        learner_l2 = 1e-4
        adversary_l2 = 1e-4
        adversary_norm_reg = 1e-3
        train_adversary_every = 3

    elif est == "KernelLossAGMM":
        pass

    return (
        learner_lr,
        adversary_lr,
        learner_l2,
        adversary_l2,
        adversary_norm_reg,
        n_epochs,
        bs,
        train_learner_every,
        train_adversary_every
    )


def experiment(
    dgp,
    iv_strength,
    tau_fn,
    num_data,
    est,
    device="cpu",
    DEBUG=False,
    g_features=None,
    return_history=False,
    n_centers=None
):
    # --------------------------------------------------
    # Kernel-related hyperparameters
    # --------------------------------------------------
    g_features, n_centers, kernel_fn, centers, sigmas, sigma = kernel_hyperparam_select(
        est,
        dgp,
        g_features_override=g_features,
        n_centers_override=n_centers
   )

    # --------------------------------------------------
    # Architecture hyperparameters
    # --------------------------------------------------
    n_t = 1
    dropout_p, n_hidden = arch_hyperparam_select(est, dgp)

    # --------------------------------------------------
    # Training hyperparameters
    # --------------------------------------------------
    (
        learner_lr,
        adversary_lr,
        learner_l2,
        adversary_l2,
        adversary_norm_reg,
        n_epochs,
        bs,
        train_learner_every,
        train_adversary_every
    ) = train_hyperparam_select(est, dgp)

    # --------------------------------------------------
    # Generate data
    # --------------------------------------------------
    X_IMAGE, Z_IMAGE = dgp_to_bools(dgp)
    n_instruments = 1

    data = generate_data(
        X_IMAGE=X_IMAGE,
        Z_IMAGE=Z_IMAGE,
        tau_fn=tau_fn,
        n_samples=num_data,
        n_dev_samples=num_data // 2,
        n_instruments=n_instruments,
        iv_strength=iv_strength,
        device=device
    )

    Z_train, T_train, Y_train, G_train = data[0]
    Z_val, T_val, Y_val, G_val = data[1]
    Z_test, T_test, Y_test, G_test = data[2]
    Z_dev, T_dev, Y_dev, G_dev = data[3]

    # --------------------------------------------------
    # Train estimator
    # --------------------------------------------------
    if est == "AGMM":
        estimator = train_agmm(
            Z_train, T_train, Y_train, G_train,
            Z_dev, T_dev, Y_dev, G_dev,
            Z_val, T_val, Y_val, G_val,
            T_test, G_test,
            X_IMAGE=X_IMAGE,
            Z_IMAGE=Z_IMAGE,
            n_t=n_t,
            n_instruments=n_instruments,
            n_hidden=n_hidden,
            dropout_p=dropout_p,
            learner_lr=learner_lr,
            adversary_lr=adversary_lr,
            learner_l2=learner_l2,
            adversary_l2=adversary_l2,
            adversary_norm_reg=adversary_norm_reg,
            n_epochs=n_epochs,
            batch_size=bs,
            train_learner_every=train_learner_every,
            train_adversary_every=train_adversary_every,
            device=device,
            DEBUG=DEBUG
        )

    elif est == "KernelLayerMMDGMM":
        estimator = train_kernellayergmm(
            Z_train, T_train, Y_train, G_train,
            Z_dev, T_dev, Y_dev, G_dev,
            Z_val, T_val, Y_val, G_val,
            T_test, G_test,
            g_features=g_features,
            n_centers=n_centers,
            kernel_fn=kernel_fn,
            centers=centers,
            sigmas=sigmas,
            X_IMAGE=X_IMAGE,
            Z_IMAGE=Z_IMAGE,
            n_t=n_t,
            n_instruments=n_instruments,
            n_hidden=n_hidden,
            dropout_p=dropout_p,
            learner_lr=learner_lr,
            adversary_lr=adversary_lr,
            learner_l2=learner_l2,
            adversary_l2=adversary_l2,
            adversary_norm_reg=adversary_norm_reg,
            n_epochs=n_epochs,
            batch_size=bs,
            train_learner_every=train_learner_every,
            train_adversary_every=train_adversary_every,
            device=device,
            DEBUG=DEBUG
        )

    elif est == "CentroidMMDGMM":
        estimator = train_centroidmmdgmm(
            Z_train, T_train, Y_train, G_train,
            Z_dev, T_dev, Y_dev, G_dev,
            Z_val, T_val, Y_val, G_val,
            T_test, G_test,
            n_centers=n_centers,
            g_features=g_features,
            kernel_fn=kernel_fn,
            sigma=sigma,
            X_IMAGE=X_IMAGE,
            Z_IMAGE=Z_IMAGE,
            n_t=n_t,
            n_instruments=n_instruments,
            n_hidden=n_hidden,
            dropout_p=dropout_p,
            learner_lr=learner_lr,
            adversary_lr=adversary_lr,
            learner_l2=learner_l2,
            adversary_l2=adversary_l2,
            adversary_norm_reg=adversary_norm_reg,
            n_epochs=n_epochs,
            batch_size=bs,
            train_learner_every=train_learner_every,
            train_adversary_every=train_adversary_every,
            device=device,
            DEBUG=DEBUG
        )

    elif est == "KernelLossAGMM":
        estimator = train_kernellossagmm(
            Z_train, T_train, Y_train, G_train,
            Z_dev, T_dev, Y_dev, G_dev,
            Z_val, T_val, Y_val, G_val,
            T_test, G_test,
            g_features=g_features,
            kernel_fn=kernel_fn,
            sigma=sigma,
            X_IMAGE=X_IMAGE,
            Z_IMAGE=Z_IMAGE,
            n_t=n_t,
            n_instruments=n_instruments,
            n_hidden=n_hidden,
            dropout_p=dropout_p,
            learner_lr=learner_lr,
            adversary_lr=adversary_lr,
            learner_l2=learner_l2,
            adversary_l2=adversary_l2,
            adversary_norm_reg=adversary_norm_reg,
            n_epochs=n_epochs,
            batch_size=bs,
            train_learner_every=train_learner_every,
            train_adversary_every=train_adversary_every,
            device=device,
            DEBUG=DEBUG
        )

    else:
        raise ValueError(f"Unknown estimator name: {est}")

    # --------------------------------------------------
    # Evaluate performance
    # --------------------------------------------------
    results = eval_performance(
        estimator,
        T_test,
        true_of_T_test=G_test
    )

    # --------------------------------------------------
    # Optional: return convergence history if available
    # --------------------------------------------------
    if return_history:
        history = None

        if hasattr(estimator, "history"):
            history = estimator.history
        elif hasattr(estimator, "loss_history"):
            history = estimator.loss_history
        elif hasattr(estimator, "training_history"):
            history = estimator.training_history

        return results, history

    return results