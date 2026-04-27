import numpy as np
import torch

from agmm.iv_dgp_generate_data import generate_data

# --------------------------------------------------
# Keras imports
# --------------------------------------------------
try:
    from tensorflow.keras.layers import Input, Dense, Concatenate
except ImportError:
    from keras.layers import Input, Dense, Concatenate

# --------------------------------------------------
# DeepIV local imports
# These files are inside: experiment/agmm/deepiv/
# --------------------------------------------------
from deepiv.models import Treatment, Response
from deepiv import architectures
from deepiv import densities


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


def _to_numpy(a):
    """
    Convert torch tensors or array-like objects to numpy arrays.
    """
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def _make_2d(a):
    """
    Ensure arrays have shape (n, d).
    """
    a = _to_numpy(a)

    if a.ndim == 1:
        a = a.reshape(-1, 1)

    return a.astype(np.float32)


def _flatten_if_image(a):
    """
    Flatten image data such as (n, 1, 28, 28) or (n, 28, 28)
    into (n, d). If already 2D, keep unchanged.
    """
    a = _to_numpy(a)

    if a.ndim > 2:
        a = a.reshape(a.shape[0], -1)

    return a.astype(np.float32)


def compute_mse_r2(y_true, y_pred):
    """
    Compute structural recovery MSE and R2.
    """
    y_true = _make_2d(y_true)
    y_pred = _make_2d(y_pred)

    mse = float(np.mean((y_true - y_pred) ** 2))
    var_y = float(np.var(y_true))

    if var_y <= 1e-12:
        r2 = np.nan
    else:
        r2 = float(1.0 - mse / var_y)

    return mse, r2


def deep_iv_fit(
    z,
    t,
    y,
    x=None,
    epochs=200,
    batch_size=100,
    hidden=(128, 64, 32),
    verbose=1,
):
    """
    Fit DeepIV.
    """

    z = _flatten_if_image(z)
    t = _make_2d(t)
    y = _make_2d(y)

    if x is not None:
        x = _flatten_if_image(x)

    n = z.shape[0]
    dropout_rate = min(1000.0 / (1000.0 + n), 0.5)

    n_components = 10
    act = "relu"

    # --------------------------------------------------
    # Stage 1: treatment model p(T | Z)
    # --------------------------------------------------
    instruments = Input(shape=(z.shape[1],), name="instruments")
    treatment = Input(shape=(t.shape[1],), name="treatment")

    if x is None:
        treatment_input = instruments

        est_treat = architectures.feed_forward_net(
            treatment_input,
            lambda h: densities.mixture_of_gaussian_output(h, n_components),
            hidden_layers=list(hidden),
            dropout_rate=dropout_rate,
            l2=0.0001,
            activations=act,
        )

        treatment_model = Treatment(inputs=[instruments], outputs=est_treat)

        treatment_model.compile(
            "adam",
            loss="mixture_of_gaussians",
            n_components=n_components,
        )

        treatment_model.fit(
            [z],
            t,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )

    else:
        features = Input(shape=(x.shape[1],), name="features")
        treatment_input = Concatenate(axis=1)([instruments, features])

        est_treat = architectures.feed_forward_net(
            treatment_input,
            lambda h: densities.mixture_of_gaussian_output(h, n_components),
            hidden_layers=list(hidden),
            dropout_rate=dropout_rate,
            l2=0.0001,
            activations=act,
        )

        treatment_model = Treatment(inputs=[instruments, features], outputs=est_treat)

        treatment_model.compile(
            "adam",
            loss="mixture_of_gaussians",
            n_components=n_components,
        )

        treatment_model.fit(
            [z, x],
            t,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )

    # --------------------------------------------------
    # Stage 2: response model h(T)
    # --------------------------------------------------
    if x is None:
        response_input = treatment

        est_response = architectures.feed_forward_net(
            response_input,
            Dense(1),
            activations=act,
            hidden_layers=list(hidden),
            l2=0.001,
            dropout_rate=dropout_rate,
        )

        response_model = Response(
            treatment=treatment_model,
            inputs=[treatment],
            outputs=est_response,
        )

        response_model.compile("adam", loss="mse")

        response_model.fit(
            [z],
            y,
            epochs=epochs,
            verbose=verbose,
            batch_size=batch_size,
            samples_per_batch=2,
        )

    else:
        features = Input(shape=(x.shape[1],), name="features")
        response_input = Concatenate(axis=1)([features, treatment])

        est_response = architectures.feed_forward_net(
            response_input,
            Dense(1),
            activations=act,
            hidden_layers=list(hidden),
            l2=0.001,
            dropout_rate=dropout_rate,
        )

        response_model = Response(
            treatment=treatment_model,
            inputs=[features, treatment],
            outputs=est_response,
        )

        response_model.compile("adam", loss="mse")

        response_model.fit(
            [z, x],
            y,
            epochs=epochs,
            verbose=verbose,
            batch_size=batch_size,
            samples_per_batch=2,
        )

    return response_model


def experiment(
    dgp,
    iv_strength,
    tau_fn,
    num_data,
    est="DeepIV",
    device="cpu",
    DEBUG=False,
    epochs=200,
    batch_size=100,
    estimator=None,
):
    """
    DeepIV experiment aligned with the existing AGMM experiment interface.
    """

    if estimator is not None:
        est = estimator

    if est != "DeepIV":
        raise ValueError(f"This file only supports est='DeepIV'. Got est={est}")

    # --------------------------------------------------
    # This DeepIV implementation is for z_image only:
    # high-dimensional Z, scalar treatment T.
    # --------------------------------------------------
    if dgp in ["x_image", "xz_image"]:
        raise ValueError(
            "This DeepIV implementation expects scalar treatment. "
            "Use dgp='z_image'."
        )

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
        device=device,
    )

    Z_train, T_train, Y_train, G_train = data[0]
    Z_val, T_val, Y_val, G_val = data[1]
    Z_test, T_test, Y_test, G_test = data[2]
    Z_dev, T_dev, Y_dev, G_dev = data[3]

    # --------------------------------------------------
    # Combine train and dev samples for DeepIV training.
    # Test set remains untouched.
    # --------------------------------------------------
    Z_train_np = _flatten_if_image(Z_train)
    T_train_np = _make_2d(T_train)
    Y_train_np = _make_2d(Y_train)

    Z_dev_np = _flatten_if_image(Z_dev)
    T_dev_np = _make_2d(T_dev)
    Y_dev_np = _make_2d(Y_dev)

    Z_fit = np.vstack([Z_train_np, Z_dev_np])
    T_fit = np.vstack([T_train_np, T_dev_np])
    Y_fit = np.vstack([Y_train_np, Y_dev_np])

    T_test_np = _make_2d(T_test)
    G_test_np = _make_2d(G_test)

    if T_fit.shape[1] != 1:
        raise ValueError(
            f"DeepIV expected scalar treatment with shape (n, 1), "
            f"but got treatment dimension {T_fit.shape[1]}."
        )

    verbose = 1 if DEBUG else 0

    response_model = deep_iv_fit(
        z=Z_fit,
        t=T_fit,
        y=Y_fit,
        x=None,
        epochs=epochs,
        batch_size=batch_size,
        hidden=(128, 64, 32),
        verbose=verbose,
    )

    pred_test = response_model.predict([T_test_np])
    pred_test = _make_2d(pred_test)

    mse, r2 = compute_mse_r2(G_test_np, pred_test)

    results = {
        "MSEearlystop": mse,
        "R2fin": r2,
        "MSEfin": mse,
        "MSEavg": mse,
        "R2avg": r2,
        "R2earlystop": r2,
    }

    return results