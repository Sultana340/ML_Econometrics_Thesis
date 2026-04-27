"""
Model wrappers for the DeepIV experiment.

This file defines two classes:

1. Treatment:
   First-stage model for estimating the treatment distribution p(T | Z).

2. Response:
   Second-stage model for estimating the structural response h(T).

The implementation is designed to work with the DeepIV experiment wrapper used
in the thesis comparison between AGMM and DeepIV.
"""

import numpy as np

# --------------------------------------------------
# TensorFlow / Keras imports
# --------------------------------------------------
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
except ImportError:
    import tensorflow as tf
    from keras.models import Model


# --------------------------------------------------
# Local density utilities
# --------------------------------------------------
try:
    from agmm.deepiv import densities
except ImportError:
    from . import densities


def _as_list(x):
    """
    Ensure input is a list.
    """
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _to_numpy(x):
    """
    Convert TensorFlow tensors or array-like objects to numpy arrays.
    """
    if isinstance(x, tf.Tensor):
        return x.numpy()
    return np.asarray(x)


def _make_2d(x):
    """
    Ensure array has shape (n, d).
    """
    x = _to_numpy(x)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    return x.astype(np.float32)


def _softmax_np(logits):
    """
    Numerically stable softmax using numpy.
    """
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


class Treatment(Model):
    """
    First-stage DeepIV treatment model.

    This model estimates the conditional treatment distribution p(T | Z)
    using a mixture-of-Gaussians output.

    Example
    -------
    treatment_model = Treatment(inputs=[instruments], outputs=est_treat)

    treatment_model.compile(
        "adam",
        loss="mixture_of_gaussians",
        n_components=10,
    )

    treatment_model.fit([z], t)
    """

    def __init__(self, inputs, outputs, **kwargs):
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)
        self.n_components = None

    def compile(self, optimizer="adam", loss=None, n_components=10, **kwargs):
        """
        Compile the treatment model.

        Parameters
        ----------
        optimizer : str or keras optimizer
            Optimizer used for training.

        loss : str or callable
            If loss is "mixture_of_gaussians", this is converted into
            the negative log-likelihood loss for a Gaussian mixture.

        n_components : int
            Number of Gaussian mixture components.
        """

        self.n_components = n_components

        if loss in ["mixture_of_gaussians", "mixture_of_gaussian"]:
            loss_fn = densities.mixture_of_gaussians_loss(n_components)
        else:
            loss_fn = loss

        super().compile(
            optimizer=optimizer,
            loss=loss_fn,
            **kwargs,
        )

    def _predict_mixture_params(self, inputs):
        """
        Predict mixture logits, means, and standard deviations.
        """
        if self.n_components is None:
            raise ValueError(
                "Treatment model has no n_components value. "
                "Compile with n_components before calling sample()."
            )

        inputs = _as_list(inputs)
        y_pred = self.predict(inputs, verbose=0)
        y_pred = np.asarray(y_pred, dtype=np.float32)

        k = self.n_components

        logits = y_pred[:, :k]
        means = y_pred[:, k : 2 * k]
        stds = y_pred[:, 2 * k : 3 * k]

        stds = np.maximum(stds, 1e-6)

        return logits, means, stds

    def sample(self, inputs, n_samples=1):
        """
        Draw samples from the estimated treatment distribution p(T | Z).

        Parameters
        ----------
        inputs : array or list of arrays
            Inputs to the treatment model, usually [Z] or [Z, X_controls].

        n_samples : int
            Number of treatment samples per observation.

        Returns
        -------
        samples : ndarray
            Sampled treatments with shape (n * n_samples, 1).
        """

        logits, means, stds = self._predict_mixture_params(inputs)
        probs = _softmax_np(logits)

        n = probs.shape[0]
        samples = []

        for i in range(n):
            for _ in range(n_samples):
                component = np.random.choice(self.n_components, p=probs[i])
                draw = np.random.normal(
                    loc=means[i, component],
                    scale=stds[i, component],
                )
                samples.append(draw)

        samples = np.asarray(samples, dtype=np.float32).reshape(-1, 1)

        return samples


class Response:
    """
    Second-stage DeepIV response model.

    This model estimates the structural response function h(T).

    In training, it uses the first-stage Treatment model to sample treatments
    from p(T | Z), then trains the response network on those sampled treatments.

    Example without controls
    ------------------------
    response_model = Response(
        treatment=treatment_model,
        inputs=[treatment],
        outputs=est_response,
    )

    response_model.compile("adam", loss="mse")
    response_model.fit([z], y, samples_per_batch=2)

    response_model.predict([t_test])
    """

    def __init__(self, treatment, inputs, outputs, **kwargs):
        self.treatment = treatment
        self.response_model = Model(inputs=inputs, outputs=outputs, **kwargs)

    def compile(self, optimizer="adam", loss="mse", **kwargs):
        """
        Compile the response model.
        """
        self.response_model.compile(
            optimizer=optimizer,
            loss=loss,
            **kwargs,
        )

    def fit(
        self,
        inputs,
        y,
        epochs=1,
        batch_size=100,
        verbose=0,
        samples_per_batch=2,
        **kwargs,
    ):
        """
        Fit the response model.

        Parameters
        ----------
        inputs : list
            For the no-control case, this is [Z].
            For the control case, this is [Z, X_controls].

        y : array
            Outcome variable.

        epochs : int
            Number of response-stage training epochs.

        batch_size : int
            Batch size.

        verbose : int
            Keras verbosity level.

        samples_per_batch : int
            Number of treatment samples drawn per observation.
        """

        inputs = _as_list(inputs)
        y = _make_2d(y)

        history = {"loss": []}

        n_response_inputs = len(self.response_model.inputs)

        for _ in range(epochs):
            # Sample treatments from the first-stage model p(T | Z)
            sampled_t = self.treatment.sample(
                inputs,
                n_samples=samples_per_batch,
            )

            # Repeat y to match the sampled treatments
            y_rep = np.repeat(y, samples_per_batch, axis=0)

            if n_response_inputs == 1:
                # Response model h(T)
                response_inputs = sampled_t

            elif n_response_inputs == 2:
                # Response model h(X_controls, T)
                if len(inputs) < 2:
                    raise ValueError(
                        "Response model expects controls, but fit() received only one input."
                    )

                controls = _make_2d(inputs[1])
                controls_rep = np.repeat(controls, samples_per_batch, axis=0)

                response_inputs = [controls_rep, sampled_t]

            else:
                raise ValueError(
                    f"Unsupported number of response inputs: {n_response_inputs}"
                )

            fit_history = self.response_model.fit(
                response_inputs,
                y_rep,
                epochs=1,
                batch_size=batch_size,
                verbose=verbose,
                **kwargs,
            )

            if "loss" in fit_history.history:
                history["loss"].append(fit_history.history["loss"][-1])

        class SimpleHistory:
            pass

        simple_history = SimpleHistory()
        simple_history.history = history

        return simple_history

    def predict(self, inputs, **kwargs):
        """
        Predict structural response h(T).

        For the no-control case, use:
            response_model.predict([T_test])

        For the control case, use:
            response_model.predict([X_controls, T_test])
        """

        inputs = _as_list(inputs)

        n_response_inputs = len(self.response_model.inputs)

        if n_response_inputs == 1:
            pred_inputs = _make_2d(inputs[0])

        elif n_response_inputs == 2:
            if len(inputs) != 2:
                raise ValueError(
                    "Response model expects [controls, treatment] for prediction."
                )

            controls = _make_2d(inputs[0])
            treatment = _make_2d(inputs[1])
            pred_inputs = [controls, treatment]

        else:
            raise ValueError(
                f"Unsupported number of response inputs: {n_response_inputs}"
            )

        return self.response_model.predict(pred_inputs, **kwargs)

    def save(self, filepath, **kwargs):
        """
        Save the response network.
        """
        return self.response_model.save(filepath, **kwargs)

    def summary(self, **kwargs):
        """
        Print the response model summary.
        """
        return self.response_model.summary(**kwargs)