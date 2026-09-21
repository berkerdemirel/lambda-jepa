import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class TeacherNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TeacherNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)  # Assuming no bias
        self.fc2 = nn.Linear(hidden_size, 1, bias=False)

        # Initialize weights
        self._init_weights(input_size, hidden_size)

    def _init_weights(self, input_size, hidden_size):

        # Using the somewhat ad-hoc initialization from Chizat:
        # - hidden neurons are random vectors on the sphere (w1)
        # - read-out weights are random \pm 1 (w2)
        w1 = torch.randn(hidden_size, input_size)
        w1 = w1 / torch.norm(w1, dim=1, keepdim=True)
        with torch.no_grad():
          self.fc1.weight.copy_(w1)
        w2 = torch.randn(hidden_size,)
        with torch.no_grad():
          self.fc2.weight.copy_(torch.sign(w2))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class StudentNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, scale, alpha=1., symmetrize=False, leak_parameter=None):
        super(StudentNetwork, self).__init__()
        if symmetrize:
          assert (hidden_size % 2) == 0

        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, 1, bias=False)
        self.alpha = alpha

        # Initialize weights
        self._init_weights(input_size, hidden_size, scale, symmetrize=symmetrize)

        # If `leak_parameter` is None we will use ReLU;
        # as `leak_parameter` approaches 1-, model becomes linear, approaches 0+, model becomes ReLU
        assert leak_parameter is None or isinstance(leak_parameter, float)
        self.leak_parameter = leak_parameter

    def _init_weights(self, input_size, hidden_size, scale, symmetrize=False):

        # Hidden neuron directions are initialized to be unit vectors
        # FIXME: Chizat text says Gaussian
        # Readouts were sampled as Unif[0, 1] before, e.g.:
        # w2 = torch.rand(hidden_size,) * scale

        # Symmetrize if desired
        if symmetrize:
          _l = hidden_size // 2
          w1 = torch.randn(_l, input_size)
          w1 = w1 / torch.norm(w1, dim=1, keepdim=True) * scale * self.alpha
          w1 = w1.repeat(2, 1)
 
          w2 = torch.sign(torch.randn(_l,)) * scale / self.alpha

          # Really for this kind of initialization would want to balance per-layer
          # w2 = torch.randn(_l,) * scale / self.alpha

          w2 = w2.repeat(2)
          w2[_l:] *= -1
        else:
          w1 = torch.randn(hidden_size, input_size)
          w1 = w1 / torch.norm(w1, dim=1, keepdim=True) * scale * self.alpha
          w2 = torch.sign(torch.randn(hidden_size,)) * scale / self.alpha

        with torch.no_grad():
          self.fc1.weight.copy_(w1)
        with torch.no_grad():
          self.fc2.weight.copy_(w2)

    def forward(self, x, return_hidden=False):
        if self.leak_parameter is None:
            h = torch.relu(self.fc1(x))
        else:
            h = torch.nn.LeakyReLU(self.leak_parameter)(self.fc1(x))
        out = self.fc2(h)
        if return_hidden:
            return out, h
        return out


def neuron(trajectory, input_size, hidden_size):
    num_steps = len(trajectory)

    neuron_dirs = np.zeros((num_steps, hidden_size, input_size))
    neuron_sign = np.zeros((num_steps, hidden_size))

    for t in range(num_steps):
        W = trajectory[t]['W']  # (hidden_size, input_size)
        a = trajectory[t]['a']  # (1, hidden_size)
        assert a.shape[0] == 1, f"This was giving trouble {a.shape}"
        neuron_dirs[t] = W
        neuron_sign[t] = a[0]

    return neuron_dirs, neuron_sign


# def train(student, criterion, optimizer, inputs, labels, epochs=100, return_all=False, checkpoint_frequency=1_000):
def train(student, criterion, optimizer, inputs, labels, n_iter=100, checkpoint_frequency=1_000,
          checkpoints_to_save=None, conditioner=None, w_reg=0.0):
    """`conditioner`/`w_reg`: when `conditioner` is not None, adds `w_reg * conditioner(h)` to
    the loss, where `h = relu(fc1(inputs))` (the post-nonlinearity hidden layer) — e.g. an
    `sslgap.methods._common.SpectralConditioner` instance. `checkpoints_to_save`: optional
    explicit iterations to checkpoint at (mirrors `fit_nn_gd`); `None` keeps the old
    `checkpoint_frequency`-based behavior.
    """
    trajectory, losses, reg_losses, preds = [], [], [], []
    student.train()
    for _it in range(n_iter + 1):

        if checkpoints_to_save is None:
            is_checkpoint = (_it % checkpoint_frequency) == 0
        else:
            is_checkpoint = _it in checkpoints_to_save

        # NOTE: Suppose `checkpoint_frequency=1000`; then will save at start of first it,
        # start of 1000 (i.e. after 1000 gradient steps)
        if is_checkpoint:
            with torch.no_grad():
                W = student.fc1.weight.data.detach().clone().numpy()
                a = student.fc2.weight.data.detach().clone().numpy()
                trajectory.append({'W':W, 'a':a})

        # One optimizer step
        optimizer.zero_grad()
        if conditioner is None:
            outputs_student = student(inputs)
            loss = criterion(outputs_student, labels)
        else:
            outputs_student, h = student(inputs, return_hidden=True)
            reg = conditioner(h)
            loss = criterion(outputs_student, labels) + w_reg * reg
        loss.backward()
        optimizer.step()

        # NOTE: Losses and predictions *are* those computed on parameters saved above
        if is_checkpoint:
            with torch.no_grad():  # Shouldn't be necessary? Just added.
                losses.append(loss.item())
                if conditioner is not None:
                    reg_losses.append(reg.item())
                preds.append(outputs_student.detach().clone().numpy())

    if conditioner is None:
        return trajectory, losses, preds
    return trajectory, losses, preds, reg_losses


def relu(x, leak_parameter=None):
    # TODO: rename
    if leak_parameter is not None:
        x[x < 0] = leak_parameter * x[x < 0]
        return x
    else:
        return np.maximum(0, x)


def relu_grad(x, leak_parameter=None):
    # TODO: rename

    x[x >= 0] = 1

    # Before:
    # x[x < 0] = 0
    # Now:
    leak = 0 if leak_parameter is None else leak_parameter
    x[x < 0] = leak

    return x


def get_features_two_layer_relu(W, a, X, leak_parameter=None):
    """
    Returned value is designed such that calling `.reshape(m, d)` on dimensions
    [m:] is the reshape that agrees with the dimensions of W
    """
    n, d = X.shape
    m = W.shape[0]
    features = np.zeros((n, m + m * d), dtype=np.float32)

    # ith feature (ith row of `features`) only depends on x_i (ith row of X)
    # (this is for term from gradient w.r.t. a)
    features[:, :m] = relu(X @ W.T, leak_parameter=leak_parameter)

    # Eh being lazy, vectorize
    for _i in range(n):
        # NOTE: a has a leading dimension of 1, but that works out fine
        features[_i, m:] = np.outer(a * relu_grad(W @ X[_i], leak_parameter=leak_parameter), X[_i]).reshape(-1,)

    # _, S, _ = np.linalg.svd(features)
    # _, S, _ = np.linalg.svd(features[:, m:])
    # print(S)

    return features


# This should likely fly
# def fit_nn_gd(W0, a0, X, y, leak_parameter=None, lr=1e-4, n_iter=10_000, checkpoint_frequency=1_000):
# NOTE: May 31; changing to 100
def fit_nn_gd(W0, a0, X, y, leak_parameter=None, lr=1e-4, n_iter=10_000, checkpoint_frequency=100, checkpoints_to_save=None):
    n, _ = X.shape

    W = copy.deepcopy(W0)
    a = copy.deepcopy(a0)[0]

    # for _p in (W, a, X, y):
    #     assert _p.dtype == np.float32

    # W = np.float64(W)
    # a = np.float64(a)
    # X = np.float64(X)
    # y = np.float64(y)

    trajectory, losses, all_preds = [], [], []

    for _i in range(n_iter + 1):
        # NOTE: does save on first iteration (keeping in line with torch code). Saved predictions and losses
        # are those obtained with saved parameters.
        if ((_i % checkpoint_frequency) == 0 or _i == n_iter):
            preds = relu(X @ W.T, leak_parameter=leak_parameter) @ a  # assume a is just shape (m,)
            loss = ((preds - y[:, 0])**2).mean()

            all_preds.append(preds)
            losses.append(loss)

            # Maybe do always when `checkpoints_to_save` is None
            if checkpoints_to_save is None:
                trajectory.append({"W": copy.deepcopy(W), "a": copy.deepcopy(a)[None]})
        
        # This is maybe a bit suboptimal, but just saving checkpoints that are asked for
        if checkpoints_to_save is not None and _i in checkpoints_to_save:
            trajectory.append({"W": copy.deepcopy(W), "a": copy.deepcopy(a)[None]})

        # Gradient step
        _lin = X @ W.T
        activations = relu(_lin, leak_parameter=leak_parameter)
        preds = activations @ a
        residual = 2 * (y[:, 0] - preds)

        a_grad = -activations.T @ residual
        # NOTE: this `outer` is already keeping 2/n scaling
        # print(np.outer(a, residual).shape)
        W_grad = -(np.outer(a, residual) * relu_grad(_lin.T, leak_parameter=leak_parameter)) @ X  # most expensive call in this loop (around 1 ms)

        W -= lr / n * W_grad
        a -= lr / n * a_grad
        
    return trajectory, losses, all_preds


def fit_linearization_gd(W0, a0, X, y, leak_parameter=None, lr=1e-4, n_iter=10_000, checkpoint_frequency=1_000):
    # print(f"W0={W0.dtype}, a0={a0.dtype}, X={X.dtype}, y={y.dtype}")
    n, d = X.shape
    m = W0.shape[0]

    features = get_features_two_layer_relu(W0, a0, X, leak_parameter=leak_parameter)
    
    theta = np.zeros(features.shape[1], dtype=np.float32)
    assert y.shape[1] == 1

    # print(f"features={features.dtype}, theta={theta.dtype}")

    trajectory, losses, all_preds = [], [], []

    # Track initial parameters, predictions, and loss
    # preds = features @ theta
    # loss = ((preds - y[:, 0])**2).mean()  # Before was hard-coding: losses.append((y[:, 0]**2).mean())
    # trajectory.append({"W": W0, "a": a0})
    # losses.append(loss)
    # all_preds.append(preds)

    for _i in range(n_iter + 1):
        # NOTE: does save on first iteration (keeping in line with torch code). Saved predictions and losses
        # are those obtained with saved parameters.
        if (_i % checkpoint_frequency) == 0:
            preds = features @ theta
            loss = ((preds - y[:, 0])**2).mean()
            W = theta[m:].reshape(m, d) + W0
            a = theta[None, :m] + a0  # keep this with leading dimension 1
            assert a.shape[0] == 1

            all_preds.append(preds)
            losses.append(loss)
            trajectory.append({"W": W, "a": a})

        # Gradient step
        residual = 2/n * (y[:, 0] - features @ theta)  # NOTE: changed to times 2
        theta -= lr * (-features.T @ residual)
        
    return trajectory, losses, all_preds


def spherical_coordinates(W, a, scale_by_a=False):
    """
    W and a:
        either (n_steps, m, d) and (n_steps, m) respectively
        or (m, d) and (1, m) 

    m is number of hidden neurons in the student
    """
    # Handling here could be more graceful (without if/else) but it seems ok
    if len(W.shape) == 3:
        assert a.shape == (W.shape[0], W.shape[1])
        if scale_by_a:
            neuron_dirs = W * np.abs(a)[:, :, None]
        else:
            neuron_dirs = W
        radius = np.linalg.norm(neuron_dirs, axis=2)
        angle = np.arctan2(neuron_dirs[:, :, 1], neuron_dirs[:, :, 0])
        angle[angle < 0] += 2 * np.pi
    elif len(W.shape) == 2:
        assert a.shape == (1, W.shape[0])
        if scale_by_a:
            neuron_dirs = W * np.abs(a).T
        else:
            neuron_dirs = W
        radius = np.linalg.norm(neuron_dirs, axis=1)
        angle = np.arctan2(neuron_dirs[:, 1], neuron_dirs[:, 0])
        angle[angle < 0] += 2 * np.pi
    else:
        raise ValueError
    return radius, angle


def test_one_hidden_layer_relu(W, a, teacher, n_test=10_000, test_seed=None, leak_parameter=None):

    assert test_seed is not None
    torch.manual_seed(test_seed)

    assert W.dtype == np.float64 and a.dtype == np.float64

    m, d = W.shape
    assert a.shape == (m,)
    inputs = torch.randn(n_test, d)
    inputs = inputs / torch.norm(inputs, dim=1, keepdim=True)
    with torch.no_grad():
        labels = teacher(inputs)

    # NOTE: changing to double precision
    inputs, labels = np.float64(inputs.detach().numpy()), np.float64(labels.detach().numpy())

    preds = relu(inputs @ W.T, leak_parameter=leak_parameter) @ a  # assume a is just shape (m,)
    assert labels.shape[1] == 1
    loss = ((preds - labels[:, 0])**2).mean()
    return loss


def style_heatmaps(ax, xlabels=True, ylabels=True, xlim=None, ylim=None, labelsize=24):
    if xlabels:
        ax.tick_params(axis="x", which="both", bottom=True, top=False,
                       labelbottom=True, left=True, right=False,
                       labelleft=True, direction='out',length=7,width=1.5,pad=0,
                       labelsize=labelsize,labelrotation=45)
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    else:
        ax.tick_params(axis="x", which="both", bottom=True, top=False,
                       labelbottom=True, left=True, right=False,
                       labelleft=True, direction='out',length=7,width=1.5,pad=0,
                       labelsize=labelsize,labelrotation=45)
        ax.set_xlabel("")
    if ylabels:
        ax.tick_params(axis="y", which="both", bottom=True, top=False,
                   labelbottom=True, left=True, right=False,
                   labelleft=True, direction='out',length=7,width=1.5,pad=4,
                   labelsize=labelsize)   
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    else:
        ax.tick_params(axis="y", which="both", bottom=True, top=False,
                   labelbottom=False, left=True, right=False,
                   labelleft=False, direction='out',length=7,width=1.5,pad=4,
                   labelsize=labelsize)
        ax.set_ylabel("")
    ax.xaxis.offsetText.set_fontsize(20)

    # Boundary
    for dir in ["top", "bottom", "right", "left"]:
        ax.spines[dir].set_linewidth(3)
    
    # Limits
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def kernel_distance_from_initial(K):
    """
    K: (n_iter, n, n) where n is the number of samples in the training dataset
    """
    # K0 = K[0]
    # kernel_norms = np.sqrt((K**2).sum(axis=(1, 2)))
    # alignment = 1 - (K * K0[None]).sum(axis=(1, 2)) / (kernel_norms * kernel_norms[0])

    normalized_K = K / np.sqrt((K**2).sum(axis=(1, 2), keepdims=True))
    alignment = 1. - (normalized_K * normalized_K[0][None]).sum(axis=(1, 2))  # second term is (1, n, n)

    print(f"0: {alignment[0]}, n negative: {(alignment < 0).sum()}")

    return alignment


def test_kernel(W_lin, a_lin, W_nn, a_nn, W0, a0, n_test):
    """
    W0 and a0 needed to get estimator for kernel model, since what we're learning is a difference
    from the initialization
    """

    m, d = W_lin.shape
    assert W_nn.shape == W_lin.shape

    # Sample data
    X = np.random.randn(n_test, d)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    # Get features for kernel
    features_lin = get_features_two_layer_relu(W0, a0, X)

    theta_lin = np.zeros(m + m * d)
    theta_lin[:m] = a_lin - a0
    theta_lin[m:] = (W_lin - W0).reshape(-1,)
    preds_lin = features_lin @ theta_lin

    preds_nn = relu(X @ W_nn.T) @ a_nn

    return ((preds_lin - preds_nn)**2).mean()


def compute_hamming_distance(W, X):
    """Just get trajectory of hamming distance
    """
    # Verify shapes
    n_steps, m, d = W.shape
    assert X.shape[1] == d
    n = X.shape[0]

    print(X.shape, W.shape)

    # (n_steps, n, m)
    signs = np.sign(X[None] @ W.transpose(0, 2, 1))  # check broadcasting on X
    assert signs.shape == (n_steps, n, m)

    signs_start = signs[0][None]

    # (n_steps, n, m)
    dists = (signs != signs_start).mean(axis=1).sum(axis=1)

    print(f"dists has shape {dists.shape}")

    return dists


def get_kernel_trajectory(W, a, X, mode=None):
    """Compute the kernel trajectory 

    Parameters
    ----------
    W: (n_steps, m, d)
    a: (n_steps, m)
    X: (n, d)

    Returns
    -------
    K: (n_steps, n, n)

    Notes
    -----
    This will be too slow likely, loops everywhere

    Also consumes a bit too much memory to track all the kernels
    """
    n, d = X.shape
    n_steps, m, _ = W.shape
    assert W.shape[2] == d

    features = []
    for _i in range(n_steps):
        features.append(get_features_two_layer_relu(W[_i], a[_i], X))
    features = np.array(features)
    assert features.shape == (n_steps, n, m + m * d)

    # Restrict to only W features if requested
    if mode == "W":
        assert False, "Not using right now"
        features = features[:, :, m:]

    # Kernel as a function of time
    K = features @ features.transpose(0, 2, 1)

    return K


def plot_trajectory(W, a, ax):
    # Verify shapes
    n_steps, m, d = W.shape
    assert d == 2
    assert a.shape == (n_steps, m)

    betas = W * np.abs(a[:, :, None])
    # betas = W

    # Plot one neuron's trajectory at a time
    for i in range(m):
        final_a_i_sign = np.sign(a[-1, i])
        ax.plot(betas[:, i, 0], betas[:, i, 1], lw=0.5, c="k", alpha=0.5)
        color = 'blue' if final_a_i_sign >= 0 else 'red'  # FIXME: flip these?
        ax.scatter(betas[-1, i, 0], betas[-1, i, 1], color=color, s=8, alpha=0.5)


def draw_teacher_dirs(W, a, ax):
    m0 = W.shape[0]
    assert a.shape == (1, m0)
    for i in range(m0):
        w_i = W[i, :]
        a_i = a[0, i]
        ax.plot((0, np.abs(a_i) * w_i[0]), (0, np.abs(a_i) * w_i[1]), c='k')
    _t = np.linspace(0, 2 * np.pi, 1_000)
    ax.plot(np.cos(_t), np.sin(_t), linewidth=1)


def get_alpha(delta, scale):
    """This assumes that \|w\| and |a| are 1 at the "base" setting
    """
    alpha = np.sqrt((np.sqrt(4 + (delta/scale**2)**2) - delta/scale**2) / 2)
    return alpha


# TODO: make it so that this can take an alpha *or* a delta
# def get_results(scale, alpha, seed, lr, n_iter, teacher, n_samples, m, input_size, leak_parameter, checkpoints_to_save):
def get_results(scale, delta, seed, lr, n_iter, teacher, n_samples, m, input_size, leak_parameter, checkpoints_to_save):
    assert leak_parameter is None

    results = {}

    # This should work ok in a subprocess
    torch.manual_seed(seed)

    # Instantiate student model (TODO: change symmetrize)
    alpha = get_alpha(delta, scale)
    student = StudentNetwork(input_size, m, scale, alpha=alpha, symmetrize=True, leak_parameter=leak_parameter)

    # Sample data
    inputs = torch.randn(n_samples, input_size)
    inputs = inputs / torch.norm(inputs, dim=1, keepdim=True)
    with torch.no_grad():
        labels = teacher(inputs)
    print(f"input mean: {inputs.mean()}, label mean: {labels.mean()}")

    # Extract initial student network paramters
    with torch.no_grad():
        W0, a0 = student.fc1.weight.detach().clone().numpy(), student.fc2.weight.detach().clone().numpy()
    
    W0, a0 = np.float64(W0), np.float64(a0)

    # # Fit linearization using gradient descent
    # trajectory_lin_gd, losses_lin_gd, preds_lin_gd = fit_linearization_gd(
    #     W0, a0, inputs.detach().clone().numpy(), labels.detach().clone().numpy(), leak_parameter=leak_parameter, lr=lr, n_iter=n_iter)
    # Ws_lin_gd, as_lin_gd = neuron(trajectory_lin_gd, input_size, m)
    # # assert Ws_lin_gd.shape == (n_iter + 1, m, input_size) and as_lin_gd.shape == (n_iter + 1, m)  # + 1 just since also saving initial
    # results["lin"] = (Ws_lin_gd, as_lin_gd, losses_lin_gd, preds_lin_gd)

    # Try training NN with numpy
    t0 = time.time()
    _results, losses_relu, preds_relu = fit_nn_gd(
        W0, a0, np.float64(inputs.detach().clone().numpy()), np.float64(labels.detach().clone().numpy()), leak_parameter=leak_parameter, lr=lr, n_iter=n_iter, checkpoints_to_save=checkpoints_to_save)
    t1 = time.time()
    print(f"Training took {t1 - t0} s")
    Ws_relu, as_relu = neuron(_results, input_size, m)
    # results["nn"] = (Ws_relu, as_relu, losses_relu, preds_relu)
    results["nn"] = (Ws_relu, as_relu, losses_relu, preds_relu, np.float64(inputs.detach().clone().numpy()))  # also returning data for computing hamming distance
    print(f"Rest took {time.time() - t1} s")

    # Train model
    # criterion = nn.MSELoss()
    # optimizer = optim.SGD(student.parameters(), lr=lr)  # FIXME: change lr with changing scale?
    # _results, losses_relu, preds_relu = train(student, criterion, optimizer, inputs, labels, n_iter)  # , return_all=True)
    # Ws_relu, as_relu = neuron(_results, input_size, m)
    # # assert Ws_relu.shape == (n_iter + 1, m, input_size) and as_relu.shape == (n_iter + 1, m)  # + 1 since also saving initial
    # results["nn"] = (Ws_relu, as_relu, losses_relu, preds_relu)

    return results