"""Two-layer ReLU teacher and student networks, the training loop with the optional SACReg term, the two-layer NTK feature map, kernel distance from initialization, the held-out test loss."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

class TeacherNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TeacherNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, 1, bias=False)

        self._init_weights(input_size, hidden_size)

    def _init_weights(self, input_size, hidden_size):

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

        self._init_weights(input_size, hidden_size, scale, symmetrize=symmetrize)

        assert leak_parameter is None or isinstance(leak_parameter, float)
        self.leak_parameter = leak_parameter

    def _init_weights(self, input_size, hidden_size, scale, symmetrize=False):

        if symmetrize:
          _l = hidden_size // 2
          w1 = torch.randn(_l, input_size)
          w1 = w1 / torch.norm(w1, dim=1, keepdim=True) * scale * self.alpha
          w1 = w1.repeat(2, 1)
 
          w2 = torch.sign(torch.randn(_l,)) * scale / self.alpha

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
        W = trajectory[t]['W']
        a = trajectory[t]['a']
        assert a.shape[0] == 1, f"This was giving trouble {a.shape}"
        neuron_dirs[t] = W
        neuron_sign[t] = a[0]

    return neuron_dirs, neuron_sign

def train(student, criterion, optimizer, inputs, labels, n_iter=100, checkpoint_frequency=1_000,
          checkpoints_to_save=None, conditioner=None, w_reg=0.0):
    """`conditioner`/`w_reg`: when `conditioner` is not None, adds `w_reg * conditioner(h)` to
    the loss, where `h = relu(fc1(inputs))` (the post-nonlinearity hidden layer) — e.g. an
    `sslgap.methods._common.SACReg` instance. `checkpoints_to_save`: optional
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

        if is_checkpoint:
            with torch.no_grad():
                W = student.fc1.weight.data.detach().clone().numpy()
                a = student.fc2.weight.data.detach().clone().numpy()
                trajectory.append({'W':W, 'a':a})

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

        if is_checkpoint:
            with torch.no_grad():
                losses.append(loss.item())
                if conditioner is not None:
                    reg_losses.append(reg.item())
                preds.append(outputs_student.detach().clone().numpy())

    if conditioner is None:
        return trajectory, losses, preds
    return trajectory, losses, preds, reg_losses

def relu(x, leak_parameter=None):
    if leak_parameter is not None:
        x[x < 0] = leak_parameter * x[x < 0]
        return x
    else:
        return np.maximum(0, x)

def relu_grad(x, leak_parameter=None):

    x[x >= 0] = 1

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

    features[:, :m] = relu(X @ W.T, leak_parameter=leak_parameter)

    for _i in range(n):
        features[_i, m:] = np.outer(a * relu_grad(W @ X[_i], leak_parameter=leak_parameter), X[_i]).reshape(-1,)

    return features

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

    inputs, labels = np.float64(inputs.detach().numpy()), np.float64(labels.detach().numpy())

    preds = relu(inputs @ W.T, leak_parameter=leak_parameter) @ a
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

    for dir in ["top", "bottom", "right", "left"]:
        ax.spines[dir].set_linewidth(3)
    
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

def kernel_distance_from_initial(K):
    """
    K: (n_iter, n, n) where n is the number of samples in the training dataset
    """

    normalized_K = K / np.sqrt((K**2).sum(axis=(1, 2), keepdims=True))
    alignment = 1. - (normalized_K * normalized_K[0][None]).sum(axis=(1, 2))

    print(f"0: {alignment[0]}, n negative: {(alignment < 0).sum()}")

    return alignment

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

    if mode == "W":
        assert False, "Not using right now"
        features = features[:, :, m:]

    K = features @ features.transpose(0, 2, 1)

    return K

def get_alpha(delta, scale):
    """This assumes that \|w\| and |a| are 1 at the "base" setting
    """
    alpha = np.sqrt((np.sqrt(4 + (delta/scale**2)**2) - delta/scale**2) / 2)
    return alpha

