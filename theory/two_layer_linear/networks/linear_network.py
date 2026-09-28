"""Two-layer linear network trained by full-batch gradient descent with weight decay and a log-determinant term on either layer."""
import numpy as np

class LinearNetworkreg:
    def __init__(self, in_dim, hidden_dim, out_dim, lambda_reg=0.0,lambda_det_W1=0.0,lambda_det_W2=0.0, init_w1=None, init_w2=None):

        self.lambda_reg = lambda_reg
        self.lambda_det_W1 =lambda_det_W1
        self.lambda_det_W2 = lambda_det_W2
        if lambda_det_W1 != 0.0 and lambda_det_W2 != 0.0:
            raise ValueError(
                "Log-Determinant regularization (lambda_det_W1 and lambda_det_W2) is mutually exclusive. Only one can be non-zero.")

        if init_w1 is not None and init_w2 is not None:
            self.W1 = init_w1.copy()
            self.W2 = init_w2.copy()
        else:
            self.W1 = np.random.randn(hidden_dim, in_dim)
            self.W2 = np.random.randn(out_dim, hidden_dim)

    def forward(self, x):
        self.z = self.W2 @ self.W1 @ x
        return self.z

    def backward(self, x, y, learning_rate):
        N = x.shape[1]
        forward = self.W2 @ self.W1 @ x

        data_grad_W1 = 1 / N * self.W2.T @ (forward - y) @ x.T
        data_grad_W2 = 1 / N * (forward - y) @ x.T @ self.W1.T

        reg_grad_W1 = self.lambda_reg*  self.W1
        reg_grad_W2 = self.lambda_reg * self.W2

        det_grad_W2 = 0.0
        if self.lambda_det_W2 != 0.0:
            try:
                W2_inv_T = np.linalg.inv(self.W2.T)
                det_grad_W2 =  self.lambda_det_W2 * W2_inv_T
            except np.linalg.LinAlgError:
                print("Warning: W2 is singular. Skipping Log-Det gradient update for this step.")
                det_grad_W2 = 0.0
            except ValueError:
                print("Warning: Determinant of W2 is non-positive. Skipping Log-Det gradient update for this step.")
                det_grad_W2 = 0.0

        det_grad_W1 = 0.0
        if self.lambda_det_W1 != 0.0:
            try:
                W1_inv_T = np.linalg.inv(self.W1.T)
                det_grad_W1 = self.lambda_det_W1 * W1_inv_T
            except np.linalg.LinAlgError:
                W1_inv_T = np.linalg.pinv(self.W1.T)
                det_grad_W1 = self.lambda_det_W1 * W1_inv_T
            except ValueError:
                print("Warning: Determinant of W1 is non-positive. Skipping Log-Det gradient update for this step.")
                det_grad_W1 = 0.0

        dW1 = data_grad_W1 + reg_grad_W1 + det_grad_W1
        dW2 = data_grad_W2 + reg_grad_W2 + det_grad_W2

        self.W2 -= learning_rate * dW2
        self.W1 -= learning_rate * dW1

    def train(self, X_train, Y_train, epochs, learning_rate):
        w1s = []
        w2s = []
        losses = []
        for _ in range(epochs):
            loss = np.mean((self.forward(X_train) - Y_train) ** 2)

            reg_loss = loss + self.lambda_reg / 2 * (
                        np.linalg.norm(self.W1, ord='fro') ** 2 + np.linalg.norm(self.W2, ord='fro') ** 2)
            losses.append(reg_loss)

            w1s.append(self.W1.copy())
            w2s.append(self.W2.copy())
            self.backward(X_train, Y_train, learning_rate)
        return w1s, w2s, losses

    def train_fast(self, X_train, Y_train, epochs, learning_rate):
        losses = []
        for _ in range(epochs):
            N = X_train.shape[1]

            data_loss = 1 / (2 * N) * np.linalg.norm(self.W2 @ self.W1 @ X_train - Y_train, ord='fro') ** 2

            reg_term = self.lambda_reg / 2 * (
                        np.linalg.norm(self.W1, ord='fro') ** 2 + np.linalg.norm(self.W2, ord='fro') ** 2)

            loss = data_loss + reg_term

            losses.append(loss)
            self.backward(X_train, Y_train, learning_rate)
        return losses

class LinearNetworkRepNorm:
    """
    Two-layer linear network with representation-norm regularizer (non-whitened MRNS):

        L = 1/(2N) ||W2 W1 X - Y||_F^2
          + lambda_reg * (||W1 X||_F^2 + ||W2||_F^2)
          + lambda_det * log det(W2 W2^T)

    Omega1 = W1 X (hidden representations) replaces W1 in the balancing condition,
    so the balance is between ||W1 X||_F^2 and ||W2||_F^2.

    Gradients of regularizer:
        dR/dW1 = 2 * lambda_reg * W1 @ X @ X.T
        dR/dW2 = 2 * lambda_reg * W2 + 2 * lambda_det * W2^{-T}

    Optimal singular values at the MRNS solution:
        S1^2 = S_lam - lam/2 * I,  S2^2 = S_lam + lam/2 * I
        S_lam = sqrt(S^2 + lam^2/4 * I)
    where S are singular values of Y X^+.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, lambda_reg=0.0, lambda_det=0.0,
                 init_w1=None, init_w2=None):
        self.lambda_reg = lambda_reg
        self.lambda_det = lambda_det

        if init_w1 is not None and init_w2 is not None:
            self.W1 = init_w1.copy()
            self.W2 = init_w2.copy()
        else:
            self.W1 = np.random.randn(hidden_dim, in_dim)
            self.W2 = np.random.randn(out_dim, hidden_dim)

    def forward(self, x):
        self.z = self.W2 @ self.W1 @ x
        return self.z

    def backward(self, x, y, learning_rate):
        N = x.shape[1]
        forward = self.W2 @ self.W1 @ x

        data_grad_W1 = 1 / N * self.W2.T @ (forward - y) @ x.T
        data_grad_W2 = 1 / N * (forward - y) @ x.T @ self.W1.T

        reg_grad_W1 = 2 * self.lambda_reg * self.W1 @ x @ x.T
        reg_grad_W2 = 2 * self.lambda_reg * self.W2

        det_grad_W2 = 0.0
        if self.lambda_det != 0.0:
            try:
                det_grad_W2 = 2 * self.lambda_det * np.linalg.inv(self.W2.T)
            except np.linalg.LinAlgError:
                print("Warning: W2 is singular. Skipping log-det gradient.")

        dW1 = data_grad_W1 + reg_grad_W1
        dW2 = data_grad_W2 + reg_grad_W2 + det_grad_W2

        self.W2 -= learning_rate * dW2
        self.W1 -= learning_rate * dW1

    def train(self, X_train, Y_train, epochs, learning_rate):
        N = X_train.shape[1]
        w1s, w2s, losses = [], [], []
        for _ in range(epochs):
            loss = 1 / (2 * N) * np.linalg.norm(self.forward(X_train) - Y_train, 'fro') ** 2
            losses.append(loss)
            w1s.append(self.W1.copy())
            w2s.append(self.W2.copy())
            self.backward(X_train, Y_train, learning_rate)
        return w1s, w2s, losses
