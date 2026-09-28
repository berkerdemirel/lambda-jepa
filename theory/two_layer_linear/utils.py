"""Helpers of the two-layer linear experiment: lambda-balanced weight factorizations, the two-layer linear NTK, kernel distance, the colour palette (BlindColours, adapted from the public code of https://openreview.net/forum?id=lJx2vng-KiC)."""
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def whiten(X):

    scaler = StandardScaler()

    X_standardised = scaler.fit_transform(X)
    
    pca = PCA()
    X_pca = pca.fit_transform(X_standardised)

    X_whitened = X_pca / np.sqrt(pca.explained_variance_)

    X_whitened = np.sqrt(X.shape[0] / (X.shape[0] - 1)) * X_whitened

    return X_whitened

def reshape_matrix(input_matrix, new_shape):
    old_shape = input_matrix.shape

    if new_shape[0] > old_shape[0]:

        new_matrix = np.vstack((input_matrix, np.zeros((new_shape[0] - old_shape[0], old_shape[1]))))
    elif new_shape[0] < old_shape[0]:

        new_matrix = input_matrix[:new_shape[0], :]
    else:
        new_matrix = input_matrix

    if new_shape[1] > old_shape[1]:
        new_matrix = np.hstack((new_matrix, np.zeros((new_shape[0], new_shape[1] - old_shape[1]))))
    elif new_shape[1] < old_shape[1]:
        new_matrix = new_matrix[:, :new_shape[1]]

    return new_matrix

def cosine_similarity(A, B):
    vec_A = A.flatten()
    vec_B = B.flatten()

    dot_product = np.dot(vec_A, vec_B)
    norm_A = np.linalg.norm(vec_A)
    norm_B = np.linalg.norm(vec_B)

    similarity = dot_product / (norm_A * norm_B)

    return similarity

def kernel_distance(A,B):
    K_t1 = A
    K_t2 = B

    inner_product = np.sum(K_t1 * K_t2)

    norm_t1 = np.linalg.norm(K_t1, 'fro')
    norm_t2 = np.linalg.norm(K_t2, 'fro')

    S_t1_t2 = 1 - inner_product / (norm_t1 * norm_t2)

    return S_t1_t2

def get_random_regression_task(batch_size, in_dim, out_dim, Whiten=True):
    X = np.random.randn(batch_size, in_dim)
    Y = np.random.randn(batch_size, out_dim)
    if Whiten:
        X_whitened = whiten(X)
    else:
        X_whitened = X

    return X_whitened.T, Y.T

def get_ntk(w1w1, w2w2, X, out_dim):
    return  np.kron(np.eye(out_dim), X.T @ w1w1 @ X) + np.kron(w2w2, X.T @ X)

def get_lambda_balanced(lmda, in_dim, hidden_dim, out_dim, sigma=1, sigma_yx= None, scale = None):
    if hidden_dim < min(in_dim, out_dim):
        pass
    if hidden_dim > max(in_dim, out_dim) and lmda != 0:
        print('hidden_dim cannot be the largest dimension if lambda is not 0')
        return
    if sigma_yx is None:
        w1 = sigma * np.random.randn(hidden_dim, in_dim)
        w2 = sigma * np.random.randn(out_dim, hidden_dim)
        U, S, Vt = np.linalg.svd(w2 @ w1 )
        if scale is not None:
            S = np.diag(scale * np.eye(hidden_dim))
        
    else:
        U, S, Vt= np.linalg.svd(sigma_yx, )
    matrix = np.random.randn(hidden_dim, hidden_dim)

    matrix += hidden_dim * np.eye(hidden_dim)

    matrix_inv = np.linalg.inv(matrix)

    R, _ = np.linalg.qr(np.random.randn(hidden_dim, hidden_dim))

    S2_equal_dim = (np.sqrt((np.sqrt(lmda**2  + 4 * S) + lmda) / 2))
    S1_equal_dim = (np.sqrt((np.sqrt(lmda**2  + 4 * S ) - lmda) / 2))

    if out_dim > in_dim:
        add_terms = np.asarray([np.sqrt(lmda * 0) for _ in range(hidden_dim - in_dim)])
        S2 = np.vstack([np.diag(np.concatenate((S2_equal_dim, add_terms))),
                        np.zeros((out_dim - hidden_dim, hidden_dim))])
        S1 = np.vstack([np.diag(S1_equal_dim),
                        np.zeros((hidden_dim - in_dim, in_dim))])
    elif in_dim > out_dim:
        add_terms = np.asarray([-np.sqrt(-lmda * 0) for _ in range(hidden_dim - out_dim)])
        S1 = np.hstack([np.diag(np.concatenate((S1_equal_dim, add_terms))),
                        np.zeros((hidden_dim, in_dim - hidden_dim))])
        S2 = np.hstack([np.diag(S2_equal_dim),
                        np.zeros((out_dim, hidden_dim - out_dim))])

    else:
        S2 = np.diag(S2_equal_dim)
        S1 = np.diag(S1_equal_dim)

    init_w2 = U @ S2 @ R.T
    init_w1 = R @ S1 @ Vt

    return init_w1, init_w2

import seaborn as sns
import matplotlib as mpl

class BlindColours:
    def __init__(self, reverse_cmap=True):
        sns.set_style("ticks", {
            'xtick.bottom': True,
            'xtick.top': False,
            'ytick.left': True,
            'ytick.right': False,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'xtick.color': '.1',
            'ytick.color': '.1',
        })

        sns.set_context("talk")

        hex_colours = [
            "#d65c00", "#0071b2", "#009e73", "#cc78a6", "#e59c00", "#55b2e8", "#efe440", "#000000",
            "#e69f00", "#56b4e9", "#009e73", "#f0e442", "#0072b2", "#d55e00", "#cc79a7", "#999999",
            "#0173b2", "#de8f05", "#029e73", "#d55e00", "#cc78bc", "#ca9161", "#fbafe4", "#ece133",
            "#56b4e9", "#009e73", "#f0e442", "#0072b2", "#d55e00", "#cc79a7", "#aaaaaa", "#4b0092"
        ]
        self.blind_colours = [mpl.colors.to_rgb(h) for h in hex_colours]

        div = ['#6d0000', '#720400', '#770900', '#7c0d00', '#821200', '#871600', '#8b1b00', '#901f00', '#952300',
               '#9a2700', '#9f2c00', '#a33000', '#a83400', '#ad3800', '#b13c00', '#b64000', '#bb4500', '#bf4900',
               '#c44d00', '#c85100', '#cc5604', '#cf5b09', '#d3600e', '#d66513', '#d96a18', '#dd6f1d', '#e07422',
               '#e37927', '#e67e2c', '#ea8331', '#ed8836', '#f08d3b', '#f3923f', '#f69744', '#f99b49', '#fda04e',
               '#ffa555', '#feac62', '#fdb26e', '#fdb87a', '#fcbe87', '#fbc492', '#faca9e', '#f9d5b4', '#f8dabf',
               '#f8e0ca', '#f7e5d5', '#f6ebe0', '#f6f0ea', '#ecf2f6', '#e3eef7', '#d9ebf8', '#d0e7f8', '#c6e4f9',
               '#bde0fa', '#b3ddfb', '#a9d9fc', '#9fd6fd', '#95d2fe', '#8bceff', '#85cafc', '#80c6f9', '#7bc2f6',
               '#75bef2', '#70baef', '#6bb6ec', '#66b1e9', '#61ade5', '#5ba9e2', '#56a5df', '#51a1dc', '#4c9dd8',
               '#4799d5', '#4295d2', '#3c91cf', '#378dcb', '#3289c8', '#2d85c5', '#2881c2', '#237dbf', '#2079ba',
               '#1e75b6', '#1c71b1', '#1a6dad', '#1969a8', '#1765a4', '#15619f', '#135d9b', '#115996', '#105592',
               '#0e518e', '#0c4d89', '#0a4a85', '#094681', '#07427d', '#053e79', '#033b74', '#023770', '#00346c']
        if reverse_cmap:
            div.reverse()
        self.div_cmap = mpl.colors.ListedColormap(div)

        oranges = [mpl.colors.to_rgb(h) for h in ['#871500', '#a93700', '#cc5400', '#ef721c', '#ff9c4a']]
        blues = [mpl.colors.to_rgb(h) for h in ['#00356e', '#005492', '#0975b7', '#4895d9', '#70b6fd']]
        greens = [mpl.colors.to_rgb(h) for h in ['#003e1d', '#005e39', '#008057', '#09a378', '#46c698']]
        self.colour_steps = [oranges, blues, greens]

    def get_colours(self):
        return self.blind_colours

    def get_div_cmap(self):
        return self.div_cmap

    def get_colour_steps(self):
        return self.colour_steps

