import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
def get_lambda_unbalanced_2(lmda, in_dim, hidden_dim, out_dim, scale=1):
    if hidden_dim < min(in_dim, out_dim):
        print('Network cannot be bottlenecked')
        return
    if hidden_dim > max(in_dim, out_dim) and lmda != 0:
        print('hidden_dim cannot be the largest dimension if lambda is not 0')
        return
    sigma = scale
    # add check here for dimensions and lambda
    A = np.random.randn(hidden_dim//2, in_dim)
    variance_A = np.var(A)
    mean_a = np.mean(A)
    B = np.random.randn(out_dim, hidden_dim //2)
    variance = np.var( np.concatenate([A,A.copy()],axis=0))
    mean = np.mean(np.concatenate([A,A.copy()],axis=0))

    w1 = (sigma + int(lmda>0)*lmda) * np.concatenate([A,A.copy()],axis=0)
    variance_w1 = np.var(w1)
    mean_w1 = np.mean(w1)
    w2 = (sigma - int(lmda<0)*lmda) * np.concatenate([B,-B],axis=1)
    l=  w2.T@w2-w1@w1.T
    print(l)
    return w1, w2

def get_lambda_unbalanced_3(lmda, in_dim, hidden_dim, out_dim, scale=1):
    if hidden_dim < min(in_dim, out_dim):
        print('Network cannot be bottlenecked')
        return
    if hidden_dim > max(in_dim, out_dim) and lmda != 0:
        print('hidden_dim cannot be the largest dimension if lambda is not 0')
        return
    sigma = scale
    # add check here for dimensions and lambda
    A = np.random.randn(hidden_dim//2, in_dim)
    B = np.random.randn(out_dim, hidden_dim //2)

    lmda = 2
    w1 = (sigma + int(lmda > 0) * lmda) * np.concatenate([A, A.copy()], axis=0)
    w2 = (sigma - int(lmda < 0) * lmda) * np.concatenate([B, -B], axis=1)
    U, S, Vt = np.linalg.svd(w1)
    a = (sigma + int(lmda > 0) * lmda) - (sigma - int(lmda<0)*lmda)

    l= np.linalg.norm(w2, ord='fro') - np.linalg.norm(w1, ord='fro')
    c =  np.linalg.norm(w2^2, ord='fro') + np.linalg.norm(w1, ord='fro')
    b = (sigma + int(lmda > 0) * lmda)**2 + (sigma - int(lmda < 0) * lmda)**2
    print('hello')
    return w1, w2

def get_lambda_unbalanced(lmda, in_dim, hidden_dim, out_dim, scale=1):
    if hidden_dim < min(in_dim, out_dim):
        print('Network cannot be bottlenecked')
        return
    if hidden_dim > max(in_dim, out_dim) and lmda != 0:
        print('hidden_dim cannot be the largest dimension if lambda is not 0')
        return
    sigma = scale
    # add check here for dimensions and lambda
    A = np.random.randn(hidden_dim//2, in_dim)
    B = np.random.randn(out_dim, hidden_dim //2)
    lmda =  5
    w1 = (sigma + int(lmda > 0) * lmda) * np.concatenate([A, A.copy()], axis=0)
    w2 = (sigma - int(lmda < 0) * lmda) * np.concatenate([B, -B], axis=1)
    U, S, Vt = np.linalg.svd(w1)
    a = (sigma + int(lmda > 0) * lmda) - (sigma - int(lmda<0)*lmda)
    l= np.linalg.norm(w2, ord='fro') - np.linalg.norm(w1, ord='fro')
    c =  np.linalg.norm(w2, ord='fro') + np.linalg.norm(w1, ord='fro')
    b = (sigma + int(lmda > 0) * lmda)**2 + (sigma - int(lmda < 0) * lmda)**2
    print('hello')
    return w1, w2




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

def balanced_weights(in_dim, hidden_dim, out_dim, sigma = 1):
    U, S, V = np.linalg.svd(np.random.randn(hidden_dim, hidden_dim))
    r = U @ V.T

    w1 = sigma * np.random.randn(hidden_dim, in_dim)
    w2 = sigma * np.random.randn(out_dim, hidden_dim)

    U_, S_, V_ = np.linalg.svd(w2 @ w1)
    s = np.sqrt(np.diag(S_))

    lmda = np.trace(w2 @ w1) / hidden_dim

    factor = (- lmda + np.sqrt(lmda ** 2 + 4 * s ** 2)) / 2

    s_2 = np.sqrt(np.diag(np.diag(factor)))

    s2_reshaped = reshape_matrix(s_2, (out_dim, hidden_dim))

    s_1 = np.diag(np.diag(s) / np.diag(s_2))

    s1_reshaped = reshape_matrix(s_1, (hidden_dim, in_dim))

    S_test = s2_reshaped @ s1_reshaped

    w1_out = r @ s1_reshaped @ V_.T 

    w2_out = U_ @ s2_reshaped @ r.T

    q = w1_out @ w1_out.T - w2_out.T @ w2_out

    scale_by = lmda / q[0][0]
    w1_out = scale_by * w1_out
    w2_out = scale_by * w2_out
    q = w1_out @ w1_out.T - w2_out.T @ w2_out

    return w1_out, w2_out, S_test, q


def get_lambda_balanced_aligned(lmda, in_dim, hidden_dim, out_dim, X, Y, sigma=1):
    U, _, Vt = np.linalg.svd(Y @ X.T)

    w1, w2 = get_lambda_balanced(lmda, in_dim, hidden_dim, out_dim)
    U_, _, Vt_ = np.linalg.svd(w2 @ w1)

    init_w2 = U @ U_.T @ w2 
    init_w1 = w1 @ Vt_.T @ Vt

    return init_w1, init_w2


def get_lambda_balanced(lmda, in_dim, hidden_dim, out_dim, sigma=1):
    if hidden_dim < min(in_dim, out_dim):
        print('Network cannot be bottlenecked')
        return
    if hidden_dim > max(in_dim, out_dim) and lmda != 0:
        print('hidden_dim cannot be the largest dimension if lambda is not 0')
        return
    
    #add check here for dimensions and lambda
    w1 = sigma * np.random.randn(hidden_dim, in_dim)
    w2 = sigma * np.random.randn(out_dim, hidden_dim)
    U, S, Vt = np.linalg.svd(w2 @ w1)
    R, _ = np.linalg.qr(np.random.randn(hidden_dim, hidden_dim))
    S2_equal_dim = (np.sqrt((np.sqrt(lmda**2 + 4 * S**2) + lmda) / 2))
    S1_equal_dim = (np.sqrt((np.sqrt(lmda**2 + 4 * S**2) - lmda) / 2))


    if out_dim > in_dim:
        add_terms = np.asarray([np.sqrt(lmda*0) for _ in range(hidden_dim - in_dim)])
        S2 = np.vstack([np.diag(np.concatenate((S2_equal_dim, add_terms))),
                        np.zeros((out_dim - hidden_dim, hidden_dim))]) 
        S1 = np.vstack([np.diag(S1_equal_dim), 
                        np.zeros((hidden_dim - in_dim, in_dim))])
    elif in_dim > out_dim:
        add_terms = np.asarray([-np.sqrt(-lmda*0) for _ in range(hidden_dim-out_dim)])
        S1 = np.hstack([np.diag(np.concatenate((S1_equal_dim, add_terms))),
                        np.zeros((hidden_dim, in_dim - hidden_dim))])
        S2 = np.hstack([np.diag(S2_equal_dim), 
                        np.zeros((out_dim, hidden_dim - out_dim))]) 
        
    else:
        S2 = np.diag(S2_equal_dim)
        S1 = np.diag(S1_equal_dim)

    init_w2 =  U @ S2 @ R.T
    init_w1 = R @ S1 @ Vt

    U, S, V = np.linalg.svd(init_w1)

    return init_w1, init_w2

def whiten(X):

    scaler = StandardScaler()

    X_standardised = scaler.fit_transform(X)
    
    pca = PCA()
    X_pca = pca.fit_transform(X_standardised)

    X_whitened = X_pca / np.sqrt(pca.explained_variance_)

    X_whitened = np.sqrt(X.shape[0] / (X.shape[0] - 1)) * X_whitened

    return X_whitened


def get_reversal_learning_task(cls, items_n, input_dim, output_dim, preprocessing=[]):
    R, _, _ = np.linalg.svd(np.random.normal(size=(items_n, items_n)))
    U, S, VT = np.linalg.svd(np.random.normal(size=(output_dim, input_dim)))

    # Adjust S
    SS = np.diag(S * np.sqrt(1. / S))
    smaller_dim = min(input_dim, output_dim)

    # Construct S0
    S0 = np.vstack([SS, np.zeros((items_n - smaller_dim, smaller_dim))])
    training_data = R @ S0 @ VT
    S0 = np.hstack([SS, np.zeros((smaller_dim, items_n - smaller_dim))])
    training_labels = (U @ S0 @ R.T).T

    validation_data = training_data.copy()
    validation_labels = (-U @ S0 @ R.T).T

    return training_data, training_labels, validation_data, validation_labels

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
    # Flatten the matrices into vectors
    vec_A = A.flatten()
    vec_B = B.flatten()

    # Compute the dot product and norms
    dot_product = np.dot(vec_A, vec_B)
    norm_A = np.linalg.norm(vec_A)
    norm_B = np.linalg.norm(vec_B)

    # Compute cosine similarity
    similarity = dot_product / (norm_A * norm_B)

    return similarity

def kernel_distance(A,B):
    # Extract matrices K_t1 and K_t2
    K_t1 = A
    K_t2 = B

    # Compute the Frobenius inner product
    inner_product = np.sum(K_t1 * K_t2)

    # Compute the Frobenius norms
    norm_t1 = np.linalg.norm(K_t1, 'fro')
    norm_t2 = np.linalg.norm(K_t2, 'fro')

    # Compute the similarity measure
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

def get_random_autoenco_task(batch_size, in_dim, out_dim, Whiten= False):
    X = np.random.randn(batch_size, in_dim)
    Y = np.random.randn(batch_size, out_dim)
    if Whiten:
        X_whitened = whiten(X)
    else:
        X_whitened = X

    return X_whitened.T, X_whitened.T


def get_ntk(w1w1, w2w2, X, out_dim):
    return  np.kron(np.eye(out_dim), X.T @ w1w1 @ X) + np.kron(w2w2, X.T @ X)

def frobenius_norm(A):
    return np.linalg.norm(A, 'fro')

def check_singular_val(S,S1,S2, lmda, in_dim, hidden_dim, out_dim):
    dim = min(in_dim,out_dim)
    S_1 = S1[:dim]
    S_2 = S2[:dim]
    S2_equal_dim = (np.sqrt((np.sqrt(lmda ** 2 + 4 * S ** 2) + lmda) / 2))
    S1_equal_dim = (np.sqrt((np.sqrt(lmda ** 2 + 4 * S ** 2) - lmda) / 2))

    if out_dim > in_dim:
        add_terms = np.asarray([np.sqrt(lmda * 0) for _ in range(hidden_dim - in_dim)])
        S2 = np.vstack([np.diag(np.concatenate((S2_equal_dim, add_terms))),
                        np.zeros((out_dim - hidden_dim, hidden_dim))])
        S1 = np.vstack([np.diag(S1_equal_dim),
                        np.zeros((hidden_dim - in_dim, in_dim))])
        add_terms = np.asarray([np.sqrt(lmda * 0) for _ in range(hidden_dim - in_dim)])
        S2_w = np.vstack([np.diag(np.concatenate((S_2, add_terms))),
                        np.zeros((out_dim - hidden_dim, hidden_dim))])
        S1_w = np.vstack([np.diag(S_1),
                        np.zeros((hidden_dim - in_dim, in_dim))])
    elif in_dim > out_dim:
        add_terms = np.asarray([-np.sqrt(-lmda * 0) for _ in range(hidden_dim - out_dim)])
        S1 = np.hstack([np.diag(np.concatenate((S1_equal_dim, add_terms))),
                        np.zeros((hidden_dim, in_dim - hidden_dim))])
        S2 = np.hstack([np.diag(S2_equal_dim),
                        np.zeros((out_dim, hidden_dim - out_dim))])
        S1_w = np.hstack([np.diag(np.concatenate((S_1, add_terms))),
                        np.zeros((hidden_dim, in_dim - hidden_dim))])
        S2_w = np.hstack([np.diag(S_2),
                        np.zeros((out_dim, hidden_dim - out_dim))])
    else:
        S2 = np.diag(S2_equal_dim)
        S1 = np.diag(S1_equal_dim)
        S2_w = np.diag(S_2)
        S1_w = np.diag(S_1)

    check_1 = S1_w - S1
    check_2 = S2_w - S2

    return    check_1, check_2



def get_lambda_balanced_A(lmda, in_dim, hidden_dim, out_dim, sigma_yx=None, scale=None):
    if hidden_dim < min(in_dim, out_dim):
        # print('Network cannot be bottlenecked')
        # return
        pass
    if hidden_dim > max(in_dim, out_dim) and lmda != 0:
        print('hidden_dim cannot be the largest dimension if lambda is not 0')
        return
    if sigma_yx is None:
        # add check here for dimensions and lambda
        print('error: sigma_yx cannot be None')
    else:
        U, S, Vt = np.linalg.svd(sigma_yx, )

    matrix = np.random.randn(hidden_dim, hidden_dim)

    # Make sure it's invertible (simple way: add identity scaled a bit)
    matrix += hidden_dim * np.eye(hidden_dim)

    # Compute its inverse
    matrix_inv = np.linalg.inv(matrix)

    R, _ = np.linalg.qr(np.random.randn(hidden_dim, hidden_dim))
    A = np.random.normal(loc=0, scale=1, size=(hidden_dim, hidden_dim))
    U_a, S, Vt_a = np.linalg.svd(A)


    S2_equal_dim = (np.sqrt((np.sqrt(lmda ** 2 + 4 * S) + lmda) / 2))
    S1_equal_dim = (np.sqrt((np.sqrt(lmda ** 2 + 4 * S) - lmda) / 2))

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
        # print('hello')
        S2 = np.diag(S2_equal_dim)
        S1 = np.diag(S1_equal_dim)

    U_c, S, Vt_c = np.linalg.svd(A)
    hello = U @ A @ R.T
    hello_2 = R @ A.T @ Vt

    hello.T@hello-hello_2@hello_2.T

    init_w2 = U @ U_a @ np.diag(S) @ Vt_a.T @ R.T
    init_w1 = R @ Vt_a @ S @ U_a.T @ Vt

    init_w2.T@init_w2- init_w1@init_w1.T

    return init_w1, init_w2





def get_lambda_balanced(lmda, in_dim, hidden_dim, out_dim, sigma=1, sigma_yx= None, scale = None):
    if hidden_dim < min(in_dim, out_dim):
       #print('Network cannot be bottlenecked')
       # return
        pass
    if hidden_dim > max(in_dim, out_dim) and lmda != 0:
        print('hidden_dim cannot be the largest dimension if lambda is not 0')
        return
    if sigma_yx is None:
        # add check here for dimensions and lambda
        w1 = sigma * np.random.randn(hidden_dim, in_dim)
        w2 = sigma * np.random.randn(out_dim, hidden_dim)
        U, S, Vt = np.linalg.svd(w2 @ w1 )
        if scale is not None:
            S = np.diag(scale * np.eye(hidden_dim))
            # print('here')
        
    else:
        U, S, Vt= np.linalg.svd(sigma_yx, )
    matrix = np.random.randn(hidden_dim, hidden_dim)

    # Make sure it's invertible (simple way: add identity scaled a bit)
    matrix += hidden_dim * np.eye(hidden_dim)

    # Compute its inverse
    matrix_inv = np.linalg.inv(matrix)

    R, _ = np.linalg.qr(np.random.randn(hidden_dim, hidden_dim))
    #S1 =  np.zeros([hidden_dim,in_dim])
    #S2 = np.zeros([out_dim,hidden_dim])
    #for i in range(min(hidden_dim, in_dim)):
    # S1[i,i] = (np.sqrt((np.sqrt(lmda ** 2 + 4 *  S[i] ** 2) - lmda) / 2))
    # S2[i,i] =    (np.sqrt((np.sqrt(lmda ** 2 + 4 * S[i] ** 2) + lmda) / 2))

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
        # print('hello')
        S2 = np.diag(S2_equal_dim)
        S1 = np.diag(S1_equal_dim)

    init_w2 = U @ S2 @ R.T
    init_w1 = R @ S1 @ Vt

    #init_w2 = U @ S2 @ matrix
    #init_w1 = matrix_inv @ S1 @ Vt
    #np.eye(2,2)* 3
    #matrix.T @  S2.T  @ S2 @ matrix -  matrix_inv @ S1 @S1 .T @ matrix_inv.T

    #init_w2.T @ init_w2 - init_w1 @ init_w1.T
    return init_w1, init_w2

def get_lambda_balanced_aligned(lmda, in_dim, hidden_dim, out_dim, X, Y, sigma=1):
    U, _, Vt = np.linalg.svd(Y @ X.T)

    w1, w2 = get_lambda_balanced(lmda, in_dim, hidden_dim, out_dim)
    U_, _, Vt_ = np.linalg.svd(w2 @ w1)

    init_w2 = U @ U_.T @ w2 
    init_w1 = w1 @ Vt_.T @ Vt

    return init_w1, init_w2

class SingularMatrixError(Exception):
    """Exception raised when a matrix is singular."""
    pass

def get_lambda_balanced_botteld(lmda, in_dim, hidden_dim, out_dim, sigma=1, sigma_yx= None):
# this need changing to be more general
    if hidden_dim < min(in_dim, out_dim):
        pass
    if hidden_dim > max(in_dim, out_dim) and lmda != 0:
        print('hidden_dim cannot be the largest dimension if lambda is not 0')
        return
    if sigma_yx is None:
        # add check here for dimensions and lambda
        w1 = sigma * np.random.randn(hidden_dim, in_dim)
        w2 = sigma * np.random.randn(out_dim, hidden_dim)
        U, S, Vt = np.linalg.svd(w2 @ w1 )
    else:
        U, S, Vt= np.linalg.svd(sigma_yx, )
    R, _ = np.linalg.qr(np.random.randn(hidden_dim, hidden_dim))
    S1 =  np.zeros([hidden_dim,in_dim])
    S2 = np.zeros([out_dim,hidden_dim])
    for i in range(min(hidden_dim, in_dim)):
         S1[i,i] = (np.sqrt((np.sqrt(lmda ** 2 + 4 *  S[i] ** 2) - lmda) / 2))
         S2[i,i] =    (np.sqrt((np.sqrt(lmda ** 2 + 4 * S[i] ** 2) + lmda) / 2))
    init_w2 = U @ S2 @ R.T
    init_w1 = R @ S1@ Vt
    return init_w1, init_w2

import seaborn as sns
import matplotlib as mpl
import numpy as np


# NOTE: parts of this file are recycled from the paper https://openreview.net/forum?id=lJx2vng-KiC.

# The BlindColours class is designed to provide color schemes that are accessible to individuals with color blindness.
# It includes methods to get a list of colors, a diverging colormap, and color steps in specific color ranges.

class BlindColours:
    def __init__(self, reverse_cmap=True):
        # Set the visual style for seaborn plots
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

        # Set the context for seaborn plots to "talk" (a high level of detail)
        sns.set_context("talk")

        # Define a set of colors that are accessible to individuals with color blindness
        hex_colours = [
            "#d65c00", "#0071b2", "#009e73", "#cc78a6", "#e59c00", "#55b2e8", "#efe440", "#000000",  # Original colors
            "#e69f00", "#56b4e9", "#009e73", "#f0e442", "#0072b2", "#d55e00", "#cc79a7", "#999999",
            "#0173b2", "#de8f05", "#029e73", "#d55e00", "#cc78bc", "#ca9161", "#fbafe4", "#ece133",
            "#56b4e9", "#009e73", "#f0e442", "#0072b2", "#d55e00", "#cc79a7", "#aaaaaa", "#4b0092"
        ]
        self.blind_colours = [mpl.colors.to_rgb(h) for h in hex_colours]

        # Define a diverging colormap
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
            div.reverse()  # Reverse the colormap if specified
        self.div_cmap = mpl.colors.ListedColormap(div)

        # Define color steps for specific color ranges
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


# Function to generate zero-balanced weights
def zero_balanced_weights(in_dim, hidden_dim, out_dim, sigma):
    # Generate a random orthogonal matrix r
    r, _, _ = np.linalg.svd(np.random.normal(0., 1., (hidden_dim, hidden_dim)))

    # Initialize weights w1 and w2 with a normal distribution
    w1 = np.random.normal(0., sigma, (hidden_dim, in_dim))
    w2 = np.random.normal(0., sigma, (out_dim, hidden_dim))

    # Perform SVD on the product of w2 and w1
    u, s, vt = np.linalg.svd(w2 @ w1, False)

    # Adjust the singular values
    s = np.diag(np.sqrt(s) * 1.15)

    # Determine the smaller dimension between input and output
    smaller_dim = in_dim if in_dim < out_dim else out_dim

    # Adjust w1 using the orthogonal matrix and singular values
    s0 = np.vstack([s, np.zeros((hidden_dim - smaller_dim, smaller_dim))])
    w1 = r @ s0 @ vt

    # Adjust w2 using the orthogonal matrix and singular values
    s0 = np.hstack([s, np.zeros((smaller_dim, hidden_dim - smaller_dim))])
    w2 = u @ s0 @ r.T

    return w1, w2


# Function to generate balanced weights
def balanced_weights(hidden_dim, sigma=1, lmda=1):
    # Generate a random orthogonal matrix r
    U, S, V = np.linalg.svd(np.random.randn(hidden_dim, hidden_dim))
    r = U @ V.T

    # Initialize weights w1 and w2 with a normal distribution
    w1 = sigma * np.random.randn(hidden_dim, hidden_dim)
    w2 = sigma * np.random.randn(hidden_dim, hidden_dim)

    # Perform SVD on the product of w2 and w1
    U_, S_, V_ = np.linalg.svd(w2 @ w1)
    s = np.sqrt(np.diag(S_))

    # Calculate lambda as the trace of the product of w2 and w1 divided by the hidden dimension
    lmda = np.trace(w2 @ w1) / hidden_dim

    # Calculate the factor for adjusting singular values
    factor = (- lmda + np.sqrt(lmda ** 2 + 4 * s ** 2)) / 2

    # Adjust the singular values
    s_2 = np.sqrt(np.diag(np.diag(factor)))
    s_1 = np.diag(np.diag(s) / np.diag(s_2))

    # Adjust w1 and w2 using the orthogonal matrix and adjusted singular values
    w1_out = r @ s_1 @ V.T
    w2_out = U @ s_2 @ r.T
    S_test = s_2 @ s_1

    # Calculate the difference between w1 and w2
    q = w1_out @ w1_out.T - w2_out.T @ w2_out

    # Scale the weights to balance the network
    scale_by = lmda / q[0][0]
    w1_out = scale_by * w1_out
    w2_out = scale_by * w2_out
    q = w1_out @ w1_out.T - w2_out.T @ w2_out

    return w1_out, w2_out, S_test, q


print(balanced_weights(3))
# S2_equal_dim = (np.sqrt((np.sqrt(lmda ** 2 + 4 * S ** 2) + lmda) / 2))
    # S1_equal_dim = (np.sqrt((np.sqrt(lmda ** 2 + 4 * S ** 2) - lmda) / 2))

    #if out_dim > in_dim:
        #add_terms = np.asarray([np.sqrt(lmda * 0) for _ in range(hidden_dim - in_dim)])
        # S2 = np.vstack([np.diag(np.concatenate((S2_equal_dim, add_terms))),
        #                   np.zeros((out_dim - hidden_dim, hidden_dim))])
        # S1 = np.vstack([np.diag(S1_equal_dim),
        #     np.zeros((hidden_dim - in_dim, in_dim))])
    # elif in_dim > out_dim:
         # add_terms = np.asarray([-np.sqrt(-lmda * 0) for _ in range(hidden_dim - out_dim)])
         # S1 = np.hstack([np.diag(np.concatenate((S1_equal_dim, add_terms))),
         #         np.zeros((hidden_dim, in_dim - hidden_dim))])
         # S2 = np.hstack([np.diag(S2_equal_dim),
         #         np.zeros((out_dim, hidden_dim - out_dim))])

    #    else:
        # print('hello')
         # S2 = np.diag(S2_equal_dim)
            # S1 = np.diag(S1_equal_dim)
