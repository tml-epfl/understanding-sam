import numpy as np


def loss(X, y_hat, y):
    return np.linalg.norm(y_hat - y)**2 / (2*X.shape[0])
    

def GD(X, y, X_test, y_test, gamma, u0, v0, iters_loss, num_iter, thresholds=[-1], wd=0, balancedness=0, normalized_gd=False):
    train_losses, test_losses = [], []
    u, v = u0, v0
    for i in range(num_iter):
        y_hat = X @ (u * v)
        Xerr = X.T@(y_hat - y)
        grad_u, grad_v = (Xerr * v) / X.shape[0], (Xerr * u) / X.shape[0]
        if normalized_gd and (np.linalg.norm(grad_u) > 0 and np.linalg.norm(grad_v) > 0):
            grad_u, grad_v = grad_u / np.linalg.norm(grad_u), grad_v / np.linalg.norm(grad_v)
        
        u = u - gamma * grad_u - wd*u - balancedness*u*(2*(np.abs(u)>np.abs(v))-1) 
        v = v - gamma * grad_v - wd*v - balancedness*v*(2*(np.abs(v)>np.abs(u))-1) 

        if i in thresholds:
            gamma = gamma/2

        if i in iters_loss:
            train_losses += [loss(X, X @ (u * v), y)]
            test_losses += [loss(X_test, X_test @ (u * v), y_test)]

    return train_losses, test_losses, u, v


def SGD(X, y, X_test, y_test, gamma, u_0, v_0, iters_loss, num_iter):
    train_losses, test_losses = [], []
    u, v = u_0, v_0
    for i in range(num_iter):
        i_t = np.random.randint(X.shape[0])
        error = X[i_t] @ (u * v) - y[i_t] 
        Xerr = error * X[i_t]
        grad_u, grad_v = Xerr * v,  Xerr * u
        
        u = u - gamma * grad_u   # gradient step
        v = v - gamma * grad_v   # gradient step

        if i in iters_loss:
            train_losses += [loss(X, X @ (u * v), y)]
            test_losses += [loss(X_test, X_test @ (u * v), y_test)]

    return train_losses, test_losses, u, v


def n_SAM_GD(X, y, X_test, y_test, gamma, u0, v0, iters_loss, num_iter, rho):
    train_losses, test_losses = [], []
    u, v = u0, v0
    for i in range(num_iter):
        y_hat = X @ (u * v)
        Xerr = X.T @ (y_hat - y)
        u_sam, v_sam = u + rho * (Xerr * v) / X.shape[0], v + rho * (Xerr * u) / X.shape[0]
        
        Xerr_sam = X.T @ (X @ (u_sam * v_sam) - y)
        grad_u_sam, grad_v_sam = (Xerr_sam * v_sam) / X.shape[0], (Xerr_sam * u_sam) / X.shape[0]
        
        u = u - gamma * grad_u_sam   # gradient step
        v = v - gamma * grad_v_sam   # gradient step

        if i in iters_loss:
            train_losses += [loss(X, X @ (u * v), y)]
            test_losses += [loss(X_test, X_test @ (u * v), y_test)]

    return train_losses, test_losses, u, v


def one_SAM_GD(X, y, X_test, y_test, gamma, u0, v0, iters_loss, num_iter, rho, loss_derivative_only=False):
    train_losses, test_losses = [], []
    u, v = u0, v0
    for i in range(num_iter):
        y_hat = X @ (u * v)
        r = y_hat - y
        grad_u_sam, grad_v_sam = np.zeros_like(u), np.zeros_like(v)
        for k in range(X.shape[0]):
            u_sam_k = u + 2 * rho * r[k] * X[k] * v
            v_sam_k = v + 2 * rho * r[k] * X[k] * u
            if not loss_derivative_only:
                grad_u_sam += ((X[k] * u_sam_k * v_sam_k).sum() - y[k]) * v_sam_k * X[k] / X.shape[0]
                grad_v_sam += ((X[k] * u_sam_k * v_sam_k).sum() - y[k]) * u_sam_k * X[k] / X.shape[0]
            else:
                grad_u_sam += ((X[k] * u_sam_k * v_sam_k).sum() - y[k]) * v * X[k] / X.shape[0]
                grad_v_sam += ((X[k] * u_sam_k * v_sam_k).sum() - y[k]) * u * X[k] / X.shape[0]

        u = u - gamma * grad_u_sam   # gradient step
        v = v - gamma * grad_v_sam   # gradient step

        if i in iters_loss:
            train_losses += [loss(X, X @ (u * v), y)]
            test_losses += [loss(X_test, X_test @ (u * v), y_test)]

    return train_losses, test_losses, u, v

