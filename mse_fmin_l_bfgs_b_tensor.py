from torch.nn import MSELoss
from scipy.optimize import fmin_l_bfgs_b
from numpy import float64

if 0: # Convenience testing data
    import torch
    f = lambda t: t*t + 10
    x0 = torch.Tensor((1,0.1,1,3))
    x0.requires_grad_(True)
    target = torch.Tensor((14, 12, 3, 26))
    mse_loss = MSELoss()
    mse_loss(f(x0), target).backward()


def mse_fmin_l_bfgs_b_tensor(f, x0, target, args=(), **kwargs):
    """Adapts scipy's fmin_l_bfgs_b function to work with torch.Tensor.
Try to find x such that the mean squared error between f(x) and the
given target Tensor is minimized.

Return value:

    Pair of (x, tuple), with x being a torch.Tensor with identical
    dtype as x0, and the tuple being the raw output from
    scipy.optimize.fmin_l_bfgs_b.

f: callable f(x,*args)

    Takes a torch.Tensor x and returns a torch.Tensor, with same input
    and output dtypes. x will have requires_grad=True, and the
    returned tensor must support autodifferentiation of x.

x0: torch.Tensor

    Initial estimate for x.

target: torch.Tensor

    Target tensor for f(x). The dimensions and dtype should be the
    same as the output of f(x).

args: optional sequence

    Optional additional arguments passed to f.

keyword arguments:

    Any additional keyword arguments are passed through to the underlying
    scipy.optimize.fmin_l_bfgs_b function, except for fprime and approx_grad,
    which must not be passed since torch.Tensor.backwards will be used
    to compute the gradient.

    """
    
    # Check that no-one is trying to meddle with us computing the gradient.
    if 'fprime' in kwargs or 'approx_grad' in kwargs:
        raise ValueError("mse_fmin_l_bfgs_b_tensor does not support "
                         "fprime and approx_grad arguments.")

    # Provide defaults for factr, pgtol, maxiter.
    kwargs['factr'] = kwargs.get('factr', 1)
    kwargs['pgtol'] = kwargs.get('pgtol', 1e-16)
    kwargs['maxiter'] = kwargs.get('maxiter', 20000)
    
    # mse_fmin_l_bfgs_b seems to always work in double-precision (float64),
    # so we'll have remember the dtype that f expects and convert back to
    # it when we call it so that Pytorch doesn't kvetch.
    x0_dtype = x0.dtype
    if x0_dtype != target.dtype:
        raise TypeError("x0.dtype %r must match target.dtype %r" %
             (x0.dtype, target.dtype))
    
    mse_loss = MSELoss()   

    def func(x_as_numpy):
        """Convert x_as_numpy from a numpy ndarray to a torch.Tensor,
        and use f and MSELoss to compute the loss and gradient.
        
        Returns a pair of (loss, gradient) as expected by fmin_l_bfgs_b,
        with the gradient converted back to a numpy ndarray.
        """
        x = torch.as_tensor(x_as_numpy, dtype=x0_dtype)
        x.requires_grad_(True)
        
        loss_tensor = mse_loss(f(x, *args), target)
        loss_tensor.backward()
        
        loss_scalar = loss_tensor.item()
        gradient_as_numpy = x.grad.numpy()
        # The gradient must be in float64 form (even though everyone else
        # uses float32); otherwise, all hell breaks loose in the Fortran
        # blfs code.
        if gradient_as_numpy.dtype != float64:
            gradient_as_numpy = gradient_as_numpy.astype(float64)
        
        return (loss_scalar, gradient_as_numpy)
    
    result = fmin_l_bfgs_b(
        func=func,
        x0=x0.detach().numpy().astype(float64),
        **kwargs
    )
    return (torch.as_tensor(result[0], dtype=x0_dtype), result)
