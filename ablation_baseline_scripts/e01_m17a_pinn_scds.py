## [Steady-State Convection Diffusion Problem] [PINN] Meta-Modelling
## Version 1 : 2023/8/11

## Import libraries
import jax
import flax
import optax
from jax import lax, random, numpy as jnp
from jax import random, grad, vmap, hessian, jacfwd, jit
from jax.config import config
from flax import linen as nn
from evojax.util import get_params_format_fn

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# choose GPU
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
jax.config.update("jax_enable_x64", True)



## Main-function to be called
## Input - model_type : 1 (1a) / 2 (1b) / 3 (2a) / 4 (2b)
## Input - learning_rate
## Input - seed : random seed number
def run(model_type=1, learning_rate=5e-3, seed=0, location='results_meta_pdes'):

    #### 0. Preparation

    # folder & header
    folder = '%s'%(location)

    # create working directory for results
    resDir = os.path.join(os.getcwd() , folder)
    if not os.path.isdir(resDir):
        os.mkdir(resDir)


    #### 1. Case setup

    # Function to generate analytical solution
    @jit
    def eval_u(x, v):
        Pe = v*L/k
        u = (1. - jnp.exp(Pe*x/L)) / (1. - jnp.exp(Pe))
        return u

    # fixed PDE & BC parameters k & L
    k = 1.
    L = 1. 

    # task / problem
    task_all = jnp.linspace(5, 100, 20)
    n_task = len(task_all)

    # DNN / PINN   
    class PINNs_1(nn.Module):
        """PINNs"""
        def setup(self):
            self.layers = [nn.Dense(n_nodes, kernel_init = jax.nn.initializers.glorot_uniform()),
                           nn.tanh,
                           nn.Dense(n_nodes, kernel_init = jax.nn.initializers.glorot_uniform()),
                           nn.tanh]
            self.last_layer = nn.Dense(1, kernel_init = jax.nn.initializers.glorot_uniform(), use_bias=False)

        @nn.compact
        def __call__(self, inputs):
            # split the two variables, probably just by slicing
            x, v = inputs[:,0:1], inputs[:,1:2]
            def get_u(x, v):
                f = jnp.hstack([x, v /100.])
                for i, lyr in enumerate(self.layers):
                    f = lyr(f)
                u = self.last_layer(f)
                return u

            u = get_u(x, v)

            # obtain u_x
            def get_u_x(get_u, x, v):
                u_x = jacfwd(get_u)(x, v)
                return u_x

            u_x_vmap = vmap(get_u_x, in_axes=(None, 0, 0))
            u_x = u_x_vmap(get_u, x, v)[:,:,0]   

            #obtain u_xx    
            def get_u_xx(get_u, x, v):
                u_xx = jacfwd(jacfwd(get_u))(x, v)
                return u_xx

            u_xx_vmap = vmap(get_u_xx, in_axes=(None, 0, 0))
            u_xx = u_xx_vmap(get_u, x, v)[:,:,0,0]  

            outputs = jnp.hstack([u, u_x, u_xx])   
            return outputs

        
    class PINNs_2(nn.Module):
        """PINNs"""
        def setup(self):
            self.layers = [nn.Dense(n_nodes, kernel_init = jax.nn.initializers.he_uniform()),
                           jnp.sin,
                           nn.Dense(n_nodes, kernel_init = jax.nn.initializers.he_uniform()),
                           jnp.sin]
            self.last_layer = nn.Dense(1, kernel_init = jax.nn.initializers.he_uniform(), use_bias=False)

        @nn.compact
        def __call__(self, inputs):
            # split the two variables, probably just by slicing
            x, v = inputs[:,0:1], inputs[:,1:2]
            def get_u(x, v):
                f = jnp.hstack([x, v /100.])
                for i, lyr in enumerate(self.layers):
                    f = lyr(f)
                u = self.last_layer(f)
                return u

            u = get_u(x, v)

            # obtain u_x
            def get_u_x(get_u, x, v):
                u_x = jacfwd(get_u)(x, v)
                return u_x

            u_x_vmap = vmap(get_u_x, in_axes=(None, 0, 0))
            u_x = u_x_vmap(get_u, x, v)[:,:,0]   

            #obtain u_xx    
            def get_u_xx(get_u, x, v):
                u_xx = jacfwd(jacfwd(get_u))(x, v)
                return u_xx

            u_xx_vmap = vmap(get_u_xx, in_axes=(None, 0, 0))
            u_xx = u_xx_vmap(get_u, x, v)[:,:,0,0]  

            outputs = jnp.hstack([u, u_x, u_xx])   
            return outputs
        
    
    class PINNs_3(nn.Module):
        """PINNs"""
        def setup(self):
            self.layers = [nn.Dense(n_nodes, kernel_init = jax.nn.initializers.glorot_uniform()),
                           nn.tanh]
            self.last_layer = nn.Dense(1, kernel_init = jax.nn.initializers.glorot_uniform(), use_bias=False)

        @nn.compact
        def __call__(self, inputs):
            # split the two variables, probably just by slicing
            x, v = inputs[:,0:1], inputs[:,1:2]
            def get_u(x, v):
                f = jnp.hstack([x, v /100.])
                for i, lyr in enumerate(self.layers):
                    f = lyr(f)
                u = self.last_layer(f)
                return u

            u = get_u(x, v)

            # obtain u_x
            def get_u_x(get_u, x, v):
                u_x = jacfwd(get_u)(x, v)
                return u_x

            u_x_vmap = vmap(get_u_x, in_axes=(None, 0, 0))
            u_x = u_x_vmap(get_u, x, v)[:,:,0]   

            #obtain u_xx    
            def get_u_xx(get_u, x, v):
                u_xx = jacfwd(jacfwd(get_u))(x, v)
                return u_xx

            u_xx_vmap = vmap(get_u_xx, in_axes=(None, 0, 0))
            u_xx = u_xx_vmap(get_u, x, v)[:,:,0,0]  

            outputs = jnp.hstack([u, u_x, u_xx])   
            return outputs        
        
    
    class PINNs_4(nn.Module):
        """PINNs"""
        def setup(self):
            self.layers = [nn.Dense(n_nodes, kernel_init = jax.nn.initializers.he_uniform()),
                           jnp.sin]
            self.last_layer = nn.Dense(1, kernel_init = jax.nn.initializers.he_uniform(), use_bias=False)

        @nn.compact
        def __call__(self, inputs):
            # split the two variables, probably just by slicing
            x, v = inputs[:,0:1], inputs[:,1:2]
            def get_u(x, v):
                f = jnp.hstack([x, v /100.])
                for i, lyr in enumerate(self.layers):
                    f = lyr(f)
                u = self.last_layer(f)
                return u

            u = get_u(x, v)

            # obtain u_x
            def get_u_x(get_u, x, v):
                u_x = jacfwd(get_u)(x, v)
                return u_x

            u_x_vmap = vmap(get_u_x, in_axes=(None, 0, 0))
            u_x = u_x_vmap(get_u, x, v)[:,:,0]   

            #obtain u_xx    
            def get_u_xx(get_u, x, v):
                u_xx = jacfwd(jacfwd(get_u))(x, v)
                return u_xx

            u_xx_vmap = vmap(get_u_xx, in_axes=(None, 0, 0))
            u_xx = u_xx_vmap(get_u, x, v)[:,:,0,0]  

            outputs = jnp.hstack([u, u_x, u_xx])   
            return outputs

        
    class PINNs_5(nn.Module):
        """PINNs"""
        def setup(self):
            self.layers = [nn.Dense(n_nodes, kernel_init = jax.nn.initializers.glorot_uniform()),
                           nn.tanh,
                           nn.Dense(n_nodes*2, kernel_init = jax.nn.initializers.glorot_uniform()),
                           nn.tanh]
            self.last_layer = nn.Dense(1, kernel_init = jax.nn.initializers.glorot_uniform(), use_bias=False)

        @nn.compact
        def __call__(self, inputs):
            # split the two variables, probably just by slicing
            x, v = inputs[:,0:1], inputs[:,1:2]
            def get_u(x, v):
                f = jnp.hstack([x, v /100.])
                for i, lyr in enumerate(self.layers):
                    f = lyr(f)
                u = self.last_layer(f)
                return u

            u = get_u(x, v)

            # obtain u_x
            def get_u_x(get_u, x, v):
                u_x = jacfwd(get_u)(x, v)
                return u_x

            u_x_vmap = vmap(get_u_x, in_axes=(None, 0, 0))
            u_x = u_x_vmap(get_u, x, v)[:,:,0]   

            #obtain u_xx    
            def get_u_xx(get_u, x, v):
                u_xx = jacfwd(jacfwd(get_u))(x, v)
                return u_xx

            u_xx_vmap = vmap(get_u_xx, in_axes=(None, 0, 0))
            u_xx = u_xx_vmap(get_u, x, v)[:,:,0,0]  

            outputs = jnp.hstack([u, u_x, u_xx])   
            return outputs

        
    class PINNs_6(nn.Module):
        """PINNs"""
        def setup(self):
            self.layers = [nn.Dense(n_nodes, kernel_init = jax.nn.initializers.he_uniform()),
                           jnp.sin,
                           nn.Dense(n_nodes*2, kernel_init = jax.nn.initializers.he_uniform()),
                           jnp.sin]
            self.last_layer = nn.Dense(1, kernel_init = jax.nn.initializers.he_uniform(), use_bias=False)

        @nn.compact
        def __call__(self, inputs):
            # split the two variables, probably just by slicing
            x, v = inputs[:,0:1], inputs[:,1:2]
            def get_u(x, v):
                f = jnp.hstack([x, v /100.])
                for i, lyr in enumerate(self.layers):
                    f = lyr(f)
                u = self.last_layer(f)
                return u

            u = get_u(x, v)

            # obtain u_x
            def get_u_x(get_u, x, v):
                u_x = jacfwd(get_u)(x, v)
                return u_x

            u_x_vmap = vmap(get_u_x, in_axes=(None, 0, 0))
            u_x = u_x_vmap(get_u, x, v)[:,:,0]   

            #obtain u_xx    
            def get_u_xx(get_u, x, v):
                u_xx = jacfwd(jacfwd(get_u))(x, v)
                return u_xx

            u_xx_vmap = vmap(get_u_xx, in_axes=(None, 0, 0))
            u_xx = u_xx_vmap(get_u, x, v)[:,:,0,0]  

            outputs = jnp.hstack([u, u_x, u_xx])   
            return outputs        
        
        
    # initialize model
    if (model_type == 1):
        n_nodes = 50
        model = PINNs_1() 
    if (model_type == 2):
        n_nodes = 50
        model = PINNs_2() 
    if (model_type == 3):
        n_nodes = 900
        model = PINNs_3() 
    if (model_type == 4):
        n_nodes = 900
        model = PINNs_4() 
    if (model_type == 5):
        n_nodes = 50
        model = PINNs_5() 
    if (model_type == 6):
        n_nodes = 50
        model = PINNs_6() 
        
    key, rng = random.split(random.PRNGKey(seed))

    # dummy input
    a = random.normal(key, [1,2])

    # initialization call
    params = model.init(key, a) 
    num_params, format_params_fn = get_params_format_fn(params)

    # flatten initial params
    params = jax.flatten_util.ravel_pytree(params)[0]       
        
        
    #### 2. Training
        
    # no. sample
    n_data = 1001

    x = jnp.linspace(0, L, n_data).reshape(-1, 1)
    data_x = jnp.hstack([jnp.tile(x, (n_task, 1)), jnp.repeat(task_all, repeats=n_data).reshape(-1, 1)])
    data_y = jnp.reshape(eval_u(data_x[:,0], data_x[:,1]), (-1, 1))

    # minibatching
    BS_task = 10

    @jit
    def minibatch(key):
        batch_task = random.choice(key, n_task, (BS_task,), replace=False)
        batch_task = task_all[batch_task]
        batch_data_x = jnp.hstack([jnp.tile(x, (BS_task, 1)), jnp.repeat(batch_task, repeats=n_data).reshape(-1, 1)])
        batch_data_y = jnp.reshape(eval_u(batch_data_x[:,0], batch_data_x[:,1]), (-1, 1))
        return (batch_data_x, batch_data_y)

    batch_data_x, batch_data_y = minibatch(key)    

    # loss function parameter
    lmbda = 1e4    

    def eval_loss(params, inputs, labels):
        pred = model.apply(format_params_fn(params), inputs)
        u, u_x, u_xx = pred[:,0:1], pred[:,1:2], pred[:,2:3]
        # BC
        _bc = jnp.where((jnp.equal(inputs[:,0], 0) | jnp.equal(inputs[:,0], 1)), 1, 0).reshape(-1, 1)
        bc_mse = jnp.sum(jnp.square((labels - u)*_bc)) / jnp.sum(_bc)
        # PDE (physics laws): v*u_x = k*u_xx 
        v = inputs[:,1:2]
        pde = v*u_x - k*u_xx
        pde_mse = jnp.mean(jnp.square(pde))
        # DATA
        mse = jnp.mean(jnp.square(labels - u))
        loss = lmbda* bc_mse + pde_mse + mse
        return loss

    @jit
    def eval_mse(params, inputs, labels):
        pred = model.apply(format_params_fn(params), inputs)
        u, u_x, u_xx = pred[:,0:1], pred[:,1:2], pred[:,2:3]
        # PDE (physics laws): v*u_x = k*u_xx 
        v = inputs[:,1:2]
        pde = v*u_x - k*u_xx
        pde_mse = jnp.mean(jnp.square(pde))
        # DATA
        mse = jnp.mean(jnp.square(labels - u))
        return pde_mse, mse

    loss_grad = jax.jit(jax.value_and_grad(eval_loss))    
    
    # weights update  
    def update(params, opt_state, key):
        batch_data_x, batch_data_y = minibatch(key)
        loss, grad = loss_grad(params, batch_data_x, batch_data_y)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        pde, mse = eval_mse(params, batch_data_x, batch_data_y)
        return params, opt_state, loss, pde, mse

    update = jit(update)    
    
    # optimizer
    max_iters = 50000
    max_lr = learning_rate
    lr_scheduler = optax.warmup_cosine_decay_schedule(init_value=max_lr, peak_value=max_lr, warmup_steps=int(max_iters*.4),  
                                                      decay_steps=max_iters, end_value=1e-6)
    optimizer = optax.adam(learning_rate=lr_scheduler) # Choose the method
    opt_state = optimizer.init(params)    

    # training iteration
    runtime = 0
    train_iters = 0

    store = []

    while (train_iters <= max_iters) and (runtime < 3600):
        # mini-batch update
        start = time.time()
        key, rng = random.split(rng) # update random generator
        params, opt_state, loss, pde, mse = update(params, opt_state, key)
        end = time.time()
        runtime += (end-start)    
        # append weights
        if (train_iters % 100 == 0):
            #print ('iter. = %d,  time = %.2fs,  loss = %.2e  |  pde = %.2e,  mse = %.2e'%(train_iters, runtime, loss, pde, mse))
            store.append([train_iters, runtime, loss, pde, mse])
        train_iters += 1

    store = jnp.array(store)    
      

    #### 3. Prediction
    
    # New prediction with:
    # a. SGD trained model 
    # b. SGD trained model + Pseudo-inverse

    class FrPINNs_1(nn.Module):
        """PINNs"""
        def setup(self):
            self.layers = [nn.Dense(n_nodes, kernel_init = jax.nn.initializers.glorot_uniform()),
                           nn.tanh,
                           nn.Dense(n_nodes, kernel_init = jax.nn.initializers.glorot_uniform()),
                           nn.tanh]
            self.last_layer = nn.Dense(1, kernel_init = jax.nn.initializers.glorot_uniform(), use_bias=False)

        @nn.compact
        def __call__(self, inputs):
            # split the two variables, probably just by slicing
            x, v = inputs[:,0:1], inputs[:,1:2]
            def get_u(x, v):
                f = jnp.hstack([x, v /100.])
                for i, lyr in enumerate(self.layers):
                    f = lyr(f)
                u = self.last_layer(f)
                return u, f

            u, f = get_u(x, v)

            # obtain f_x
            def get_f_x(get_u, x, v):
                u_x, f_x = jacfwd(get_u)(x, v)
                return f_x

            f_x_vmap = vmap(get_f_x, in_axes=(None, 0, 0))
            f_x = f_x_vmap(get_u, x, v)[:,:,0]       

            #obtain f_xx    
            def get_f_xx(get_u, x, v):
                u_xx, f_xx = hessian(get_u)(x, v)
                return f_xx

            f_xx_vmap = vmap(get_f_xx, in_axes=(None, 0, 0))
            f_xx = f_xx_vmap(get_u, x, v)[:,:,0,0]          

            outputs = jnp.hstack([u, f, f_x, f_xx])
            return outputs

        
    class FrPINNs_2(nn.Module):
        """PINNs"""
        def setup(self):
            self.layers = [nn.Dense(n_nodes, kernel_init = jax.nn.initializers.he_uniform()),
                           jnp.sin,
                           nn.Dense(n_nodes, kernel_init = jax.nn.initializers.he_uniform()),
                           jnp.sin]
            self.last_layer = nn.Dense(1, kernel_init = jax.nn.initializers.he_uniform(), use_bias=False)

        @nn.compact
        def __call__(self, inputs):
            # split the two variables, probably just by slicing
            x, v = inputs[:,0:1], inputs[:,1:2]
            def get_u(x, v):
                f = jnp.hstack([x, v /100.])
                for i, lyr in enumerate(self.layers):
                    f = lyr(f)
                u = self.last_layer(f)
                return u, f

            u, f = get_u(x, v)

            # obtain f_x
            def get_f_x(get_u, x, v):
                u_x, f_x = jacfwd(get_u)(x, v)
                return f_x

            f_x_vmap = vmap(get_f_x, in_axes=(None, 0, 0))
            f_x = f_x_vmap(get_u, x, v)[:,:,0]       

            #obtain f_xx    
            def get_f_xx(get_u, x, v):
                u_xx, f_xx = hessian(get_u)(x, v)
                return f_xx

            f_xx_vmap = vmap(get_f_xx, in_axes=(None, 0, 0))
            f_xx = f_xx_vmap(get_u, x, v)[:,:,0,0]          

            outputs = jnp.hstack([u, f, f_x, f_xx])
            return outputs        
        

    class FrPINNs_3(nn.Module):
        """PINNs"""
        def setup(self):
            self.layers = [nn.Dense(n_nodes, kernel_init = jax.nn.initializers.glorot_uniform()),
                           nn.tanh]
            self.last_layer = nn.Dense(1, kernel_init = jax.nn.initializers.glorot_uniform(), use_bias=False)

        @nn.compact
        def __call__(self, inputs):
            # split the two variables, probably just by slicing
            x, v = inputs[:,0:1], inputs[:,1:2]
            def get_u(x, v):
                f = jnp.hstack([x, v /100.])
                for i, lyr in enumerate(self.layers):
                    f = lyr(f)
                u = self.last_layer(f)
                return u, f

            u, f = get_u(x, v)

            # obtain f_x
            def get_f_x(get_u, x, v):
                u_x, f_x = jacfwd(get_u)(x, v)
                return f_x

            f_x_vmap = vmap(get_f_x, in_axes=(None, 0, 0))
            f_x = f_x_vmap(get_u, x, v)[:,:,0]       

            #obtain f_xx    
            def get_f_xx(get_u, x, v):
                u_xx, f_xx = hessian(get_u)(x, v)
                return f_xx

            f_xx_vmap = vmap(get_f_xx, in_axes=(None, 0, 0))
            f_xx = f_xx_vmap(get_u, x, v)[:,:,0,0]          

            outputs = jnp.hstack([u, f, f_x, f_xx])
            return outputs     
    
    
    class FrPINNs_4(nn.Module):
        """PINNs"""
        def setup(self):
            self.layers = [nn.Dense(n_nodes, kernel_init = jax.nn.initializers.he_uniform()),
                           jnp.sin]
            self.last_layer = nn.Dense(1, kernel_init = jax.nn.initializers.he_uniform(), use_bias=False)

        @nn.compact
        def __call__(self, inputs):
            # split the two variables, probably just by slicing
            x, v = inputs[:,0:1], inputs[:,1:2]
            def get_u(x, v):
                f = jnp.hstack([x, v /100.])
                for i, lyr in enumerate(self.layers):
                    f = lyr(f)
                u = self.last_layer(f)
                return u, f

            u, f = get_u(x, v)

            # obtain f_x
            def get_f_x(get_u, x, v):
                u_x, f_x = jacfwd(get_u)(x, v)
                return f_x

            f_x_vmap = vmap(get_f_x, in_axes=(None, 0, 0))
            f_x = f_x_vmap(get_u, x, v)[:,:,0]       

            #obtain f_xx    
            def get_f_xx(get_u, x, v):
                u_xx, f_xx = hessian(get_u)(x, v)
                return f_xx

            f_xx_vmap = vmap(get_f_xx, in_axes=(None, 0, 0))
            f_xx = f_xx_vmap(get_u, x, v)[:,:,0,0]          

            outputs = jnp.hstack([u, f, f_x, f_xx])
            return outputs        

  
    class FrPINNs_5(nn.Module):
        """PINNs"""
        def setup(self):
            self.layers = [nn.Dense(n_nodes, kernel_init = jax.nn.initializers.glorot_uniform()),
                           nn.tanh,
                           nn.Dense(n_nodes*2, kernel_init = jax.nn.initializers.glorot_uniform()),
                           nn.tanh]
            self.last_layer = nn.Dense(1, kernel_init = jax.nn.initializers.glorot_uniform(), use_bias=False)

        @nn.compact
        def __call__(self, inputs):
            # split the two variables, probably just by slicing
            x, v = inputs[:,0:1], inputs[:,1:2]
            def get_u(x, v):
                f = jnp.hstack([x, v /100.])
                for i, lyr in enumerate(self.layers):
                    f = lyr(f)
                u = self.last_layer(f)
                return u, f

            u, f = get_u(x, v)

            # obtain f_x
            def get_f_x(get_u, x, v):
                u_x, f_x = jacfwd(get_u)(x, v)
                return f_x

            f_x_vmap = vmap(get_f_x, in_axes=(None, 0, 0))
            f_x = f_x_vmap(get_u, x, v)[:,:,0]       

            #obtain f_xx    
            def get_f_xx(get_u, x, v):
                u_xx, f_xx = hessian(get_u)(x, v)
                return f_xx

            f_xx_vmap = vmap(get_f_xx, in_axes=(None, 0, 0))
            f_xx = f_xx_vmap(get_u, x, v)[:,:,0,0]          

            outputs = jnp.hstack([u, f, f_x, f_xx])
            return outputs

        
    class FrPINNs_6(nn.Module):
        """PINNs"""
        def setup(self):
            self.layers = [nn.Dense(n_nodes, kernel_init = jax.nn.initializers.he_uniform()),
                           jnp.sin,
                           nn.Dense(n_nodes*2, kernel_init = jax.nn.initializers.he_uniform()),
                           jnp.sin]
            self.last_layer = nn.Dense(1, kernel_init = jax.nn.initializers.he_uniform(), use_bias=False)

        @nn.compact
        def __call__(self, inputs):
            # split the two variables, probably just by slicing
            x, v = inputs[:,0:1], inputs[:,1:2]
            def get_u(x, v):
                f = jnp.hstack([x, v /100.])
                for i, lyr in enumerate(self.layers):
                    f = lyr(f)
                u = self.last_layer(f)
                return u, f

            u, f = get_u(x, v)

            # obtain f_x
            def get_f_x(get_u, x, v):
                u_x, f_x = jacfwd(get_u)(x, v)
                return f_x

            f_x_vmap = vmap(get_f_x, in_axes=(None, 0, 0))
            f_x = f_x_vmap(get_u, x, v)[:,:,0]       

            #obtain f_xx    
            def get_f_xx(get_u, x, v):
                u_xx, f_xx = hessian(get_u)(x, v)
                return f_xx

            f_xx_vmap = vmap(get_f_xx, in_axes=(None, 0, 0))
            f_xx = f_xx_vmap(get_u, x, v)[:,:,0,0]          

            outputs = jnp.hstack([u, f, f_x, f_xx])
            return outputs        
        
        
    # initialize model
    if (model_type == 1):
        model = FrPINNs_1() 
    if (model_type == 2):
        model = FrPINNs_2()    
    if (model_type == 3):
        model = FrPINNs_3() 
    if (model_type == 4):
        model = FrPINNs_4() 
    if (model_type == 5):
        model = FrPINNs_5() 
    if (model_type == 6):
        model = FrPINNs_6() 
        
    num_params, format_params_fn = get_params_format_fn(model.init(key, a))    

    # compute hidden node outputs / derivaties & construct least square problem
    @jit
    def compute_ls_task(batch_data_x, batch_data_y, v, lamb):

        # populate PDE sub-matrix
        prediction = model.apply(format_params_fn(params), batch_data_x)
        f, f_x, f_xx = jnp.split(prediction[:,1:], 3, axis=1)
        # PDE: v*u_x - k*u_xx = 0
        v = batch_data_x[0][1]
        pde = v*f_x - k*f_xx
        # construct least square problem - populate A
        A = jnp.vstack([pde, f[0], f[-1]])
        # construct least square problem - populate b
        pad0 = jnp.zeros((n_data, 1))
        b = jnp.vstack([pad0, batch_data_y[0], batch_data_y[-1]]) 
        # alternative solve (n_sample >> n_node)
        reg = lamb
        w = jnp.linalg.inv(reg*jnp.eye(A.shape[1]) + (A.T@A))@A.T@b
        ssr = np.sum((b - A @ w)**2)
        # mse (given task)
        u = f @ w
        mse = jnp.mean(jnp.square(batch_data_y - u))
        return u, ssr, mse

    # compute hidden node outputs / derivaties & construct least square problem
    @jit
    def compute_ls_task3(batch_data_x, batch_data_y, v):

        # populate PDE sub-matrix
        prediction = model.apply(format_params_fn(params), batch_data_x)
        f, f_x, f_xx = jnp.split(prediction[:,1:], 3, axis=1)
        # PDE: v*u_x - k*u_xx = 0
        v = batch_data_x[0][1]
        pde = v*f_x - k*f_xx
        # construct least square problem - populate A
        A = jnp.vstack([pde, f[0], f[-1]])
        # construct least square problem - populate b
        pad0 = jnp.zeros((n_data, 1))
        b = jnp.vstack([pad0, batch_data_y[0], batch_data_y[-1]]) 
        # alternative solve (n_sample >> n_node)
        w = format_params_fn(params)['params']['last_layer']['kernel']
        ssr = np.sum((b - A @ w)**2)
        # mse (given task)
        u = f @ w
        mse = jnp.mean(jnp.square(batch_data_y - u))
        return u, ssr, mse

    # prediction on test task    
    lambs = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 0.]

    # all task
    res_test = []
    for v in range(1, 111): 
        batch_data_x = jnp.hstack([x, jnp.ones(x.shape)*v])
        batch_data_y = jnp.reshape(eval_u(batch_data_x[:,0], batch_data_x[:,1]), (-1, 1))   
        # SGD PINN
        prediction = model.apply(format_params_fn(params), batch_data_x)
        u = prediction[:,:1]
        mse = jnp.mean(jnp.square(batch_data_y - u)) 
        # compute SSR
        u_s, ssr_s, mse_s = compute_ls_task3(batch_data_x, batch_data_y, v)   
        #print('v = %03d | (SGD PINN) MSE = %.1e | SSR = %.1e  MSE = %.1e'%(round(v), mse, ssr_s, mse_s));
        _res = [round(v), mse]
        _ssr = [ssr_s]
        for lamb in lambs:
            # SGD PINN + Pseudo-inverse
            u_sp, ssr_sp, mse_sp = compute_ls_task(batch_data_x, batch_data_y, v, lamb)
            #print('          lambda = %.e | (SGD PINN + Pseudo-inverse) SSR = %.1e  MSE  = %.1e | (FrPINN) MSE = %.1e'%(lamb, ssr_sp, mse_sp, mse_rp));
            _res = _res + [mse_sp]
            _ssr = _ssr + [ssr_sp]
        res_test.append(_res + _ssr)    
    
    # summary statistics
    columns = ['v', 'MSE_S'] + ['MSE_SP_%.e'%(lamb) for lamb in lambs] + ['SSR_S'] + ['SSR_SP_%.e'%(lamb) for lamb in lambs]
    res_test = pd.DataFrame(jnp.array(res_test), columns=columns)    
       
    # "store" to csv (colnames: Iter, Time, Loss, SSR, MSE)
    his_test = pd.DataFrame(store, columns=['Iter', 'Time', 'Loss', 'PDE', 'MSE'])
    his_test.to_csv(os.path.join(folder, 'scds_pinn_m%d_his_lr%.e_seed-%02d.csv'%(model_type, learning_rate, seed)))
    res_test.to_csv(os.path.join(folder, 'scds_pinn_m%d_res_lr%.e_seed-%02d.csv'%(model_type, learning_rate, seed)))    
    
    # exiting main
    exit_code = 1

    return exit_code




