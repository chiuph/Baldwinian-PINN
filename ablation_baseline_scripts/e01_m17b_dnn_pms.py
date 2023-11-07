## [Projectile Motion] [DNN] Meta-Modelling
## Version 1 : 2023/8/12

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

    # Initial condition @(x, y) position
    x0, y0 = 0, 0

    # fix parameter
    g, d = 9.8, 1.2

    # load tasks and target label
    task_all = jnp.load(os.path.join('data_meta_pdes', 'pms_task.npy'))
    label_xy = jnp.load(os.path.join('data_meta_pdes', 'pms_label.npy'))
    n_task = len(task_all)


    # DNN / PINN   
    class PINNs_1(nn.Module):
        """PINNs"""
        def setup(self):
            self.layers = [nn.Dense(n_nodes, kernel_init = jax.nn.initializers.glorot_uniform()),
                           nn.tanh]
            # split layers
            self.splitx = nn.Dense(n_nodes, kernel_init = jax.nn.initializers.glorot_uniform())
            self.splity = nn.Dense(n_nodes, kernel_init = jax.nn.initializers.glorot_uniform())
            self.layerx = [nn.tanh,
                           nn.Dense(n_nodes, kernel_init = jax.nn.initializers.glorot_uniform()),
                           nn.tanh]
            self.layery = [nn.tanh,
                           nn.Dense(n_nodes, kernel_init = jax.nn.initializers.glorot_uniform()),
                           nn.tanh]      
            self.last_layerx = nn.Dense(1, kernel_init = jax.nn.initializers.glorot_uniform(), use_bias=False)
            self.last_layery = nn.Dense(1, kernel_init = jax.nn.initializers.glorot_uniform(), use_bias=False) 

        @nn.compact
        def __call__(self, inputs):

            # split the input variables: t, vel0, a0, Cd, CArea, m
            t, vel0, a0, Cd, CArea, m = inputs[:,0:1], inputs[:,1:2], inputs[:,2:3], inputs[:,3:4], inputs[:,4:5], inputs[:,5:6]
            u0 = vel0 * jnp.cos(a0*jnp.pi/180)
            v0 = vel0 * jnp.sin(a0*jnp.pi/180)        
            C = 0.5 * d * Cd * CArea / m

            def get_xy(t, vel0, a0, C):
                u = jnp.hstack([t, vel0 /100., a0 /90., C])
                for i, lyr in enumerate(self.layers):
                    u = lyr(u)
                fx = self.splitx(u)
                for i, lyr in enumerate(self.layerx):
                    fx = lyr(fx)     
                x = self.last_layerx(fx)
                fy = self.splity(u)
                for i, lyr in enumerate(self.layery):
                    fy = lyr(fy) 
                y = self.last_layery(fy)
                return (x, y)

            # PINN output
            x, y = get_xy(t, vel0, a0, C)

            # axillary PDE outputs
            # obtain x_t, y_t
            def get_xy_t(get_xy, t, vel0, a0, C):
                x_t, y_t = jacfwd(get_xy, 0)(t, vel0, a0, C)
                return (x_t, y_t)        

            xy_t_vmap = vmap(get_xy_t, in_axes=(None, 0, 0, 0, 0))

            x_t, y_t = xy_t_vmap(get_xy, t, vel0, a0, C) 
            x_t, y_t = x_t[:,:,0], y_t[:,:,0]   

            # obtain x_tt, y_tt
            def get_xy_tt(get_xy, t, vel0, a0, C):
                x_tt, y_tt = hessian(get_xy, 0)(t, vel0, a0, C)
                return (x_tt, y_tt)

            xy_tt_vmap = vmap(get_xy_tt, in_axes=(None, 0, 0, 0, 0)) 

            x_tt, y_tt = xy_tt_vmap(get_xy, t, vel0, a0, C) 
            x_tt, y_tt = x_tt[:,:,0,0], y_tt[:,:,0,0]        

            outputs = jnp.hstack([x, y, x_t, y_t, x_tt, y_tt, u0, v0, C])   
            return outputs
        
        
    class PINNs_2(nn.Module):
        """PINNs"""
        def setup(self):
            self.layers = [nn.Dense(n_nodes, kernel_init = jax.nn.initializers.he_uniform()),
                           jnp.sin]
            # split layers
            self.splitx = nn.Dense(n_nodes, kernel_init = jax.nn.initializers.he_uniform())
            self.splity = nn.Dense(n_nodes, kernel_init = jax.nn.initializers.he_uniform())
            self.layerx = [jnp.sin,
                           nn.Dense(n_nodes, kernel_init = jax.nn.initializers.he_uniform()),
                           jnp.sin]
            self.layery = [jnp.sin,
                           nn.Dense(n_nodes, kernel_init = jax.nn.initializers.he_uniform()),
                           jnp.sin]      
            self.last_layerx = nn.Dense(1, kernel_init = jax.nn.initializers.he_uniform(), use_bias=False)
            self.last_layery = nn.Dense(1, kernel_init = jax.nn.initializers.he_uniform(), use_bias=False) 

        @nn.compact
        def __call__(self, inputs):

            # split the input variables: t, vel0, a0, Cd, CArea, m
            t, vel0, a0, Cd, CArea, m = inputs[:,0:1], inputs[:,1:2], inputs[:,2:3], inputs[:,3:4], inputs[:,4:5], inputs[:,5:6]
            u0 = vel0 * jnp.cos(a0*jnp.pi/180)
            v0 = vel0 * jnp.sin(a0*jnp.pi/180)        
            C = 0.5 * d * Cd * CArea / m

            def get_xy(t, vel0, a0, C):
                u = jnp.hstack([t, vel0 /100., a0 /90., C])
                for i, lyr in enumerate(self.layers):
                    u = lyr(u)
                fx = self.splitx(u)
                for i, lyr in enumerate(self.layerx):
                    fx = lyr(fx)     
                x = self.last_layerx(fx)
                fy = self.splity(u)
                for i, lyr in enumerate(self.layery):
                    fy = lyr(fy) 
                y = self.last_layery(fy)
                return (x, y)

            # PINN output
            x, y = get_xy(t, vel0, a0, C)

            # axillary PDE outputs
            # obtain x_t, y_t
            def get_xy_t(get_xy, t, vel0, a0, C):
                x_t, y_t = jacfwd(get_xy, 0)(t, vel0, a0, C)
                return (x_t, y_t)        

            xy_t_vmap = vmap(get_xy_t, in_axes=(None, 0, 0, 0, 0))

            x_t, y_t = xy_t_vmap(get_xy, t, vel0, a0, C) 
            x_t, y_t = x_t[:,:,0], y_t[:,:,0]   

            # obtain x_tt, y_tt
            def get_xy_tt(get_xy, t, vel0, a0, C):
                x_tt, y_tt = hessian(get_xy, 0)(t, vel0, a0, C)
                return (x_tt, y_tt)

            xy_tt_vmap = vmap(get_xy_tt, in_axes=(None, 0, 0, 0, 0)) 

            x_tt, y_tt = xy_tt_vmap(get_xy, t, vel0, a0, C) 
            x_tt, y_tt = x_tt[:,:,0,0], y_tt[:,:,0,0]        

            outputs = jnp.hstack([x, y, x_t, y_t, x_tt, y_tt, u0, v0, C])   
            return outputs
        
    
    class PINNs_3(nn.Module):
        """PINNs"""
        def setup(self):
            self.layers = [nn.Dense(n_nodes, kernel_init = jax.nn.initializers.glorot_uniform()),
                           nn.tanh]
            # split layers    
            self.last_layerx = nn.Dense(1, kernel_init = jax.nn.initializers.glorot_uniform(), use_bias=False)
            self.last_layery = nn.Dense(1, kernel_init = jax.nn.initializers.glorot_uniform(), use_bias=False) 

        @nn.compact
        def __call__(self, inputs):

            # split the input variables: t, vel0, a0, Cd, CArea, m
            t, vel0, a0, Cd, CArea, m = inputs[:,0:1], inputs[:,1:2], inputs[:,2:3], inputs[:,3:4], inputs[:,4:5], inputs[:,5:6]
            u0 = vel0 * jnp.cos(a0*jnp.pi/180)
            v0 = vel0 * jnp.sin(a0*jnp.pi/180)        
            C = 0.5 * d * Cd * CArea / m

            def get_xy(t, vel0, a0, C):
                f = jnp.hstack([t, vel0 /100., a0 /90., C])
                for i, lyr in enumerate(self.layers):
                    f = lyr(f)  
                x = self.last_layerx(f)
                y = self.last_layery(f)
                return (x, y)

            # PINN output
            x, y = get_xy(t, vel0, a0, C)

            # axillary PDE outputs
            # obtain x_t, y_t
            def get_xy_t(get_xy, t, vel0, a0, C):
                x_t, y_t = jacfwd(get_xy, 0)(t, vel0, a0, C)
                return (x_t, y_t)        

            xy_t_vmap = vmap(get_xy_t, in_axes=(None, 0, 0, 0, 0))

            x_t, y_t = xy_t_vmap(get_xy, t, vel0, a0, C) 
            x_t, y_t = x_t[:,:,0], y_t[:,:,0]   

            # obtain x_tt, y_tt
            def get_xy_tt(get_xy, t, vel0, a0, C):
                x_tt, y_tt = hessian(get_xy, 0)(t, vel0, a0, C)
                return (x_tt, y_tt)

            xy_tt_vmap = vmap(get_xy_tt, in_axes=(None, 0, 0, 0, 0)) 

            x_tt, y_tt = xy_tt_vmap(get_xy, t, vel0, a0, C) 
            x_tt, y_tt = x_tt[:,:,0,0], y_tt[:,:,0,0]        

            outputs = jnp.hstack([x, y, x_t, y_t, x_tt, y_tt, u0, v0, C])   
            return outputs
        
    
    class PINNs_4(nn.Module):
        """PINNs"""
        def setup(self):
            self.layers = [nn.Dense(n_nodes, kernel_init = jax.nn.initializers.he_uniform()),
                           jnp.sin]
            # split layers    
            self.last_layerx = nn.Dense(1, kernel_init = jax.nn.initializers.he_uniform(), use_bias=False)
            self.last_layery = nn.Dense(1, kernel_init = jax.nn.initializers.he_uniform(), use_bias=False) 

        @nn.compact
        def __call__(self, inputs):

            # split the input variables: t, vel0, a0, Cd, CArea, m
            t, vel0, a0, Cd, CArea, m = inputs[:,0:1], inputs[:,1:2], inputs[:,2:3], inputs[:,3:4], inputs[:,4:5], inputs[:,5:6]
            u0 = vel0 * jnp.cos(a0*jnp.pi/180)
            v0 = vel0 * jnp.sin(a0*jnp.pi/180)        
            C = 0.5 * d * Cd * CArea / m

            def get_xy(t, vel0, a0, C):
                f = jnp.hstack([t, vel0 /100., a0 /90., C])
                for i, lyr in enumerate(self.layers):
                    f = lyr(f)  
                x = self.last_layerx(f)
                y = self.last_layery(f)
                return (x, y)

            # PINN output
            x, y = get_xy(t, vel0, a0, C)

            # axillary PDE outputs
            # obtain x_t, y_t
            def get_xy_t(get_xy, t, vel0, a0, C):
                x_t, y_t = jacfwd(get_xy, 0)(t, vel0, a0, C)
                return (x_t, y_t)        

            xy_t_vmap = vmap(get_xy_t, in_axes=(None, 0, 0, 0, 0))

            x_t, y_t = xy_t_vmap(get_xy, t, vel0, a0, C) 
            x_t, y_t = x_t[:,:,0], y_t[:,:,0]   

            # obtain x_tt, y_tt
            def get_xy_tt(get_xy, t, vel0, a0, C):
                x_tt, y_tt = hessian(get_xy, 0)(t, vel0, a0, C)
                return (x_tt, y_tt)

            xy_tt_vmap = vmap(get_xy_tt, in_axes=(None, 0, 0, 0, 0)) 

            x_tt, y_tt = xy_tt_vmap(get_xy, t, vel0, a0, C) 
            x_tt, y_tt = x_tt[:,:,0,0], y_tt[:,:,0,0]        

            outputs = jnp.hstack([x, y, x_t, y_t, x_tt, y_tt, u0, v0, C])   
            return outputs

        
    # initialize model
    if (model_type == 1):
        n_nodes = 30
        model = PINNs_1() 
    if (model_type == 2):
        n_nodes = 30
        model = PINNs_2() 
    if (model_type == 3):
        n_nodes = 900
        model = PINNs_3() 
    if (model_type == 4):
        n_nodes = 900
        model = PINNs_4() 
        
    key, rng = random.split(random.PRNGKey(seed))

    # dummy input
    a = random.normal(key, [1,6])

    # initialization call
    params = model.init(key, a) 
    num_params, format_params_fn = get_params_format_fn(params)

    # flatten initial params
    params = jax.flatten_util.ravel_pytree(params)[0]       
        
        
    #### 2. Training
        
    # no. sample
    n_data = 101

    def gen_t(a_T):
        t = jnp.linspace(0, a_T, n_data).reshape(-1, 1)
        return t

    gen_t = vmap(gen_t)

    # minibatching
    BS_task = 30

    @jit
    def minibatch(key):
        batch_task = random.choice(key, n_task, (BS_task,), replace=False)
        batch_data_x = jnp.hstack([task_all[batch_task][:,:2], task_all[batch_task][:,4:7]])
        batch_data_x = jnp.repeat(batch_data_x, repeats=n_data, axis=0)
        batch_data_x = jnp.hstack([gen_t(task_all[batch_task][:,-1:]).reshape(-1, 1), batch_data_x])
        label_x, label_y = label_xy[batch_task][:,:,0].reshape(-1, 1), label_xy[batch_task][:,:,1].reshape(-1, 1)
        batch_data_y = jnp.hstack([label_x, label_y])
        return (batch_data_x, batch_data_y)

    batch_data_x, batch_data_y = minibatch(key)   

    # loss function parameter
    lmbda = 10    

    def eval_loss(params, inputs, labels):
        pred = model.apply(format_params_fn(params), inputs)
        x, y, x_t, y_t, x_tt, y_tt, u0, v0, C = jnp.split(pred, 9, axis=1)
        # DATA
        mse = jnp.mean((labels - jnp.hstack([x, y]))**2)
        loss = mse
        return loss

    @jit
    def eval_mse(params, inputs, labels):
        pred = model.apply(format_params_fn(params), inputs)
        x, y, x_t, y_t, x_tt, y_tt, u0, v0, C = jnp.split(pred, 9, axis=1)
        # PDE 1: x_tt + R*x_t = 0
        # PDE 2: y_tt + R*y_t = -g
        V = jnp.sqrt( (x_t)**2 + (y_t)**2 )
        R = C*V
        pde_1 = x_tt + R*x_t
        pde_2 = y_tt + R*y_t + g
        pde_mse_1, pde_mse_2 = jnp.mean(jnp.square(pde_1)), jnp.mean(jnp.square(pde_2))
        pde_mse = pde_mse_1 + pde_mse_2 
        # DATA
        mse = jnp.mean((labels - jnp.hstack([x, y]))**2)
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
    max_iters = 150000
    max_lr = learning_rate
    lr_scheduler = optax.warmup_cosine_decay_schedule(init_value=max_lr, peak_value=max_lr, warmup_steps=int(max_iters*.4),  
                                                      decay_steps=max_iters, end_value=1e-6)
    optimizer = optax.adam(learning_rate=lr_scheduler) # Choose the method
    opt_state = optimizer.init(params)  

    # training iteration
    runtime = 0
    train_iters = 0

    store = []

    while (train_iters <= max_iters) and (runtime < 7200):
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
                           nn.tanh]
            # split layers
            self.splitx = nn.Dense(n_nodes, kernel_init = jax.nn.initializers.glorot_uniform())
            self.splity = nn.Dense(n_nodes, kernel_init = jax.nn.initializers.glorot_uniform())
            self.layerx = [nn.tanh,
                           nn.Dense(n_nodes, kernel_init = jax.nn.initializers.glorot_uniform()),
                           nn.tanh]
            self.layery = [nn.tanh,
                           nn.Dense(n_nodes, kernel_init = jax.nn.initializers.glorot_uniform()),
                           nn.tanh]      
            self.last_layerx = nn.Dense(1, kernel_init = jax.nn.initializers.glorot_uniform(), use_bias=False)
            self.last_layery = nn.Dense(1, kernel_init = jax.nn.initializers.glorot_uniform(), use_bias=False) 

        @nn.compact
        def __call__(self, inputs):

            # split the input variables: t, vel0, a0, Cd, CArea, m
            t, vel0, a0, Cd, CArea, m = inputs[:,0:1], inputs[:,1:2], inputs[:,2:3], inputs[:,3:4], inputs[:,4:5], inputs[:,5:6]
            u0 = vel0 * jnp.cos(a0*jnp.pi/180)
            v0 = vel0 * jnp.sin(a0*jnp.pi/180)        
            C = 0.5 * d * Cd * CArea / m

            def get_fxy(t, vel0, a0, C):
                u = jnp.hstack([t, vel0 /100., a0 /90., C])
                for i, lyr in enumerate(self.layers):
                    u = lyr(u)
                fx = self.splitx(u)
                for i, lyr in enumerate(self.layerx):
                    fx = lyr(fx)     
                x = self.last_layerx(fx)
                fy = self.splity(u)
                for i, lyr in enumerate(self.layery):
                    fy = lyr(fy) 
                y = self.last_layery(fy)
                return (x, y, fx, fy)

            # PINN output
            x, y, fx, fy = get_fxy(t, vel0, a0, C)

            # axillary PDE outputs
            # obtain fx_t, fy_t
            def get_fxy_t(get_fxy, t, vel0, a0, C):
                x_t, y_t, fx_t, fy_t = jacfwd(get_fxy, 0)(t, vel0, a0, C)
                return (fx_t, fy_t)        

            fxy_t_vmap = vmap(get_fxy_t, in_axes=(None, 0, 0, 0, 0))

            fx_t, fy_t = fxy_t_vmap(get_fxy, t, vel0, a0, C) 
            fx_t, fy_t = fx_t[:,:,0], fy_t[:,:,0]   

            # obtain fx_tt, fy_tt
            def get_fxy_tt(get_fxy, t, vel0, a0, C):
                x_tt, y_tt, fx_tt, fy_tt = hessian(get_fxy, 0)(t, vel0, a0, C)
                return (fx_tt, fy_tt)

            fxy_tt_vmap = vmap(get_fxy_tt, in_axes=(None, 0, 0, 0, 0)) 

            fx_tt, fy_tt = fxy_tt_vmap(get_fxy, t, vel0, a0, C) 
            fx_tt, fy_tt = fx_tt[:,:,0,0], fy_tt[:,:,0,0]        

            outputs = jnp.hstack([x, y, u0, v0, C, fx, fy, fx_t, fy_t, fx_tt, fy_tt])   
            return outputs

        
    class FrPINNs_2(nn.Module):
        """PINNs"""
        def setup(self):
            self.layers = [nn.Dense(n_nodes, kernel_init = jax.nn.initializers.he_uniform()),
                           jnp.sin]
            # split layers
            self.splitx = nn.Dense(n_nodes, kernel_init = jax.nn.initializers.he_uniform())
            self.splity = nn.Dense(n_nodes, kernel_init = jax.nn.initializers.he_uniform())
            self.layerx = [jnp.sin,
                           nn.Dense(n_nodes, kernel_init = jax.nn.initializers.he_uniform()),
                           jnp.sin]
            self.layery = [jnp.sin,
                           nn.Dense(n_nodes, kernel_init = jax.nn.initializers.he_uniform()),
                           jnp.sin]      
            self.last_layerx = nn.Dense(1, kernel_init = jax.nn.initializers.he_uniform(), use_bias=False)
            self.last_layery = nn.Dense(1, kernel_init = jax.nn.initializers.he_uniform(), use_bias=False) 

        @nn.compact
        def __call__(self, inputs):

            # split the input variables: t, vel0, a0, Cd, CArea, m
            t, vel0, a0, Cd, CArea, m = inputs[:,0:1], inputs[:,1:2], inputs[:,2:3], inputs[:,3:4], inputs[:,4:5], inputs[:,5:6]
            u0 = vel0 * jnp.cos(a0*jnp.pi/180)
            v0 = vel0 * jnp.sin(a0*jnp.pi/180)        
            C = 0.5 * d * Cd * CArea / m

            def get_fxy(t, vel0, a0, C):
                u = jnp.hstack([t, vel0 /100., a0 /90., C])
                for i, lyr in enumerate(self.layers):
                    u = lyr(u)
                fx = self.splitx(u)
                for i, lyr in enumerate(self.layerx):
                    fx = lyr(fx)     
                x = self.last_layerx(fx)
                fy = self.splity(u)
                for i, lyr in enumerate(self.layery):
                    fy = lyr(fy) 
                y = self.last_layery(fy)
                return (x, y, fx, fy)

            # PINN output
            x, y, fx, fy = get_fxy(t, vel0, a0, C)

            # axillary PDE outputs
            # obtain fx_t, fy_t
            def get_fxy_t(get_fxy, t, vel0, a0, C):
                x_t, y_t, fx_t, fy_t = jacfwd(get_fxy, 0)(t, vel0, a0, C)
                return (fx_t, fy_t)        

            fxy_t_vmap = vmap(get_fxy_t, in_axes=(None, 0, 0, 0, 0))

            fx_t, fy_t = fxy_t_vmap(get_fxy, t, vel0, a0, C) 
            fx_t, fy_t = fx_t[:,:,0], fy_t[:,:,0]   

            # obtain fx_tt, fy_tt
            def get_fxy_tt(get_fxy, t, vel0, a0, C):
                x_tt, y_tt, fx_tt, fy_tt = hessian(get_fxy, 0)(t, vel0, a0, C)
                return (fx_tt, fy_tt)

            fxy_tt_vmap = vmap(get_fxy_tt, in_axes=(None, 0, 0, 0, 0)) 

            fx_tt, fy_tt = fxy_tt_vmap(get_fxy, t, vel0, a0, C) 
            fx_tt, fy_tt = fx_tt[:,:,0,0], fy_tt[:,:,0,0]        

            outputs = jnp.hstack([x, y, u0, v0, C, fx, fy, fx_t, fy_t, fx_tt, fy_tt])   
            return outputs
        

    class FrPINNs_3(nn.Module):
        """PINNs"""
        def setup(self):
            self.layers = [nn.Dense(n_nodes, kernel_init = jax.nn.initializers.glorot_uniform()),
                           nn.tanh]
            # split layers    
            self.last_layerx = nn.Dense(1, kernel_init = jax.nn.initializers.glorot_uniform(), use_bias=False)
            self.last_layery = nn.Dense(1, kernel_init = jax.nn.initializers.glorot_uniform(), use_bias=False) 

        @nn.compact
        def __call__(self, inputs):

            # split the input variables: t, vel0, a0, Cd, CArea, m
            t, vel0, a0, Cd, CArea, m = inputs[:,0:1], inputs[:,1:2], inputs[:,2:3], inputs[:,3:4], inputs[:,4:5], inputs[:,5:6]
            u0 = vel0 * jnp.cos(a0*jnp.pi/180)
            v0 = vel0 * jnp.sin(a0*jnp.pi/180)        
            C = 0.5 * d * Cd * CArea / m

            def get_fxy(t, vel0, a0, C):
                f = jnp.hstack([t, vel0 /100., a0 /90., C])
                for i, lyr in enumerate(self.layers):
                    f = lyr(f)  
                x = self.last_layerx(f)
                y = self.last_layery(f)
                return (x, y, f)

            # PINN output
            x, y, f = get_fxy(t, vel0, a0, C)

            # axillary PDE outputs
            # obtain f_t
            def get_fxy_t(get_fxy, t, vel0, a0, C):
                x_t, y_t, f_t = jacfwd(get_fxy, 0)(t, vel0, a0, C)
                return (f_t)        

            fxy_t_vmap = vmap(get_fxy_t, in_axes=(None, 0, 0, 0, 0))
            f_t = fxy_t_vmap(get_fxy, t, vel0, a0, C)[:,:,0] 

            # obtain f_tt
            def get_fxy_tt(get_fxy, t, vel0, a0, C):
                x_tt, y_tt, f_tt = hessian(get_fxy, 0)(t, vel0, a0, C)
                return (f_tt)

            fxy_tt_vmap = vmap(get_fxy_tt, in_axes=(None, 0, 0, 0, 0)) 
            f_tt = fxy_tt_vmap(get_fxy, t, vel0, a0, C)[:,:,0,0]         

            outputs = jnp.hstack([x, y, u0, v0, C, f, f, f_t, f_t, f_tt, f_tt])   
            return outputs
    
    
    class FrPINNs_4(nn.Module):
        """PINNs"""
        def setup(self):
            self.layers = [nn.Dense(n_nodes, kernel_init = jax.nn.initializers.he_uniform()),
                           jnp.sin]
            # split layers    
            self.last_layerx = nn.Dense(1, kernel_init = jax.nn.initializers.he_uniform(), use_bias=False)
            self.last_layery = nn.Dense(1, kernel_init = jax.nn.initializers.he_uniform(), use_bias=False) 

        @nn.compact
        def __call__(self, inputs):

            # split the input variables: t, vel0, a0, Cd, CArea, m
            t, vel0, a0, Cd, CArea, m = inputs[:,0:1], inputs[:,1:2], inputs[:,2:3], inputs[:,3:4], inputs[:,4:5], inputs[:,5:6]
            u0 = vel0 * jnp.cos(a0*jnp.pi/180)
            v0 = vel0 * jnp.sin(a0*jnp.pi/180)        
            C = 0.5 * d * Cd * CArea / m

            def get_fxy(t, vel0, a0, C):
                f = jnp.hstack([t, vel0 /100., a0 /90., C])
                for i, lyr in enumerate(self.layers):
                    f = lyr(f)  
                x = self.last_layerx(f)
                y = self.last_layery(f)
                return (x, y, f)

            # PINN output
            x, y, f = get_fxy(t, vel0, a0, C)

            # axillary PDE outputs
            # obtain f_t
            def get_fxy_t(get_fxy, t, vel0, a0, C):
                x_t, y_t, f_t = jacfwd(get_fxy, 0)(t, vel0, a0, C)
                return (f_t)        

            fxy_t_vmap = vmap(get_fxy_t, in_axes=(None, 0, 0, 0, 0))
            f_t = fxy_t_vmap(get_fxy, t, vel0, a0, C)[:,:,0] 

            # obtain f_tt
            def get_fxy_tt(get_fxy, t, vel0, a0, C):
                x_tt, y_tt, f_tt = hessian(get_fxy, 0)(t, vel0, a0, C)
                return (f_tt)

            fxy_tt_vmap = vmap(get_fxy_tt, in_axes=(None, 0, 0, 0, 0)) 
            f_tt = fxy_tt_vmap(get_fxy, t, vel0, a0, C)[:,:,0,0]         

            outputs = jnp.hstack([x, y, u0, v0, C, f, f, f_t, f_t, f_tt, f_tt])   
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
        
    num_params, format_params_fn = get_params_format_fn(model.init(key, a))    

    # compute hidden node outputs / derivaties & construct least square problem
    @jit
    def compute_ls_ntask(batch_data_x, batch_data_y, params, lamb):

        # populate PDE sub-matrix (SGD trained / randomized weights)
        prediction = model.apply(format_params_fn(params), batch_data_x)
        fx, fy, fx_t, fy_t, fx_tt, fy_tt = jnp.split(prediction[:, 5:], 6, axis=1)
        u0, v0, C = prediction[0,2], prediction[0,3], prediction[0,4]
        # PDE 1: x_tt + R*x_t = 0
        # PDE 2: y_tt + R*y_t = -g
        # first iteration - R=0
        R = 0
        # construct least square problem - populate Ax & Ay
        Ax = jnp.vstack([fx_tt + R*fx_t, fx[0], fx_t[0]])
        Ay = jnp.vstack([fy_tt + R*fy_t, fy[0], fy_t[0]])
        # construct least square problem - populate bx & by
        pad0 = jnp.zeros((n_data, 1))
        bx = jnp.vstack([pad0, x0, u0]) 
        by = jnp.vstack([pad0 - g, y0, v0]) 
        # alternative solve (n_sample >> n_node)
        reg = lamb
        wx = jnp.linalg.inv(reg*jnp.eye(Ax.shape[1]) + (Ax.T@Ax))@Ax.T@bx
        wy = jnp.linalg.inv(reg*jnp.eye(Ay.shape[1]) + (Ay.T@Ay))@Ay.T@by
        tot_iter = 15
        for i in range(tot_iter - 1):
            # update R
            x_t, y_t = fx_t @ wx, fy_t @ wy
            V = jnp.sqrt( (x_t)**2 + (y_t)**2 )
            R = C*V
            # update A (w_x, w_y)
            Ax = jnp.vstack([fx_tt + R*fx_t, fx[0], fx_t[0]])
            Ay = jnp.vstack([fy_tt + R*fy_t, fy[0], fy_t[0]])
            # alternative solve (n_sample >> n_node & n_node >> n_sample)
            wx = jnp.linalg.inv(reg*jnp.eye(Ax.shape[1]) + (Ax.T@Ax))@Ax.T@bx
            wy = jnp.linalg.inv(reg*jnp.eye(Ay.shape[1]) + (Ay.T@Ay))@Ay.T@by
            #w = A.T@jnp.linalg.inv(reg*jnp.eye(A.shape[0]) + (A@A.T))@b
        ssrx, ssry = np.sum((bx - Ax @ wx)**2), np.sum((by - Ay @ wy)**2)
        # mse (given task)
        x, y = fx @ wx, fy @ wy
        mse = jnp.mean((batch_data_y - jnp.hstack([x, y]))**2)
        return x, y, ssrx, ssry, mse

    @jit
    def compute_ls_ntask3(batch_data_x, batch_data_y, params):

        # populate PDE sub-matrix (SGD trained / randomized weights)
        prediction = model.apply(format_params_fn(params), batch_data_x)
        fx, fy, fx_t, fy_t, fx_tt, fy_tt = jnp.split(prediction[:, 5:], 6, axis=1)
        u0, v0, C = prediction[0,2], prediction[0,3], prediction[0,4]
        # PDE 1: x_tt + R*x_t = 0
        # PDE 2: y_tt + R*y_t = -g
        wx = format_params_fn(params)['params']['last_layerx']['kernel']
        wy = format_params_fn(params)['params']['last_layery']['kernel']
        x_t, y_t = fx_t @ wx, fy_t @ wy
        V = jnp.sqrt( (x_t)**2 + (y_t)**2 )
        R = C*V
        # construct least square problem - populate Ax & Ay
        Ax = jnp.vstack([fx_tt + R*fx_t, fx[0], fx_t[0]])
        Ay = jnp.vstack([fy_tt + R*fy_t, fy[0], fy_t[0]])
        # construct least square problem - populate bx & by
        pad0 = jnp.zeros((n_data, 1))
        bx = jnp.vstack([pad0, x0, u0]) 
        by = jnp.vstack([pad0 - g, y0, v0]) 
        # ssr
        ssrx, ssry = np.sum((bx - Ax @ wx)**2), np.sum((by - Ay @ wy)**2)
        # mse (given task)
        x, y = fx @ wx, fy @ wy
        mse = jnp.mean((batch_data_y - jnp.hstack([x, y]))**2)
        return x, y, ssrx, ssry, mse

    
    # load new tasks and target label
    ntask_all = jnp.load(os.path.join('data_meta_pdes', 'pms_test_task.npy'))
    nlabel_xy = jnp.load(os.path.join('data_meta_pdes', 'pms_test_label.npy'))
    
    # prediction on test task    
    lambs = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 0.]

    # all task
    res_test = []
    for task in range(len(ntask_all)):
        vel0, a0, g, d, Cd, CArea, m, ball, a_T = ntask_all[task]
        batch_task = jnp.array([task])
        batch_data_x = jnp.hstack([ntask_all[batch_task][:,:2], ntask_all[batch_task][:,4:7]])
        batch_data_x = jnp.repeat(batch_data_x, repeats=n_data, axis=0)
        batch_data_x = jnp.hstack([gen_t(ntask_all[batch_task][:,-1:]).reshape(-1, 1), batch_data_x])
        label_x, label_y = nlabel_xy[batch_task][:,:,0].reshape(-1, 1), nlabel_xy[batch_task][:,:,1].reshape(-1, 1)
        batch_data_y = jnp.hstack([label_x, label_y])
        # SGD PINN
        prediction = model.apply(format_params_fn(params), batch_data_x)
        x, y = prediction[:,0:1], prediction[:,1:2]
        mse = jnp.mean((batch_data_y - jnp.hstack([x, y]))**2)
        # compute SSR
        x_s, y_s, ssrx_s, ssry_s, mse_s = compute_ls_ntask3(batch_data_x, batch_data_y, params)   
        ssr_s = (ssrx_s + ssry_s) /2.
        _res = [vel0, a0, ball, mse]
        _ssr = [ssr_s]
        for lamb in lambs:
            # SGD PINN + Pseudo-inverse
            x_sp, y_sp, ssrx_sp, ssry_sp, mse_sp = compute_ls_ntask(batch_data_x, batch_data_y, params, lamb)
            ssr_sp = (ssrx_sp + ssry_sp) /2.
            _res = _res + [mse_sp]
            _ssr = _ssr + [ssr_sp]
        res_test.append(_res + _ssr)    
    
    # summary statistics
    columns = ['vel0', 'a0', 'ball', 'MSE_S'] + ['MSE_SP_%.e'%(lamb) for lamb in lambs] + ['SSR_S'] + ['SSR_SP_%.e'%(lamb) for lamb in lambs]
    res_test = pd.DataFrame(jnp.array(res_test), columns=columns)    
       
    # "store" to csv (colnames: Iter, Time, Loss, SSR, MSE)
    his_test = pd.DataFrame(store, columns=['Iter', 'Time', 'Loss', 'PDE', 'MSE'])
    his_test.to_csv(os.path.join(folder, 'pms_dnn_m%d_his_lr%.e_seed-%02d.csv'%(model_type, learning_rate, seed)))
    res_test.to_csv(os.path.join(folder, 'pms_dnn_m%d_res_lr%.e_seed-%02d.csv'%(model_type, learning_rate, seed)))    
    
    # exiting main
    exit_code = 1

    return exit_code




