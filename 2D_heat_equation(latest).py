import tensorflow as tf
import datetime, os
#hide tf logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'},
#0 (default) shows all, 1 to filter out INFO logs, 2 to additionally filter out WARNING logs, and 3 to additionally filter out ERROR logs
import scipy.optimize
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import time
import pandas as pd
import seaborn as sns
import codecs, json
from functools import partial #plotting set tick
from mpl_toolkits.mplot3d import axes3d
from pyDOE import lhs         #Latin Hypercube Sampling




# generates same random numbers each time
np.random.seed(1234)
tf.random.set_seed(1234)

print("TensorFlow version: {}".format(tf.__version__))

#data prep
data = scipy.io.loadmat('Data/2D-heat-equation-T=1.mat')     # Load data from file
x = data['x']                                   # 100 points between 0 and 1 [100x1]
y = data['y']                                   # 100 points between 0 and 1 [100x1]
t = data['t']                                   # 50 time points between 0 and 1 [50sx1]
usol = data['usol']                            # solution of 100x100x50 grid points

X, Y, T = np.meshgrid(x, y, t)
#test data
X_u_test = np.hstack((X.flatten(order='F')[:,None], Y.flatten(order='F')[:,None],T.flatten(order='F')[:,None]))


# Domain bounds
lb = np.array([0, 0, 0]) #lower bound
ub = np.array([1, 1, 1])  #upper bound

u = usol.flatten('F')[:,None]


#training data
def trainingdata(N_u, N_f):
    # Boundary Condition x = 0, 0 <= y <= 1, 0<= t<= 1
    leftedge_x = np.reshape(np.dstack((X[:,0,:][:,:,None], Y[:,0,:][:,:,None],T[:,0,:][:,:,None])),(len(y)*len(t),3)) # 5000x3
    leftedge_u = np.reshape(usol[:,0,:][:,:,None], (len(y)*len(t),1)) # 5000x1
    # Boundary Condition  x = 1 , 0 <= y <= 1 , 0<= t<= 1
    rightedge_x = np.reshape(np.dstack((X[:,-1,:][:,:,None], Y[:,-1,:][:,:,None],T[:,-1,:][:,:,None])),(len(y)*len(t),3)) # 5000x3
    rightedge_u = np.reshape(usol[:,-1,:][:,:,None], (len(y)*len(t),1)) # 5000x1
    # Boundary Condition  0 <= x <= 1, y = 1 , 0<= t<= 1
    topedge_y = np.reshape(np.dstack((X[-1,:,:][:,:,None], Y[-1,:,:][:,:,None],T[-1,:,:][:,:,None])),(len(x)*len(t),3)) # 5000x3
    topedge_u = np.reshape(usol[-1,:,:][:,:,None], (len(x)*len(t),1)) # 5000x1
    # Boundary Condition  0 <= x <= 1, y = 0 , 0<= t<= 1
    bottomedge_y = np.reshape(np.dstack((X[0,:,:][:,:,None], Y[0,:,:][:,:,None],T[0,:,:][:,:,None])),(len(x)*len(t),3)) # 5000x3
    bottomedge_u = np.reshape(usol[0,:,:][:,:,None], (len(x)*len(t),1)) # 5000x1
    # initial Condition  0 <= x <= 1, 0 <= y = 1 , t=0
    initialedge_xy = np.reshape(np.dstack((X[:, :, 0][:, :, None], Y[:, :, 0][:, :, None], T[:, :, 0][:, :, None])), # 10000x3
                              (len(x) * len(y), 3))
    initialedge_u = np.reshape(usol[:, :, 0][:, :, None], (len(x) * len(y), 1)) # 10000x1

    all_XY_u_train = np.vstack([leftedge_x, rightedge_x, topedge_y, bottomedge_y, initialedge_xy]) # 30000x3
    all_u_train = np.vstack([leftedge_u, rightedge_u, topedge_u, bottomedge_u,initialedge_u]) # 30000x1

    # choose random N_u points for training
    idxy = np.random.choice(all_XY_u_train.shape[0], N_u, replace=False)


    X_u_train = all_XY_u_train[idxy[0:N_u], :]  # choose indices from  set 'idx' (x,t)
    u_train = all_u_train[idxy[0:N_u], :]  # choose corresponding u

    '''Collocation Points'''

    # Latin Hypercube sampling for collocation points
    # N_f sets of tuples(x,t)
    X_f = lb + (ub - lb) * lhs(3, N_f) #x, y, t 3D
    X_f_train = np.vstack((X_f, X_u_train))  # append training points to collocation points

    return X_f_train, X_u_train, u_train

#pinn
class Sequentialmodel(tf.Module):
    def __init__(self, layers, name=None):
        self.W = []  # Weights and biases
        self.parameters = 0  # total number of parameters
        self.list = []
        self.epochtimes = 0
        self.lr = 0.001 #ADAM learning rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr)

        for i in range(len(layers) - 1):
            input_dim = layers[i]
            output_dim = layers[i + 1]

            # Xavier standard deviation
            std_dv = np.sqrt((2.0 / (input_dim + output_dim)))

            # weights = normal distribution * Xavier standard deviation + 0
            w = tf.random.normal([input_dim, output_dim], dtype='float64') * std_dv

            #             w = tf.cast(tf.ones([input_dim, output_dim]), dtype = 'float64')

            w = tf.Variable(w, trainable=True, name='w' + str(i + 1))

            b = tf.Variable(tf.cast(tf.zeros([output_dim]), dtype='float64'), trainable=True, name='b' + str(i + 1))

            self.W.append(w)
            self.W.append(b)

            self.parameters += input_dim * output_dim + output_dim

        self.X = np.zeros(self.parameters)  # store iterates
        self.G = np.zeros(self.parameters)  # store gradients
        self.store = np.zeros((max_iter, 3))  # store computed values for plotting
        self.iter_counter = 0  # iteration counter for optimizer


    def evaluate(self, x):

        # preprocessing input
        x = (x - lb) / (ub - lb)  # feature scaling

        a = x

        for i in range(len(layers) - 2):
            z = tf.add(tf.matmul(a, self.W[2 * i]), self.W[2 * i + 1])
            a = tf.nn.tanh(z)

        a = tf.add(tf.matmul(a, self.W[-2]), self.W[-1])  # For regression, no activation to last layer

        return a

    def get_weights(self):

        parameters_1d = []  # [.... W_i,b_i.....  ] 1d array

        for i in range(len(layers) - 1):
            w_1d = tf.reshape(self.W[2 * i], [-1])  # flatten weights
            b_1d = tf.reshape(self.W[2 * i + 1], [-1])  # flatten biases

            parameters_1d = tf.concat([parameters_1d, w_1d], 0)  # concat weights
            parameters_1d = tf.concat([parameters_1d, b_1d], 0)  # concat biases

        return parameters_1d

    def set_weights(self, parameters):

        for i in range(len(layers) - 1):
            shape_w = tf.shape(self.W[2 * i]).numpy()  # shape of the weight tensor
            size_w = tf.size(self.W[2 * i]).numpy()  # size of the weight tensor

            shape_b = tf.shape(self.W[2 * i + 1]).numpy()  # shape of the bias tensor
            size_b = tf.size(self.W[2 * i + 1]).numpy()  # size of the bias tensor

            pick_w = parameters[0:size_w]  # pick the weights
            self.W[2 * i].assign(tf.reshape(pick_w, shape_w))  # assign
            parameters = np.delete(parameters, np.arange(size_w), 0)  # delete

            pick_b = parameters[0:size_b]  # pick the biases
            self.W[2 * i + 1].assign(tf.reshape(pick_b, shape_b))  # assign
            parameters = np.delete(parameters, np.arange(size_b), 0)  # delete

    def loss_BC(self, x, y):

        loss_u = tf.reduce_mean(tf.square(y - self.evaluate(x)))
        return loss_u


    def loss_PDE(self, x_to_train_f):

        g = tf.Variable(x_to_train_f, dtype='float64', trainable=False)

        cof =  (8*np.power(np.pi,2)-1)

        x_f = g[:, 0:1]
        y_f = g[:, 1:2]
        t_f = g[:, 2:3]

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_f)
            tape.watch(y_f)
            tape.watch(t_f)

            g = tf.stack([x_f[:, 0], y_f[:, 0], t_f[:, 0]], axis=1)

            u = self.evaluate(g)
            u_x = tape.gradient(u, x_f)
            u_y = tape.gradient(u, y_f)
            u_t = tape.gradient(u, t_f)

        u_xx = tape.gradient(u_x, x_f)
        u_yy = tape.gradient(u_y, y_f)
        del tape

        f = u_t - u_xx - u_yy - cof * np.exp(-1*t_f) * np.sin(2*np.pi * x_f) * np.sin(2*np.pi * y_f) # residual

        loss_f = tf.reduce_mean(tf.square(f))

        return loss_f

    def loss(self, x, y, g):

        loss_u = self.loss_BC(x, y)
        loss_f = self.loss_PDE(g)

        loss = loss_u + loss_f

        return loss, loss_u, loss_f

    def optimizerADAM(self, parameters):#ADAM

        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)

            loss_val, loss_u, loss_f = self.loss(X_u_train, u_train, X_f_train)

        gradients = tape.gradient(loss_val, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        u_pred = self.evaluate(X_u_test)
        error_vec = np.linalg.norm((u - u_pred), 2) / np.linalg.norm(u, 2)
        list = self.list.append(loss_val)
        tf.print(loss_val, loss_u, loss_f, error_vec)

    def optimizerfunc(self, parameters):

        self.set_weights(parameters)

        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)

            loss_val, loss_u, loss_f = self.loss(X_u_train, u_train, X_f_train)

        grads = tape.gradient(loss_val, self.trainable_variables)

        del tape

        grads_1d = []  # store 1d grads

        for i in range(len(layers) - 1):
            grads_w_1d = tf.reshape(grads[2 * i], [-1])  # flatten weights
            grads_b_1d = tf.reshape(grads[2 * i + 1], [-1])  # flatten biases

            grads_1d = tf.concat([grads_1d, grads_w_1d], 0)  # concat grad_weights
            grads_1d = tf.concat([grads_1d, grads_b_1d], 0)  # concat grad_biases

        return loss_val.numpy(), grads_1d.numpy()

    def optimizer_callback(self, parameters):

        loss_value, loss_u, loss_f = self.loss(X_u_train, u_train, X_f_train)

        u_pred = self.evaluate(X_u_test)
        error_vec = np.linalg.norm((u - u_pred), 2) / np.linalg.norm(u, 2)
        list = self.list.append(loss_value)
        tf.print(loss_value, loss_u, loss_f, error_vec)


#loss function
N_u = 50 #Total number of data points for 'u'
N_f = 9000 #Total number of collocation points

# Training data
X_f_train, X_u_train, u_train = trainingdata(N_u,N_f)
layers = np.array([3, 50, 50, 50, 50, 50, 50, 1]) #第1個3表示3D, 6 hidden layers

maxcor = 200
max_iter = 1800


start_time = time.time()
PINN = Sequentialmodel(layers)
init_params = PINN.get_weights()  # 可以再加numpy()

tf.print('ADAM start!')
for i in range(min(max_iter, 200)):
    PINN.optimizerADAM(init_params)

second_params = PINN.get_weights() #要用回ADAM訓練後的variable
# train the model with Scipy L-BFGS optimizer
tf.print('L-BFGS-B start!')
results = scipy.optimize.minimize(fun = PINN.optimizerfunc,
                                  x0 = second_params,
                                  args=(),
                                  method='L-BFGS-B',
                                  jac= True,        # If jac is True, fun is assumed to return the gradient along with the objective function
                                  callback = PINN.optimizer_callback,
                                  options = {'disp': None,
                                            'maxcor': maxcor,
                                            'ftol': 1 * np.finfo(float).eps,  #The iteration stops when (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol
                                            'gtol': 5e-10,
                                            'maxfun':  50000,
                                            'maxiter': max_iter-200,
                                            'iprint': -1,   #print update every 50 iterations
                                            'maxls': 50})

elapsed = time.time() - start_time
print('Training time: %.2f' % (elapsed))

print(results)


PINN.set_weights(results.x)

''' Model Accuracy '''
u_pred = PINN.evaluate(X_u_test)

error_vec = np.linalg.norm((u-u_pred),2)/np.linalg.norm(u,2)        # Relative L2 Norm of the error (Vector)
print('Test Error: %.5f'  % (error_vec))

#Ground truth
u_pred = np.reshape(u_pred, (len(x), len(y), len(t)), order='F')  # Fortran Style ,stacked column wise!

x_1 = x[:,0]
y_1 = y[:,0]
t_1 = [0, 0.5, 1]
t_pos1 = [0, 24, -1]

fig_1 = plt.figure(1, figsize=(18, 15))

for i in range(len(t_1)):
    plt.subplot(3, 3, 1+3*i)
    plt.pcolor(x_1, y_1, usol[:,:, t_pos1[i]], cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x$', fontsize=18)
    plt.ylabel(r'$y$', fontsize=18)
    plt.title('Ground Truth $u(x,y,t)$, t='+str(t_1[i]), fontsize=15)

# Prediction
    plt.subplot(3, 3, 2+3*i)
    plt.pcolor(x_1, y_1, u_pred[:,:, t_pos1[i]], cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x$', fontsize=18)
    plt.ylabel(r'$y$', fontsize=18)
    plt.title('Predicted $\hat u(x,y,t)$, t='+str(t_1[i]), fontsize=15)

# Error
    plt.subplot(3, 3, 3+3*i)
    plt.pcolor(x_1, y_1, np.abs(usol[:,:, t_pos1[i]] - u_pred[:, :, t_pos1[i]]), cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x$', fontsize=18)
    plt.ylabel(r'$y$', fontsize=18)
    plt.title(r'Absolute error $|u(x,y,t)- \hat u(x,y,t)|$, t='+str(t_1[i]), fontsize=15)
    plt.tight_layout()

fig_1.savefig('2D_heat_equation_non_stiff.png', dpi = 500, bbox_inches='tight')

#Plot of collocation points
X_f_train, X_u_train, u_train = trainingdata(N_u,N_f)
fig_2 = plt.figure()
ax_2 = fig_2.add_subplot(projection='3d')

ax_2.scatter3D(X_u_train[:,0],X_u_train[:,1],X_u_train[:,2], s = 10, cmap='Blue', marker = '*' )
ax_2.scatter3D(X_f_train[:,0],X_f_train[:,1],X_f_train[:,2], s = 0.5, cmap='Red', marker = 'o')

ax_2.set_xlabel('X')
ax_2.set_ylabel('Y')
ax_2.set_zlabel('Z')
plt.title('Collocation points')
plt.show()
fig_2.savefig('collocation_points_2D_Heat_equation.png', dpi = 500)

#Loss Plot
itertations = np.arange(0,max_iter,1)
fig_3 ,ax_3 = plt.subplots()
ax_3.plot(itertations, PINN.list, 'r-')
ax_3.set_xlabel('iterations')
ax_3.set_ylabel('loss')
plt.title('Loss with respect to number of iterations')
plt.yscale('symlog')
plt.show()

fig_3.savefig('Loss_2D_heat_equation.png', dpi=500, bbox_inches='tight')

#Error_vac Plot for every t
fig_4 , ax_4 = plt.subplots()
error_vec_list = []
for i in range(len(t)):
    error_vec_list.append((np.linalg.norm((usol[:,:,i]-u_pred[:,:,i]),2)/np.linalg.norm(usol[:,:,i],2)))

ax_4.plot(t, error_vec_list, 'b-')
ax_4.set_xlabel('t')
ax_4.set_ylabel('Error_vac')
plt.title('Error_vac for every t ')
plt.show()

fig_4.savefig('Error_vac_2D_heat_equation.png', dpi=500, bbox_inches='tight')

