from __future__ import print_function

import theano
import theano.tensor as T 
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import csv
import random
import numpy as np

X = []
y = [] 

GLIMPSES = 32   # How many glimpses (and writes) does DRAW get? 
BATCHSIZE = 100

sampler = T.shared_randomstreams.RandomStreams()

with open('train.csv', 'rb') as csvfile:
    read = csv.reader(csvfile, delimiter = ' ', quotechar = '|')
    counter = 0
    for row in read:
        if counter == 0:
            pass
        
        else:
            X.append(np.asarray(map(int, row[0][2:].split(','))))
            onehot = np.zeros(10)
            onehot[int(row[0][0])] = 1
            y.append(onehot)
        
        counter += 1

X = np.asarray(X) > 0
X = X.astype(float)

def init_weights(shape1, shape2):
    shape = (shape1, shape2)
    return theano.shared(np.asarray(np.random.randn(shape1, shape2) / np.sqrt(shape1), dtype = theano.config.floatX))

inp = T.matrix()
Z = T.tensor3()

lr = T.scalar() 
#encoder_cell = T.matrix() 
#encoder_hidden = T.matrix()
#decoder_cell = T.matrix()
#decoder_hidden = T.matrix() 
#canvas_state = T.matrix()
#latent_loss = T.vector()

A = np.asarray([[[i/28.0 for i in range (28)] for j in range (3)] for k in range (BATCHSIZE)])
B = np.asarray([[[i/28.0 for i in range (28)] for j in range (3)] for k in range (BATCHSIZE)])

MEANS_X = np.asarray([[i - 2 for i in range (1, 4)] for j in range (BATCHSIZE)]) 
MEANS_Y = np.asarray([[j - 2 for i in range (1, 4)] for i in range (BATCHSIZE)]) 

hidden_size = 256

params_1, params_2, params_3, params_4, params_5, params_6, params_7, params_8 = [init_weights(hidden_size + 784 * 2, hidden_size), init_weights(hidden_size + 784 * 2, hidden_size), init_weights(hidden_size + 784 * 2, hidden_size), init_weights(hidden_size + 784 * 2, hidden_size), init_weights(hidden_size, hidden_size), init_weights(hidden_size, hidden_size), init_weights(hidden_size, hidden_size), init_weights(hidden_size, hidden_size)]

params_17, params_18 = [init_weights(hidden_size, 100), init_weights(hidden_size, 100)]

params_9, params_10, params_11, params_12, params_13, params_14, params_15, params_16 = [init_weights(100, hidden_size), init_weights(100, hidden_size), init_weights(100, hidden_size), init_weights(100, hidden_size), init_weights(hidden_size, hidden_size), init_weights(hidden_size, hidden_size), init_weights(hidden_size, hidden_size), init_weights(hidden_size, hidden_size)]

params_19 = init_weights(hidden_size, 784)

enc_lstm_params = [params_1, params_2, params_3, params_4, params_5, params_6, params_7, params_8]
dec_lstm_params = [params_9, params_10, params_11, params_12, params_13, params_14, params_15, params_16]
floatX = theano.config.floatX
m_v_params = [params_17, params_18]
reconstruct_params= params_19

def sample(size = (BATCHSIZE, 20,)):   # Samples from unit gaussian distribution
    return np.random.randn(GLIMPSES, BATCHSIZE, 100)

def DOT(A, B):
    C = A.dimshuffle([0,1,2,'x']) * B.dimshuffle([0,'x',1,2])      
    return C.sum(axis=-2)

def lstm(prev_h, prev_c, inp, params):
    # params[0] = w_f 
    # params[1] = w_i
    # params[2] = w_o 
    # params[3] = w_c
    # params[4] = u_f
    # params[5] = u_i 
    # params[6] = u_o 
    # params[7] = u_c

    # params[0] should have shape inp size x hidden size 
    # params[4] should have shape hidden size x hidden size
    forget_gate = T.nnet.sigmoid(T.dot(inp, params[0]) + T.dot(prev_h, params[4]))

    # params[1] should have shape inp size x hidden size 
    # params[5] should have shape hidden size x hidden size 
    input_gate = T.nnet.sigmoid(T.dot(inp, params[1]) + T.dot(prev_h, params[5]))

    # params[2] should have shape inp size x hidden size 
    # params[6] should have shape hidden size x hidden size
    output_gate = T.nnet.sigmoid(T.dot(inp, params[2]) + T.dot(prev_h, params[6]))

    cell_state = T.mul(forget_gate, prev_c) + T.mul(input_gate, T.tanh(T.dot(inp, params[3]) + T.dot(prev_h, params[7])))

    hidden_state = T.mul(output_gate, cell_state)

    return [cell_state, hidden_state]

def encoder_step(prev_h, prev_c, inp, lstm_params, mean_variances_params):
    cell_state, hidden_state = lstm(prev_h, prev_c, inp, lstm_params)

    means, variances = [T.dot(hidden_state, mean_variances_params[0]), T.dot(hidden_state, mean_variances_params[1])]

    return [cell_state, hidden_state, means, variances]

def decoder_step(prev_h, prev_c, inp, lstm_params, reconstruction_params):
    cell_state, hidden_state = lstm(prev_h, prev_c, inp, lstm_params)

    write = T.dot(hidden_state, reconstruction_params)

    return [cell_state, hidden_state, write]

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def write_head(write_params):
    gx, gy, log_variance, log_stride, log_intensity, write = write_params 
    gx = 29/2.0 * (gx + 1)
    gy = 29/2.0 * (gy + 1)
    variance = T.exp(log_variance)
    stride = T.exp(log_stride)

    tol = 1e-4
    N = 3

    rng = T.arange(N, dtype=floatX)-N/2.+0.5  # e.g.  [1.5, -0.5, 0.5, 1.5]

    muX = gx.dimshuffle([0, 'x']) + stride.dimshuffle([0, 'x'])*rng
    muY = gy.dimshuffle([0, 'x']) + stride.dimshuffle([0, 'x'])*rng

    a = T.arange(self.img_width, dtype=floatX)
    b = T.arange(self.img_height, dtype=floatX)
    
    FX = T.exp( -(a-muX.dimshuffle([0,1,'x']))**2 / 2. / variance.dimshuffle([0,'x','x'])**2 )
    FY = T.exp( -(b-muY.dimshuffle([0,1,'x']))**2 / 2. / variance.dimshuffle([0,'x','x'])**2 )
    FX = FX / (FX.sum(axis=-1).dimshuffle(0, 1, 'x') + tol)
    FY = FY / (FY.sum(axis=-1).dimshuffle(0, 1, 'x') + tol)

    write = T.reshape(write, (BATCHSIZE, 3, 3))
    DOT(DOT(FY.transpose([0, 2, 1]), write), FX)

    return T.reshape(write, (BATCHSIZE, 9))

def one_step(z, prev_canvas, prev_enc_h, prev_enc_c, prev_dec_h, prev_dec_c, prev_latent_loss, params_1, params_2, params_3, params_4, params_5, params_6, params_7, params_8, params_9, params_10, params_11, params_12, params_13, params_14, params_15, params_16, params_17, params_18, params_19, inp):
    params = [params_1, params_2, params_3, params_4, params_5, params_6, params_7, params_8, params_9, params_10, params_11, params_12, params_13, params_14, params_15, params_16, params_17, params_18, params_19]

    encoder_lstm_params = params[:8]
    decoder_lstm_params = params[8:16]
    mean_variances_params = params[16:18]
    reconstruction_params = params[18]

    error_image = inp - T.nnet.sigmoid(prev_canvas)
    read = T.concatenate([inp, error_image], axis = 1)

    encoder_cell_state, encoder_hidden_state, means, variances = encoder_step(prev_enc_h, prev_enc_c, T.concatenate([prev_dec_h, read], axis = 1), encoder_lstm_params, mean_variances_params)

    z = z * variances + means 

    prev_latent_loss += T.sum(means ** 2 + variances ** 2 - T.log(variances ** 2 + 1e-10), axis = 1) 
    decoder_cell_state, decoder_hidden_state, write = decoder_step(prev_dec_h, prev_dec_c, z, decoder_lstm_params, reconstruction_params)

    write_params = [write[0], write[1], write[2], write[3], write[4], write[5:-1]]
    #prev_canvas += T.tanh(write_head(write_params)) 
    prev_canvas += T.tanh(write)

    return [prev_canvas, encoder_hidden_state, encoder_cell_state, decoder_hidden_state, decoder_cell_state, prev_latent_loss]

def one_step_generate(z, prev_canvas, prev_dec_h, prev_dec_c, params_1, params_2, params_3, params_4, params_5, params_6, params_7, params_8, params_9):
    params = [params_1, params_2, params_3, params_4, params_5, params_6, params_7, params_8, params_9]

    decoder_lstm_params = params[:8]
    reconstruction_params = params[8]

    decoder_cell_state, decoder_hidden_state, write = decoder_step(prev_dec_h, prev_dec_c, z, decoder_lstm_params, reconstruct_params)

    prev_canvas += write

    return [prev_canvas, decoder_hidden_state, decoder_cell_state]

def generating(Z):
    ([canvas_states, decoder_hidden_states, decoder_cell_states], updates) = theano.scan(fn = one_step_generate, sequences = Z, outputs_info = [np.zeros(shape = (BATCHSIZE, 784)), np.zeros(shape = (BATCHSIZE, 256)), np.zeros(shape = (BATCHSIZE, 256))], non_sequences = [params_9, params_10, params_11, params_12, params_13, params_14, params_15, params_16, params_19])

    canvas_states = T.nnet.sigmoid(canvas_states)

    return canvas_states 

def training(inp):
    ([canvas_states, encoder_hidden_states, encoder_cell_states, decoder_hidden_states, decoder_cell_states, latent_losses], updates) = theano.scan(fn = one_step, sequences = Z, outputs_info = [np.zeros(shape = (BATCHSIZE, 784)), np.zeros(shape = (BATCHSIZE, 256)), np.zeros(shape = (BATCHSIZE, 256)), np.zeros(shape = (BATCHSIZE, 256)), np.zeros(shape = (BATCHSIZE, 256)), np.zeros(shape = (BATCHSIZE))], non_sequences = [params_1, params_2, params_3, params_4, params_5, params_6, params_7, params_8, params_9, params_10, params_11, params_12, params_13, params_14, params_15, params_16, params_17, params_18, params_19, inp], n_steps = GLIMPSES)

    #[canvas_states, encoder_hidden_states, encoder_cell_states, decoder_hidden_states, decoder_cell_states, latent_losses], updates) = theano.scan(fn = one_step, sequences  = Z, outputs_info = [canvas_state, encoder_hidden, encoder_cell, decoder_hidden, decoder_cell, latent_loss], non_sequences = [enc_lstm_params, dec_lstm_params, m_v_params, reconstruct_params, inp], n_steps = GLIMPSES)

    end_latent_loss = .7 * T.mean(latent_losses[-1] - GLIMPSES/2)
    reconstruction = T.nnet.sigmoid(canvas_states[-1])
    time_reconstruction = T.nnet.sigmoid(canvas_states)

    reconstruction_loss = -T.mean(T.sum(inp * T.log(1e-10 + reconstruction) + (1 - inp) * T.log(1e-10 + 1 - reconstruction), axis = 1))

    #total_loss = end_latent_loss + reconstruction_loss
    #total_loss = reconstruction_loss
    total_loss = end_latent_loss + reconstruction_loss

    updates = RMSprop(cost = total_loss, params = [params_1, params_2, params_3, params_4, params_5, params_6, params_7, params_8, params_9, params_10, params_11, params_12, params_13, params_14, params_15, params_16, params_17, params_18, params_19], lr = lr)
            
    return reconstruction, time_reconstruction, updates, end_latent_loss, reconstruction_loss 

reconstructions, time_reconstruction, updates, end_latent_loss, reconstruction_loss = training(inp)

generated = generating(Z)

train = theano.function(inputs = [inp, Z, lr], outputs = [reconstructions, time_reconstruction, end_latent_loss, reconstruction_loss], updates = updates, allow_input_downcast=True)
generate = theano.function(inputs = [Z], outputs = [generated], allow_input_downcast=True)

import pygame
from pygame.locals import*

pygame.init()

lr_ = 0.001

screen = pygame.display.set_mode((990, 300))
import time
stuff_to_print = []

for epoch in range (100):
    print("Training epoch", (epoch + 1))
    avg_latent_loss = 0
    avg_reconstruction_loss = 0
    epoch_str, batch_str = "", ""
    for point in range (0, len(X), BATCHSIZE):
        #if epoch_str != "" and batch_str != "":
            #print (len(batch_str))
            #pass
        if batch_str != "":
            print('\b' * len(batch_str), end = '')

        if epoch == 0:
            lr_ += (0.001 - 0.0001)/(len(X)/BATCHSIZE)
        
        if epoch > 10:
            if epoch % 4 == 0:
                lr_ /= 2

        #epoch_str = "Training epoch "+ str(epoch + 1) + " - Avg latent loss: "+ str(round(avg_latent_loss/((point + 100)/BATCHSIZE), 3)) + " | Avg reconstruction loss: " + str(round(avg_reconstruction_loss/((point + 100)/BATCHSIZE), 3)) +  " | lr: "+ str(round(lr_, 5))
        #epoch_str += " " * (110 - len(epoch_str))

        z = sample(size = (GLIMPSES, BATCHSIZE, 100))
        #print type(X[point:point+BATCHSIZE])
        reconstructions, time_reconstruction, end_latent_loss, reconstruction_loss = train(X[point:point+BATCHSIZE], z, lr_)

        batch_str = "Batch " + str(((point + 100)/BATCHSIZE )) + '  / 500 ( ' + str((10 - ((point + 100)/BATCHSIZE) % 10)) + " batches until display ) | Latent loss: "+ str(round(end_latent_loss, 3)) + " | Reconstruction loss: " + str(round(reconstruction_loss, 3)) +  " | lr: "+ str(round(lr_, 5))
        #back = "\b" * (len(batch_str) + len(epoch_str))

        print(batch_str, end = "")
        

        avg_latent_loss += end_latent_loss
        avg_reconstruction_loss += reconstruction_loss

        if point % 1000 == 0:
            time_reconstruction = generate(sample(size = (10, BATCHSIZE, 100)))[0]
            for timestep in range (GLIMPSES):
                reconstructions = time_reconstruction[timestep]
                for y_ in range (10):
                    for x_ in range (1):
                        img = reconstructions[y_ * 10 + x_].reshape((28, 28))
                        for y in range (28):
                            for x in range (28):
                                intensity = int(img[y][x] * 255)
                                intensity = (intensity, intensity, intensity)
                                screen.set_at((x_ * 30 + x, y_ * 30 + y), intensity)
                        
                        for y in range (28):
                            for x in range (28):
                                intensity = int(img[y][x] * 255)
                                intensity = (intensity, intensity, intensity)
                                screen.set_at((timestep * 30 + x + 30, y_ * 30 + y), intensity)
                        
                        pygame.event.get() 
                pygame.display.flip()
                #time.sleep(0.02)    