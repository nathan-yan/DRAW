import theano
import theano.tensor as T

import numpy as np

import pygame
from pygame.locals import *

pygame.init()

class LSTM:
    def __init__(self, inp_size, hidden_size):
        self.inp_size = inp_size
        self.hidden_size = hidden_size

        self.weights = {
                            "f:x" : init_weights([self.inp_size, self.hidden_size]),
                            "f:h" : init_weights([self.hidden_size, self.hidden_size]),
                            "f:b" : init_weights([self.hidden_size]),
                            "i:x" : init_weights([self.inp_size, self.hidden_size]),
                            "i:h" : init_weights([self.hidden_size, self.hidden_size]),
                            "i:b" : init_weights([self.hidden_size]),
                            "o:x" : init_weights([self.inp_size, self.hidden_size]),
                            "o:h" : init_weights([self.hidden_size, self.hidden_size]),
                            "o:b" : init_weights([self.hidden_size]),
                            "c:x" : init_weights([self.inp_size, self.hidden_size]),
                            "c:h" : init_weights([self.hidden_size, self.hidden_size]),
                            "c:b" : init_weights([self.hidden_size]),
                       }

    def get_weights(self):
        return [self.weights[key] for key in self.weights.keys()]

    def recurrence(self, inp, prev_hidden, prev_cell):
        forget = T.nnet.sigmoid(T.dot(inp, self.weights["f:x"]) +\
                                T.dot(prev_hidden, self.weights["f:h"]) +\
                                self.weights["f:b"])

        input_ = T.nnet.sigmoid(T.dot(inp, self.weights["i:x"]) +\
                                T.dot(prev_hidden, self.weights["i:h"]) +\
                                self.weights["i:b"])

        output = T.nnet.sigmoid(T.dot(inp, self.weights["o:x"]) +\
                                T.dot(prev_hidden, self.weights["o:h"]) +\
                                self.weights["o:b"])

        cell = T.mul(forget, prev_cell) + T.mul(input_, T.tanh(T.dot(inp, self.weights["c:x"]) +\
        T.dot(prev_hidden, self.weights["c:h"]) +\
        self.weights["c:b"]))

        hidden = T.mul(output, cell)

        return hidden, cell

def init_weights(shape):
    return theano.shared(np.array(np.random.randn(*shape) * 0.01).astype(np.float32))

def RMSprop(cost, params, lr=3e-4, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []

    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2

        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling

        updates.append((acc, acc_new))
        updates.append((p, p - T.clip(lr * g, -0.01, 0.01)))

    return updates

inp = T.matrix()   # batchsize x imgsize ** 2
randn = T.tensor3() # timestep x batchsize x latent_vector_size

enc = LSTM(784 + 784 + 784, 256)
dec = LSTM(256, 784)

enc_to_mean = init_weights([256, 256])
enc_to_variance = init_weights([256, 256])

dec_to_write = init_weights([784, 784])

def encoder(canvas, decoder_hidden_1, encoder_hidden_1, encoder_cell_1):
    error = inp - T.nnet.sigmoid(canvas)

    read_vec = T.concatenate([inp, error, decoder_hidden_1], axis = 1)

    enc_hidden, enc_cell = enc.recurrence(read_vec, encoder_hidden_1, encoder_cell_1)

    return enc_hidden, enc_cell

def decoder(gaussian_sample, decoder_hidden_1, decoder_cell_1):
    dec_hidden, dec_cell = dec.recurrence(gaussian_sample, decoder_hidden_1, decoder_cell_1)

    return dec_hidden, dec_cell

batchsize = 32

def draw(randn, latent_loss, canvas, encoder_hidden_1, encoder_cell_1, decoder_hidden_1, decoder_cell_1, inp):
    enc_hidden, enc_cell = encoder(canvas, decoder_hidden_1, encoder_hidden_1, encoder_cell_1)

    mean = T.dot(enc_hidden, enc_to_mean)
    variance = T.exp(T.dot(enc_hidden, enc_to_variance))

    latent_vector = randn * variance + mean

    dec_hidden, dec_cell = decoder(latent_vector, decoder_hidden_1, decoder_cell_1)
    write = T.dot(dec_hidden, dec_to_write)

    canvas += write

    # Calculate latent_loss
    # Means will be of size batchsize x latent_vector_size
    latent_loss += .5 * T.mean(T.sum(mean ** 2, axis = 1) + T.sum(variance ** 2, axis = 1) - T.log(T.sum(variance ** 2, axis = 1)))

    return latent_loss, canvas, enc_hidden, enc_cell, dec_hidden, dec_cell

[[latent_loss, canvas, _, _, _, _], updates] = theano.scan(fn = draw,
                                    sequences = randn,
                                    outputs_info = [np.float32(0.),
                                                    np.zeros([batchsize, 784]).astype(np.float32), np.zeros([batchsize, 256]).astype(np.float32), np.zeros([batchsize, 256]).astype(np.float32), np.zeros([batchsize, 784]).astype(np.float32),
                                                    np.zeros([batchsize, 784]).astype(np.float32)],
                                    non_sequences = inp)

final_canvas = canvas[-1]               # Batchsize x 784
sigmoided_canvas = T.nnet.sigmoid(final_canvas)

construction_loss = -T.mean(T.sum(T.log(sigmoided_canvas) * inp + T.log(1 - sigmoided_canvas) * (1 - inp), axis = 1))

latent_loss = latent_loss[-1] - 16        # single scalar

final_loss = construction_loss + latent_loss

# Optimize for the final_loss
weights = enc.get_weights() + dec.get_weights() + [dec_to_write]

updates = RMSprop(cost = final_loss, params = weights)
train = theano.function(inputs = [inp, randn], outputs = [T.nnet.sigmoid(canvas), final_loss], updates = updates)

def main():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    screen = pygame.display.set_mode((28 * 8 * 2, 28 * 8 * 2))

    images = (mnist.train.images > 0).astype(np.float32)
    epochs = 100
    import time

    for epoch in range (epochs):
        for batch in range (0, 54976, 32):
            randn = np.random.randn(32, 32, 256).astype(np.float32)
            canvases, loss = train(images[batch : batch + 32], randn)

            if batch % 320 == 0:
                #print(canvases[0])
                for t in range (32):
                    for y in range (2):
                        for x in range (4):
                            coordinate = y * 8 + x
                            canvas = canvases[t][coordinate].reshape([28, 28])

                            pygame.event.get()

                            for y_ in range (28):
                                for x_ in range (28):
                                    color = int(canvas[y_][x_] * 255)
                                    pygame.draw.rect(screen, (color, color, color), (x * 28 + x_ * 1, y * 28 + y_ * 1, 1, 1))

                    pygame.display.flip()

                        #canvas = canvases[coordinate][0].reshape([28, 28])

                        #for y_ in range (28):
                        #    for x_ in range (28):
                        #        color = int(canvas[y_][x_] * 255)
                        #        pygame.draw.rect(screen, (color, color, color), (x * 28 + x_ * 1, y * 28 + y_ * 1 + 150, 1, 1))

            print(loss)

main()
