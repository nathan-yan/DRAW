import pygame
from pygame.locals import *

import numpy as np

pygame.init()

screen = pygame.display.set_mode((500, 500))

disp = np.zeros([28, 28])
write_vec = np.array([[.1, .5, .1],
                      [.5, .1, .5],
                      [.1, .5, .1]])

N = 3
stride = 0.1
variance = .5

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            break;

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                variance -= 0.01
            elif event.key == pygame.K_RIGHT:
                variance += 0.01

            elif event.key == pygame.K_UP:
                stride += 0.01
            elif event.key == pygame.K_DOWN:
                stride -= 0.01

    mouse_pos = list(pygame.mouse.get_pos())

    if 0 < mouse_pos[0] < (28 * 3) and 0 < mouse_pos[1] < (28 * 3):
        mean = np.array([mouse_pos[0]/(28 * 3.), mouse_pos[1]/(28. * 3)])

        positions_x = np.array([_ for _ in range (0, N)])
        positions_y = np.array([_ for _ in range (0, N)])

        display_positions_x = np.array([[i/28. for i in range (28)] for j in range (3)])

        mean_position_x = mean[0] + (positions_x - N/2. - .5) * stride
        mean_position_y = mean[1] + (positions_y - N/2. - .5) * stride

        filterbank_x = np.exp(-(display_positions_x - mean_position_x.reshape([3, 1])) ** 2/(2 * variance ** 2))
        filterbank_x = filterbank_x/np.sum(filterbank_x, axis = 1).reshape([3, 1])
        filterbank_y = np.exp(-(display_positions_x - mean_position_y.reshape([3, 1])) ** 2/(2 * variance ** 2))

        filterbank_y = filterbank_y/np.sum(filterbank_y, axis = 1).reshape([3, 1])

        write = np.dot(np.dot(filterbank_y.T, write_vec), filterbank_x)

        for y in range (28):
            for x in range (28):
                color = np.clip(write[y][x] * 255, 0, 255)
                pygame.draw.rect(screen, (color, color, color), (x * 3, y * 3, 3, 3))

        pygame.display.flip()
