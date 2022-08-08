import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize



def dist_calc(time:int, sr:int, speed:int):
    return(speed * (time/ sr))

class solution:
    def __init__(self, positions, distances):
        self.pos = positions
        self.dist = distances
    
    def solution_func(self, coords):
        output = 0
        for position, distance in zip(self.pos, self.dist):
            output += np.sum(
                np.sqrt(((position[0] - coords[0])**2
                +(position[1] - coords[1])**2
                +(position[2] - coords[2])**2)) 
                - distance)**2
        return(output)


def main():
    data_folder = 'data/{}'
    trans = np.loadtxt(data_folder.format('Transmitter.txt'))
    receiv = np.loadtxt(data_folder.format('Receiver.txt'))

    #* Sampling rate
    sr = 100000

    #* Speed of sound
    sound_speed = 1125

    #* Trans positions
    positions = np.array([[0, 0, 10],
                    [20, 0, 10],
                    [20, 20, 10],
                    [0, 20, 10]])
        
    distances = np.array([])

    for values in trans:
        #* Corr calculating
        t = np.argmax(np.correlate(receiv, values))

        #* Dist calculating
        distances = np.append(distances, dist_calc(t, sr, sound_speed))

    start_find_position = np.random.uniform(0, 10, 3)

    lsfunc = solution(positions, distances)
    resulted_coords = optimize.least_squares(lsfunc.solution_func, start_find_position)['x']
    print(f'x: {resulted_coords[0]}, y: {resulted_coords[1]}, z: {resulted_coords[2]}')



if __name__ ==  '__main__':
    main()