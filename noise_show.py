from PIL import Image
import numpy as np
import cv2

def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:  ,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:  ].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:  ,1:  ].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]  )) * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]  )) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)


mix = np.random.random()
noise = np.random.gumbel(0., .5 , [10,10])
noise = np.random.gumbel(.0,  1. , [10,10]) * mix + noise * (1-mix)
noise = np.random.gumbel(.6, .05 , [10,10]) * mix + noise * (1-mix)
noise = cv2.resize(noise, dsize=(28,28), interpolation=cv2.INTER_CUBIC)
noise = np.clip(noise * 256, 0, 255)
print(noise)
img = Image.fromarray(np.ndarray.astype(noise, dtype="uint8"), "L").save('noise.png')

freqx = np.random.randint(5,9)
freqy = np.random.randint(5,9)
freqn = np.random.randint(2,5)
noise = generate_perlin_noise_2d([freqx * freqn, freqy * freqn], [freqn, freqn])
noise = (noise + 1 ) * 128
print(noise)
img = Image.fromarray(np.ndarray.astype(noise, dtype="uint8"), "L").save('perlinnoise.png')

