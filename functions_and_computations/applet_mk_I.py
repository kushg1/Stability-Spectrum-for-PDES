import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Slider, Button, RadioButtons

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
t = np.arange(0.0, 1.0, 0.001)
a0 = 5
f0 = 3
delta_f = 5.0
the_image = mpimg.imread('images/1mkdvC0V0.1k0.7571067811865475.png')
l = plt.imshow(the_image)
plt.axis([0, 1, -10, 10])

axcolor = 'lightgoldenrodyellow'
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=f0, valstep=delta_f)
samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)

# Allowed amplitudes
amp_list = np.arange(0,10,0.02)
# Allowed frequencies
freq_list = np.arange(0,10,0.05)
# For you, instead you will have to import all of your filenames, and extract
# the V and k values.

# Function for finding closest value in numpy array

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def update(val):
    amp = samp.val
    freq = sfreq.val
    # finds the nearest value in our list amp_list to the amp found by the
    # slider
    amp = find_nearest(amp_list, amp)
    # finds the nearest value in our list freq_list to the freq found by the
    # slider
    freq = find_nearest(freq_list, freq)
    print('amp=',amp)
    print('freq=',freq)
    the_image = mpimg.imread('images/1mkdvC0V' + str(amp) + 'k' + str(freq) + '.png')
    l = plt.imshow(the_image)
    fig.canvas.draw_idle()
sfreq.on_changed(update)
samp.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    sfreq.reset()
    samp.reset()
button.on_clicked(reset)

rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)


def colorfunc(label):
    l.set_color(label)
    fig.canvas.draw_idle()
radio.on_clicked(colorfunc)

plt.show()