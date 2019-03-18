import matplotlib.pyplot as plt
import numpy as np


input_data = np.arange(0,np.pi*2, 0.1)
correct_data = np.sin(input_data)


plt.plot(input_data, correct_data)
plt.show()
