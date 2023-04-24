
import numpy as np

# glorot_normal
fan_in = 10
fan_out = 8
scale_value = np.sqrt(2/(fan_in + fan_out))

print(f'scale : {scale_value}')

weights = np.random.normal(loc=0.0, scale=scale_value, size=(10,10))

