import matplotlib.pyplot as plt
keys=[44.054054054054056, 5.675675675675676, 5.135135135135135, 11.891891891891893, 29.45945945945946, 3.5135135135135136, 0.0]

values = ['Happy', 'Angry', 'Neutral', 'Sad', 'Fearful', 'Surprised', 'Disgusted']

plt.bar(values, keys, color =['orange','red', 'blue', 'black', 'purple', 'yellow','green'])
plt.show()
