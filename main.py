import matplotlib.pyplot as plt

# dj38 - djibouti
# wi29 - western sahara

"""Wczytywanie pliku. Nie wiem czemu na tej stronie z danymi rysunki sa obocone 
w lewo o 90 stopni dlatego troche inaczej wyglada. """


x, y = [], []
with open("wi29.txt") as f:
    for line in f:
        line = line.split()
        x.append(float(line[1]))
        y.append(float(line[2]))


print(x)
plt.scatter(x, y)
plt.show()
