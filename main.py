import matplotlib.pyplot as plt
import random
import numpy as np
# dj38 - djibouti
# wi29 - western sahara

"""Wczytywanie pliku. Nie wiem czemu na tej stronie z danymi rysunki sa obocone 
w lewo o 90 stopni dlatego troche inaczej wyglada. """


class City:
    def __init__(self, x, y, id):
        self.x = x
        self.y = y
        self.id = id

    def distance(self, city):
        """Get distance between this city and another"""
        distance = distanceTable[self.id][city.id]
        return distance

    def __repr__(self):
        return "(City " + str(self.id) + ")"


def generate_distance_table(cities):
    table = np.zeros((totalCities, totalCities))
    for i in range(totalCities):
        curr_city = cities[i]  # now doing outgoing connections from this city
        for j in range(totalCities):
            xDis = abs(curr_city.x - cities[j].x)
            yDis = abs(curr_city.y - cities[j].y)
            distance = np.sqrt((xDis ** 2) + (yDis ** 2))
            table[i][j] = distance + random.randint(-5000, 5000)
            if table[i][j] < 0: table[i][j] = 0

    return table


cities = []

f = open("wi29.txt")
for i, line in enumerate(f):
    line = line.split()
    cities.append(City(float(line[1]), float(line[2]), i))

totalCities = len(cities)
distanceTable = generate_distance_table(cities)
populationSize = 20
pc = 0.8  # Crossover probability

# Initial population
routes = []
for i in range(populationSize):
    routes.append(random.sample(cities, len(cities)))

