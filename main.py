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


def get_fitness(route):
    routeDistance = 0
    for i in range(len(route)):
        routeDistance += route[i].distance(route[(i+1) % 4])
    return 1/routeDistance


# List of all cities loaded from file
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

# Get fitness of each route
fitness = []
for i in range(populationSize):
    fitness.append(get_fitness(routes[i]))

# Na razie ruletka, czytałem że rangowo-ruletkowa jest dobra
# Get parents
parents = []
fitnessSum = sum(fitness)
fitnessNormalized = [x / fitnessSum for x in fitness]

# Sort fitness and corresponding routes in ascending order (lowest fitness/route first)
fitnessNormalized, routes = (list(t) for t in zip(*sorted(zip(fitnessNormalized, routes))))
distributionFunction = np.cumsum(fitnessNormalized)

# Roulette
for i in range(populationSize):
    r = random.random()  # from 0 to 1
    for j in range(len(fitnessNormalized)):
        if r <= distributionFunction[j]:
            parents.append(routes[j])
            break

# Teraz mamy 20 rodziców. Trzeba dobrać ich w pary i zrobić krzyżowanie i mutację, tak aby powstało 20 dzieci (nowa populacja)
new_population = []

