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
            if table[i][j] < 0:
                table[i][j] = 0
    return table


def get_fitness(route):
    routeDistance = 0
    for i in range(len(route)):
        routeDistance += route[i].distance(route[(i+1) % 4])
    return 1/routeDistance


# Krzyżowanie OX -> https://intellipaat.com/community/21890/order-crossover-ox-genetic-algorithm
# length -> ilość miast w badanym kraju, start i end to pierwsza i ostatnia pozycja z kroku 1. w linku
def ox_crossover(parent1, parent2, length):

    # Przekopiowanie losowego fragmentu parent1 do dziecka
    child = [1000] * length
    start = random.randint(0, length - 1)
    end = random.randint(0, length - 1)
    if end < start:
        end, start = start, end
    for i in range(start, end + 1):
        child[i] = parent1[i]

    # Przepisanie do dziecka po kolei elementów parent2, których jeszcze w dziecku nie ma
    for i in range(length):
        used = False
        for j in range(length):
            if child[j] == parent2[i]:
                used = True
        if used is False:
            for j in range(length):
                if child[j] == 1000:
                    child[j] = parent2[i]
                    break
    return child


# Krzyżowanie PMX -> http://www.rubicite.com/Tutorials/GeneticAlgorithms/CrossoverOperators/PMXCrossoverOperator.aspx/
def pmx_crossover(parent1, parent2, length):

    # KROK 1 Przekopiowanie losowego fragmentu parent1 do dziecka
    child = [1000] * length
    start = random.randint(0, length - 1)
    end = random.randint(0, length - 1)
    if end < start:
        end, start = start, end
    for i in range(start, end + 1):
        child[i] = parent1[i]

    # KROK 2 Przeszukanie obszaru w parent2
    for i in range(start, end + 1):
        present = False
        for j in range(start, end + 1):
            if child[j] == parent2[i]:
                present = True
        val = parent1[i]
        while present is False:
            for j in range(length):
                if parent2[j] == val:
                    if j < start or j > end:
                        child[j] = parent2[i]
                        present = True
                    else:
                        val = parent1[j]

    # KROK 3 Wypełnienie reszty miejsc po kolei wartościami z parent2
    for i in range(length):
        used = False
        for j in range(length):
            if child[j] == parent2[i]:
                used = True
        if used is False:
            for j in range(length):
                if child[j] == 1000:
                    child[j] = parent2[i]
                    break
    return child


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

# Jakby sie chciało ręcznie sprawdzić czy dobrze sie krzyżują rodzice
# print(parents[0])
# print(parents[1])
# print(ox_crossover(parents[0], parents[1], totalCities))
# Test PMX z wartościami z linku
# A = [8, 4, 7, 3, 6, 2, 5, 1, 9, 0]
# B = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# print(pmx_crossover(A, B, 10))

# Teraz mamy 20 rodziców. Trzeba dobrać ich w pary i zrobić krzyżowanie i mutację, tak aby powstało 20 dzieci (nowa populacja)
new_population = []
method = 'ox'

for i in range(int(populationSize / 2)):
    if method == 'ox':
        new_population.append(ox_crossover(parents[2 * i], parents[2 * i + 1], totalCities))
        new_population.append(ox_crossover(parents[2 * i + 1], parents[2 * i], totalCities))
    elif method == 'pmx':
        new_population.append(pmx_crossover(parents[2 * i], parents[2 * i + 1], totalCities))
        new_population.append(pmx_crossover(parents[2 * i + 1], parents[2 * i], totalCities))
    else:
        print("nie ma metody")
        break
