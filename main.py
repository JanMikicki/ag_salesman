import matplotlib.pyplot as plt
import random
import numpy as np
import secrets
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

    def __lt__(self, other):
        return random.random() < random.random()


def generate_distance_table(cities):
    table = np.zeros((totalCities, totalCities))
    for i in range(totalCities):
        curr_city = cities[i]  # now doing outgoing connections from this city
        for j in range(totalCities):
            xDis = abs(curr_city.x - cities[j].x)
            yDis = abs(curr_city.y - cities[j].y)
            distance = np.sqrt((xDis ** 2) + (yDis ** 2))
            variation = int(distance / 5)
            table[i][j] = distance + random.randint(-variation, variation)
            if table[i][j] < 0:
                table[i][j] = 0
    return table


def get_fitness(route):
    routeDistance = 0
    for i in range(len(route)):
        routeDistance += route[i].distance(route[(i+1) % totalCities])
    return 1/routeDistance


# Krzyżowanie OX -> https://intellipaat.com/community/21890/order-crossover-ox-genetic-algorithm
# length -> ilość miast w badanym kraju, start i end to pierwsza i ostatnia pozycja z kroku 1. w linku
def ox_crossover(parent1, parent2, length, start, end):

    # Przekopiowanie losowego fragmentu parent1 do dziecka
    child = [1000] * length
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
def pmx_crossover(parent1, parent2, length, start, end):

    # KROK 1 Przekopiowanie losowego fragmentu parent1 do dziecka
    child = [1000] * length
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

f = open("dj38.txt")
for i, line in enumerate(f):
    line = line.split()
    cities.append(City(float(line[1]), float(line[2]), i))

totalCities = len(cities)
distanceTable = generate_distance_table(cities)
populationSize = 200
iterations = 200
eliteSize = int(populationSize / 4)
pc = 0.8  # Crossover probability
pm = 0.01  # Mutation probability

# Initial population
routes = []
for i in range(populationSize):
    routes.append(random.sample(cities, len(cities)))

fittest_individuals = []  # For plotting
fitness_average = []

for n in range(iterations + 1):

    # Get fitness of each route
    fitness = []
    for f in range(populationSize):
        fitness.append(get_fitness(routes[f]))

    fittest_individuals.append(1/max(fitness))  # Remember shortest route for plotting
    indx_max = np.argmax(fitness)
    final_route = routes[indx_max]

    # Na razie ruletka, czytałem że rangowo-ruletkowa jest dobra
    # Get parents
    parents = []
    fitnessSum = sum(fitness)
    fitnessAverage = fitnessSum/populationSize
    fitness_average.append(fitnessAverage)
    fitnessNormalized = [x / fitnessSum for x in fitness]

    # Sort fitness and corresponding routes in ascending order (lowest fitness/route first)
    fitnessNormalized, routes = (list(t) for t in zip(*sorted(zip(fitnessNormalized, routes))))
    distributionFunction = np.cumsum(fitnessNormalized)

    # Copy elite
    for i in range(1, eliteSize + 1):
        parents.append(routes[-i])

    # Roulette
    for i in range(eliteSize, populationSize):
        r = float(secrets.randbelow(100)) / 100.  # from 0 to 0.99
        for j in range(0, populationSize):
            if r <= distributionFunction[j]:
                parents.append(routes[j])
                break

    # Crossover
    new_population = []
    method = 'ox'

    for i in range(int(populationSize / 2)):
        # if random.random() >= pc:
        #     continue
        start = random.randint(0, totalCities - 1)
        end = random.randint(0, totalCities - 1)
        if end < start:
            end, start = start, end

        if method == 'ox':
            new_population.append(ox_crossover(parents[2 * i], parents[2 * i + 1], totalCities, start, end))
            new_population.append(ox_crossover(parents[2 * i + 1], parents[2 * i], totalCities, start, end))
        elif method == 'pmx':
            new_population.append(pmx_crossover(parents[2 * i], parents[2 * i + 1], totalCities, start, end))
            new_population.append(pmx_crossover(parents[2 * i + 1], parents[2 * i], totalCities, start, end))
        else:
            print("no crossover method")
            break

    for new_member in new_population:
        if float(secrets.randbelow(100)) / 100. < pm:
            first_swap = random.randrange(totalCities)
            second_swap = random.randrange(totalCities)
            temp = new_member[first_swap]
            new_member[first_swap] = new_member[second_swap]
            new_member[second_swap] = temp

    routes = new_population


fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel("iteracja")
ax1.set_ylabel('najlepszy dystans', color=color)
ax1.plot(fittest_individuals, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.ticklabel_format(axis='y', style='sci', scilimits=(1, 2))
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('średnie przystosowanie', color=color)  # we already handled the x-label with ax1
ax2.plot(fitness_average, color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.ticklabel_format(axis='y', style='sci', scilimits=(-1, 0))

fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.figure()
cities_x = [city.x for city in cities]
cities_y = [city.y for city in cities]
plt.scatter(cities_x, cities_y)
final_route.append(final_route[0])  # Append first element to the end so it will plot the entire route
route_x = [city.x for city in final_route]
route_y = [city.y for city in final_route]
plt.plot(route_x, route_y)
plt.show()

# plt.plot(fittest_individuals)
# plt.xlabel("iteracja")
# plt.ylabel("najlepszy dystans")
# plt.show()
