import numpy as np
import sqlite3
import string

package_types = ['R', 'L', 'V', 'C', 'E', 'U', 'Z']
package_types_probabilities = [1 / len(package_types) for _ in package_types]

myConnection = sqlite3.connect('data/factbook.db')
myCursor = myConnection.cursor()
myQuery = """SELECT population, name,  code FROM facts ORDER BY population DESC LIMIT 20;"""
myCursor.execute(myQuery)
Large_countries = myCursor.execute(myQuery).fetchall()
all_countries_keys = [s[2].upper() for s in Large_countries[1:]]


def type_generator():
    return np.random.choice(package_types, p=package_types_probabilities)


def letter_generator():
    return np.random.choice([chr(i) for i in range(65, 91)])


def numbers_generator():
    return np.random.randint(1, 9, 8)


def control_number(numbers):
    coeffs = np.array([8, 6, 4, 2, 3, 5, 9, 7])
    result = 11 - np.sum(coeffs * numbers) % 11
    if result == 10:
        result = 0
    elif result == 11:
        result = 5
    return result


def country_generator():
    return np.random.choice(all_countries_keys)


def generate_tracking_number():
    numbers = numbers_generator()
    return type_generator() + letter_generator() + "".join([str(n) for n in numbers]) + str(
        control_number(numbers)) + country_generator()


f = open("data/EMS_track_numbers.txt", 'w')
for i in range(1, 100):
    f.write(str(i) + " " + generate_tracking_number() + '\n')
