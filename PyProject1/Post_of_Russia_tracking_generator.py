import numpy as np
from dbfread import DBF

db = DBF("data/PIndx21.dbf")
inds = []
for x in db:
    inds.append(x['INDEX'])


def index_generator():
    return np.random.choice(inds)


def month_generator():
    n = np.random.randint(1, 100)
    return '0' + str(n) if n < 10 else str(n)


def numbers_generator():
    numbers = np.random.randint(0, 10, 5)
    return "".join([str(n) for n in numbers])


def control_number(numbers):
    sum = np.sum([3 * numbers[i] if i % 2 == 0 else numbers[i] for i in range(len(numbers))])
    result = 10 - sum % 10
    return result if result < 10 else 0


def generate_tracking_number():
    result = index_generator() + month_generator() + numbers_generator()
    numbers = [ord(c) for c in result]
    return result + str(control_number(numbers))


f = open("data/Post_of_Russia_track_numbers.txt", 'w')
for i in range(1, 100):
    f.write(str(i) + " " + generate_tracking_number() + '\n')
