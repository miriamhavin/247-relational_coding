import multiprocessing
import csv

def mp_worker(number):
    number += 1
    return (625, 'sss', number, number + 1)

def mp_handler():
    p = multiprocessing.Pool(32)
    numbers = list(range(1000))
    with open('results.csv', 'w') as f:
        writer=csv.writer(f, delimiter=",", lineterminator="\r\n") 
        writer.writerow(('sid','electrode','prod','comp'))
        for result in p.map(mp_worker, numbers):
            writer.writerow(result)

if __name__=='__main__':
    mp_handler()