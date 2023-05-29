import statistics

def analysis(data):
    mean = statistics.mean(data)
    stdev = statistics.stdev(data)

    print("Mean: ", mean * 100)
    print("Standard deviation: ", stdev * 100)

A = [0.800, 0.798, 0.803]
F = [0.001, 0.000, 0.002]
I = [0.046, 0.047, 0.046]

analysis(A)
analysis(F)
analysis(I)