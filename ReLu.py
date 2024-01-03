
def relu(x):
    return max(0, x)

def reluPrime(x):
    return 1 if x > 0 else 0
