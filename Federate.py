class FedAvg:
    def __init__(self, name):
        self.name = name
        self.total = 0
        self.count = 0

    def update(self, value):
        self.total += value
        self.count += 1

    def __str__(self):
        return '{} ({}): {}'.format(self.name, self.count, self.total)

class FedSGD:
    def __init__(self, name, lr):
        self.name = name
        self.lr = lr
        self.total = 0
        self.count = 0