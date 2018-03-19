import numpy as np
from skopt.space import Real, Integer, Categorical


class RealGenerator:
    def __init__(self, real):
        self.low = real.low
        self.high = real.high

    def ask(self, n_points=1):
        if n_points < 1:
            raise ValueError("n_points should be greater than 1")
        return np.random.uniform(self.low, self.high, n_points)


class IntegerGenerator:
    def __init__(self, integer):
        self.low = integer.low
        self.high = integer.high

    def ask(self, n_points=1):
        if n_points < 1:
            raise ValueError("n_points should be greater than 1")
        return np.random.random_integers(self.low, self.high, n_points)


class CategoricalGenerator:
    def __init__(self, category):
        self.category = category.categories

    def ask(self, n_points=1):
        if n_points < 1:
            raise ValueError("n_points should be greater than 1")
        return np.random.choice(self.category, n_points)


class RandomGridSearch:
    def __init__(self, dimensions, random_state=1):
        self.dimensions = dimensions
        self.sampler = self.sampler_generator()
        self.explored = set()
        self.y = []
        self.x = []
        self.random_state = random_state
        np.random.seed(self.random_state)

    def ask(self, n_points=1):
        if n_points < 1:
            raise ValueError("n_points should be greater than 1")
        params = []
        count = 0
        explored = set()
        while count < n_points:
            param = list(map(lambda x: x.ask()[0], self.sampler))
            check = "".join((map(str, param)))
            if check not in self.explored and check not in explored:
                explored.add(check)
                params.append(param)
                count += 1
        if n_points == 1:
            return params[0]
        return params

    def sampler_generator(self):
        sampler = []
        for each in self.dimensions:
            if isinstance(each, Real):
                sampler.append(RealGenerator(each))
            elif isinstance(each, Categorical):
                sampler.append(Categorical(each))
            elif isinstance(each, Integer):
                sampler.append(IntegerGenerator(each))
            else:
                raise TypeError
        return sampler

    def tell(self, x, y):
        if hasattr(y, '__iter__'):
            if len(x) != len(y):
                raise ValueError("The Dimensions of X and Y do not match")
            for i in range(len(y)):
                self.y.append(y[i])
                self.x.append(x[i])
                check = "".join((map(str, x[i])))
                self.explored.add(check)
        else:
            self.y.append(y)
            check = "".join((map(str, x)))
            self.x.append(x)
            self.explored.add(check)


def main():
    num_leaves = Integer(20, 50)
    num_boost_round = Integer(100, 300)
    learning_rate = Real(0.1, 1)
    lambda_l1 = Real(0, 1)
    lambda_l2 = Real(0, 1)
    # Dimensions
    dimensions = [num_leaves, num_boost_round, learning_rate, lambda_l1, lambda_l2]
    opt = RandomGridSearch(dimensions)
    z = opt.ask()
    print(z)
    opt.tell(z, 1)
    print(opt.x)
    z1 = opt.ask()
    opt.tell(z1, 1)
    print(opt.x)


if __name__ == '__main__':
    main()
