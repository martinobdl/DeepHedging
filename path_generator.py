import QuantLib as ql
import numpy as np
from flags import FLAGS


def scheduler(tradeDate, maturity_date):
    """scheduler.

    :param tradeDate:
    :param maturity_date:
    """

    calendar = ql.UnitedStates()

    schedule = ql.Schedule(tradeDate,
                           maturity_date,
                           ql.Period(ql.Daily),
                           calendar,
                           ql.Following,
                           ql.Following,
                           ql.DateGeneration.Forward,
                           False,
                           tradeDate)
    return list(schedule)


class DscGenerator:
    def __init__(self,
                 tradeDate=FLAGS.TODAY,
                 maturity_date=FLAGS.MATURITY,
                 rf_rate=FLAGS.RF_RATE,
                 n_paths=FLAGS.SMALL_SAMPLE):

        self.dates = scheduler(tradeDate, maturity_date)
        self.yearfraction = [ql.Actual365Fixed().yearFraction(self.dates[0], d) for d in self.dates]
        self.n_paths = n_paths
        self.nSteps = len(self.yearfraction) - 1
        rate = ql.SimpleQuote(rf_rate)
        disc_curve = ql.FlatForward(tradeDate, ql.QuoteHandle(rate), ql.Actual365Fixed())
        disc_curve.enableExtrapolation()
        self.discount = np.vectorize(ql.YieldTermStructureHandle(disc_curve).discount)
        self.paths = np.zeros(shape=((self.n_paths), self.nSteps + 1))
        for i in range(self.n_paths):
            self.paths[i] = self.discount(self.yearfraction)

    def generate(self, idx):
        return self.paths[idx, :], self.dates


class Generator:
    """Generator."""

    def __init__(self,
                 tradeDate=FLAGS.TODAY,
                 maturity_date=FLAGS.MATURITY,
                 spot_price=FLAGS.SPOT,
                 process_parameters={'drift': FLAGS.BS_DRIFT, 'sigma': FLAGS.BS_SIGMA},
                 n_paths=FLAGS.SMALL_SAMPLE):
        """__init__.

        :param tradeDate:
        :param maturity_date:
        :param spot_price:
        :param process_parameters:
        :param n_paths:
        """

        self.tradeDate = tradeDate
        self.maturity_date = maturity_date
        self.spot_price = spot_price
        self.process_parameters = process_parameters
        self.n_paths = n_paths
        self.dates = scheduler(self.tradeDate, self.maturity_date)
        self._check()
        day_count = ql.Actual365Fixed()
        self.maturity = day_count.yearFraction(self.tradeDate, self.maturity_date)

    def _check(self):
        """_check."""
        raise NotImplementedError("This method have to be implemented by subclasses.")

    def generate():
        """generate."""
        raise NotImplementedError("This method have to be implemented by subclasses.")

    def GeneratePaths(self, process, nSteps):
        """GeneratePaths.

        :param process:
        :param maturity:
        :param nPaths:
        :param nSteps:
        """
        generator = ql.UniformRandomGenerator()
        sequenceGenerator = ql.UniformRandomSequenceGenerator(nSteps, generator)
        gaussianSequenceGenerator = ql.GaussianRandomSequenceGenerator(sequenceGenerator)
        paths = np.zeros(shape=((self.n_paths), nSteps + 1))
        pathGenerator = ql.GaussianPathGenerator(process, self.maturity, nSteps, gaussianSequenceGenerator, False)
        for i in range(0, self.n_paths, 2):
            path = pathGenerator.next().value()
            paths[i, :] = np.array([path[j] for j in range(nSteps + 1)])
            anti_path = pathGenerator.antithetic().value()
            paths[i+1, :] = np.array([anti_path[j] for j in range(nSteps + 1)])
        return paths


class BS_Generator(Generator):
    """BS_Generator."""

    def __init__(self, **kwargs):
        super(BS_Generator, self).__init__(**kwargs)
        drift = self.process_parameters['drift']
        sigma = self.process_parameters['sigma']

        GBM = ql.GeometricBrownianMotionProcess(self.spot_price, drift, sigma)
        self.gbm_paths = self.GeneratePaths(GBM, len(self.dates)-1)

    def _check(self):
        assert 'drift' in self.process_parameters.keys()
        assert 'sigma' in self.process_parameters.keys()

    def generate(self, idx):

        return self.gbm_paths[idx, :], self.dates


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    gbm_path, time_grid = BS_Generator(n_paths=FLAGS.SMALL_SAMPLE).generate(0)

    plt.figure()
    plt.plot(np.transpose(gbm_path), color='blue', alpha=0.5, marker='*')
    plt.grid(1)
    plt.xticks(range(len(time_grid)), time_grid, rotation=90)
    plt.title('Prices')
    plt.tight_layout()
    plt.show()

    dsc_path, time_grid = DscGenerator(n_paths=FLAGS.SMALL_SAMPLE).generate(0)

    plt.figure()
    plt.plot(np.transpose(dsc_path), color='blue', alpha=0.5, marker='*')
    plt.grid(1)
    plt.xticks(range(len(time_grid)), time_grid, rotation=90)
    plt.title('Prices')
    plt.tight_layout()
    plt.show()
