import numpy as np
import unittest
import path_generator
from flags import FLAGS
import QuantLib as ql


class TestDsc(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        n_paths = 1
        print(f"Generating {n_paths} Dsc paths")
        self.paths, self.dates = path_generator.DscGenerator(tradeDate=FLAGS.TODAY,
                                                             maturity_date=FLAGS.MATURITY,
                                                             rf_rate=FLAGS.RF_RATE,
                                                             n_paths=n_paths).generate()

    @classmethod
    def tearDownClass(self):
        pass

    def test_rf(self):
        yf = np.array([ql.Actual365Fixed().yearFraction(self.dates[i-1], self.dates[i])
                       for i in range(1, len(self.dates))])
        list1 = self.paths[0, 1:]/self.paths[0, :-1]
        list2 = np.exp(-yf*FLAGS.RF_RATE)
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(a, b)


class TestGBM(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        n_paths = int(2.5*10**5)
        print(f"Generating {n_paths} GBM paths")
        self.maturity = FLAGS.TODAY+ql.Period(1, ql.Years)
        self.yf = ql.Actual365Fixed().yearFraction(FLAGS.TODAY, self.maturity)
        self.paths, _ = path_generator.BS_Generator(tradeDate=FLAGS.TODAY,
                                                    maturity_date=self.maturity,
                                                    spot_price=100,
                                                    process_parameters={'drift': 0.05, 'sigma': 0.2},
                                                    n_paths=n_paths).generate()

    @classmethod
    def tearDownClass(self):
        pass

    def test_mean(self):
        mean_empirical = np.mean(np.log(self.paths[:, -1]/100))
        mean_theory = (0.05-0.2**2/2)*self.yf
        self.assertAlmostEqual(mean_empirical, mean_theory, places=3)

    def test_vol(self):
        std_empirical = np.std(np.log(self.paths[:, -1]/100))
        std_theory = 0.2*np.sqrt(self.yf)
        self.assertAlmostEqual(std_empirical, std_theory, places=3)

    def test_yf(self):
        self.assertAlmostEqual(self.yf, 1, places=7)


if __name__ == "__main__":
    unittest.main()
