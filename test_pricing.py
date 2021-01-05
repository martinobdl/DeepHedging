import unittest
import pricing
from flags import FLAGS
import QuantLib as ql


class TestPricingGBM(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        self.maturity = FLAGS.TODAY+ql.Period(1, ql.Years)
        self.yf = ql.Actual365Fixed().yearFraction(FLAGS.TODAY, self.maturity)

    @classmethod
    def tearDownClass(self):
        pass

    def test_yf(self):
        self.assertAlmostEqual(self.yf, 1, places=7)

    def test_price1(self):

        price = pricing._price_BS_option(val_date=FLAGS.TODAY,
                                         maturity_date=self.maturity,
                                         spot_price=16,
                                         risk_free_rate=0.04,
                                         sigma=0.40,
                                         strike_price=18)
        self.assertAlmostEqual(price, 2.04083, places=4)

    def test_price2(self):

        price = pricing._price_BS_option(val_date=FLAGS.TODAY,
                                         maturity_date=FLAGS.TODAY,
                                         spot_price=16,
                                         risk_free_rate=0.04,
                                         sigma=0.40,
                                         strike_price=18)
        self.assertAlmostEqual(price, 0, places=4)

    def test_price3(self):
        price = pricing._price_BS_option(val_date=FLAGS.TODAY,
                                         maturity_date=FLAGS.TODAY,
                                         spot_price=18,
                                         risk_free_rate=0.04,
                                         sigma=0.40,
                                         strike_price=16)
        self.assertAlmostEqual(price, 2, places=4)

    def test_delta(self):
        delta = pricing._delta_BS_option(val_date=FLAGS.TODAY,
                                         maturity_date=self.maturity,
                                         spot_price=8,
                                         risk_free_rate=0.04,
                                         sigma=0.40,
                                         strike_price=8)

        self.assertAlmostEqual(delta, 0.61791, places=4)


if __name__ == "__main__":
    unittest.main()
