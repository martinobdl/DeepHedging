import QuantLib as ql
from flags import FLAGS
import numpy as np


def _gbm_option(val_date=FLAGS.TODAY,
                maturity_date=FLAGS.MATURITY,
                spot_price=FLAGS.SPOT,
                risk_free_rate=FLAGS.RF_RATE,
                sigma=FLAGS.BS_SIGMA,
                strike_price=FLAGS.STRIKE):
    """_gbm_option. Helper to create the option object in QuantLib

    :param val_date: valuation date
    :param maturity_date: maturity of the call option
    :param spot_price:
    :param risk_free_rate: risk free rate for the BS model
    :param sigma: volatility of the BS model
    :param strike_price: strike price of the call option
    """

    option_type = ql.Option.Call
    day_count = ql.Actual365Fixed()
    calendar = ql.UnitedStates()
    payoff = ql.PlainVanillaPayoff(option_type, strike_price)
    eu_exercise = ql.EuropeanExercise(maturity_date)
    european_option = ql.VanillaOption(payoff, eu_exercise)

    spot_handle = ql.QuoteHandle(
        ql.SimpleQuote(spot_price)
    )

    flat_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(val_date, risk_free_rate, day_count)
    )

    dividend_rate = 0
    dividend_yield = ql.YieldTermStructureHandle(
        ql.FlatForward(val_date, dividend_rate, day_count)
    )

    flat_vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(val_date, calendar, sigma, day_count)
    )

    bsm_process = ql.BlackScholesMertonProcess(spot_handle,
                                               dividend_yield,
                                               flat_ts,
                                               flat_vol_ts)

    european_option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))

    return european_option


def _price_BS_option(val_date=FLAGS.TODAY,
                     maturity_date=FLAGS.MATURITY,
                     spot_price=FLAGS.SPOT,
                     risk_free_rate=FLAGS.RF_RATE,
                     sigma=FLAGS.BS_SIGMA,
                     strike_price=FLAGS.STRIKE):
    """_price_BS_option.

    :param val_date:
    :param maturity_date:
    :param spot_price:
    :param risk_free_rate:
    :param sigma:
    :param strike_price:
    """

    option = _gbm_option(val_date=val_date,
                         maturity_date=maturity_date,
                         spot_price=spot_price,
                         risk_free_rate=risk_free_rate,
                         sigma=sigma,
                         strike_price=strike_price)

    return option.NPV()


def _delta_BS_option(val_date=FLAGS.TODAY,
                     maturity_date=FLAGS.MATURITY,
                     spot_price=FLAGS.SPOT,
                     risk_free_rate=FLAGS.RF_RATE,
                     sigma=FLAGS.BS_SIGMA,

                     strike_price=FLAGS.STRIKE):
    """_delta_BS_option.

    :param val_date:
    :param maturity_date:
    :param spot_price:
    :param risk_free_rate:
    :param sigma:
    :param strike_price:
    """

    option = _gbm_option(val_date=val_date,
                         maturity_date=maturity_date,
                         spot_price=spot_price,
                         risk_free_rate=risk_free_rate,
                         sigma=sigma,
                         strike_price=strike_price)

    return option.delta()


def price_BS_option(val_date=FLAGS.TODAY,
                    maturity_date=FLAGS.MATURITY,
                    spot_price=FLAGS.SPOT,
                    risk_free_rate=FLAGS.RF_RATE,
                    sigma=FLAGS.BS_SIGMA,
                    strike_price=FLAGS.STRIKE):
    """price_BS_option.

    :param val_date:
    :param maturity_date:
    :param spot_price:
    :param risk_free_rate:
    :param sigma:
    :param strike_price:
    """

    if isinstance(val_date, ql.Date) and isinstance(spot_price, (int, float)):
        return _price_BS_option(val_date, maturity_date, spot_price, risk_free_rate, sigma, strike_price)

    prices = []
    assert(len(val_date) == len(spot_price))
    for date, s0 in zip(val_date, spot_price):
        prices.append(_price_BS_option(date, maturity_date, s0, risk_free_rate, sigma, strike_price))
    return np.array(prices)


def delta_BS_option(val_date=FLAGS.TODAY,
                    maturity_date=FLAGS.MATURITY,
                    spot_price=FLAGS.SPOT,
                    risk_free_rate=FLAGS.RF_RATE,
                    sigma=FLAGS.BS_SIGMA,
                    strike_price=FLAGS.STRIKE):
    """delta_BS_option.

    :param val_date:
    :param maturity_date:
    :param spot_price:
    :param risk_free_rate:
    :param sigma:
    :param strike_price:
    """

    if isinstance(val_date, ql.Date) and isinstance(spot_price, (int, float)):
        return _delta_BS_option(val_date, maturity_date, spot_price, risk_free_rate, sigma, strike_price)

    prices = []
    assert(len(val_date) == len(spot_price))
    for date, s0 in zip(val_date, spot_price):
        prices.append(_delta_BS_option(date, maturity_date, s0, risk_free_rate, sigma, strike_price))
    return np.array(prices)


if __name__ == '__main__':
    pass
