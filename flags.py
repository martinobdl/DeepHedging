import QuantLib as ql
import datetime


class flags():
    """flags."""

    def __init__(self):
        self.SPOT = 100
        self.BS_DRIFT = 0.1
        self.BS_SIGMA = 0.2
        self.STRIKE = 100
        self.SMALL_SAMPLE = 500
        self.SAMPLES = 100000
        self.RF_RATE = 0.0
        self.TODAY = ql.Date(datetime.date.today().day, datetime.date.today().month, datetime.date.today().year)
        calendar = ql.UnitedStates()
        period = ql.Period(60, ql.Days)
        self.MATURITY = calendar.advance(self.TODAY, period)
        self.BATCH_SIZE = 256


FLAGS = flags()
ql.Settings.instance().evaluationDate = FLAGS.TODAY
ql.Settings.instance().includeReferenceDateEvents = True
