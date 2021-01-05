import model
import pytorch_lightning as pl
import pricing
from path_generator import BS_Generator, DscGenerator
import matplotlib.pyplot as plt
from flags import FLAGS
from pytorch_lightning.loggers import TensorBoardLogger
import pnl_loss


if __name__ == "__main__":
    logger = TensorBoardLogger('lightning_logs', name='0.99Cvar-B256')
    net = model.VanillaModel()
    generatorStock = BS_Generator(n_paths=FLAGS.SAMPLES)
    generatorDsc = DscGenerator(n_paths=FLAGS.SAMPLES)
    loss = pnl_loss.loss_CVar(0.99)
    m = model.LitModel(model=net,
                       generators=[generatorDsc, generatorStock],
                       ptf_function=model.ptf_function,
                       loss=loss)
    trainer = pl.Trainer(max_epochs=15,
                         logger=logger)
    trainer.fit(m)

    # Examples

    dl = m.val_dataloader()
    df = dl.dataset
    dates = df.dates
    print(pricing.price_BS_option(val_date=FLAGS.TODAY,
                                  maturity_date=FLAGS.MATURITY,
                                  spot_price=FLAGS.SPOT,
                                  risk_free_rate=FLAGS.RF_RATE,
                                  sigma=FLAGS.BS_SIGMA,
                                  strike_price=FLAGS.STRIKE))

    prices = []
    hedge = []
    model_h = []

    for idx in range(5):
        x, y = df[idx]
        x = x.unsqueeze(0)
        # breakpoint()
        y_hat = m(x)
        nn_hedges = y_hat[0, 1, :-1].detach().numpy()
        path = x[0, 1, :].numpy().astype(float)
        prices.extend(list(path[:-1]))
        model_hedges = pricing.delta_BS_option(val_date=dates, spot_price=path)
        hedge.extend(list(nn_hedges))
        model_h.extend(list(model_hedges[:-1]))
        # breakpoint()
        plt.figure()
        plt.subplot(121)
        plt.plot(path)
        # plt.grid(1)
        # plt.xticks(range(len(dates)), dates, rotation=90)
        plt.subplot(122)
        plt.plot(nn_hedges)
        # plt.grid(1)
        plt.title('path')
        plt.tight_layout()
        # plt.xticks(range(len(dates)), dates, rotation=90)
        plt.plot(model_hedges)
        plt.title('hedge')
        plt.legend(['nn', 'model'])
        plt.tight_layout()
        plt.show()

    plt.figure()
    plt.scatter(prices, hedge)
    plt.scatter(prices, model_h)
    plt.show()

    x, y = df.take(1000)
    y_hat = m(x)

    import torch
    MH = torch.zeros_like(x[:, :1, :])
    for i in range(x.shape[0]):
        xx = x[i, 1, :].numpy().astype(float)
        m_h = pricing.delta_BS_option(val_date=dates, spot_price=xx)
        MH[i, :, :] = torch.Tensor(m_h).unsqueeze(0)
    MH = m.ptf_function(x, MH)
    PnL_model = pnl_loss.Loss_function().compute_PnL(MH, y).detach().numpy()
    PnL_nn = pnl_loss.Loss_function().compute_PnL(y_hat, y).detach().numpy()
    plt.figure()
    plt.hist(PnL_nn, bins=50, alpha=0.5)
    plt.hist(PnL_model, bins=50, alpha=0.5)
    plt.legend(['nn', 'model'])
    plt.title('PnL distribuition')
    plt.show()
