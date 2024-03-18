# Descriptions and Parameters for saved models

Load models with pickle package

* `rnn_tanh_batches_6batches_100epochs.pkl`
    * X: 6 batches of (2 years / 24 months) each starting at:
        * years_train = [1980, 1983, 1990, 1993, 2000, 2003]
    * y: 6 batches of (1 year, 12 months) each starting at years_train + 2
        * 1982, 1985, 1992, 1995, 2002, 2005
    * FCLU: sizeout=20 parameters (weights for FCLW 20x20 and FCLV 20x12 based off that)
    * ACT1: Tanh
    * RMSE Validation = 1.78 +/- 0.38
    * epochs: 100
        * Early stopping, continued to slightly increase below 2.43 by 100 epochs but think overfitting
    * Notes:
        * While it doesn't overfit a single batch, it predicts nearly the same thing for each when minimizing the average RMSE for training

* `rnn_relu_batches_6batches_100epochs.pkl`
    * X: 6 batches of (2 years / 24 months) each starting at:
        * years_train = [1980, 1983, 1990, 1993, 2000, 2003]
    * y: 6 batches of (1 year, 12 months) each starting at years_train + 2
        * 1982, 1985, 1992, 1995, 2002, 2005
    * FCLU: sizeout=20 parameters (weights for FCLW 20x20 and FCLV 20x12 based off that)
    * ACT1: ReLU
    * RMSE Validation = 1.62 +/- 0.35
    * epochs: 700
    * Notes:
        * While it doesn't overfit a single batch, it predicts nearly the same thing for each when minimizing the average RMSE for training

    