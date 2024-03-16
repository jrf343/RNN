# Descriptions and Parameters for saved models

Load models with pickle package

* `rnn_tanh_1980_opt`
    * X: 1980-1981, Jan start (2 years / 24 months)
    * y: 1982, Jan start (1 year / 12 months)
        * sizeout of FCLV is 12
    * FCLU: sizeout=20 parameters (weights for FCLW 20x20 and FCLV 20x12 based off that)
    * ACT1: Tanh
    * RMSE validation avg. +/- RMSE stdev across other years = 3.10 +/- 0.55
        * ALEC: This was better in an earlier run...
    * Notes:
        * "opt" because used while loop to get to minimum error instead of for loop with # of epochs
            * Issue with getting consistent results with defined number of epochs even with using fixed (non-random) state
        * Ultimately overfits and just guesses values for 1982

* `rnn_tanh_batches_6batches`
    * X: 6 batches of (2 years / 24 months) each starting at:
        * years_train = [1980, 1983, 1990, 1993, 2000, 2003]
    * y: 6 batches of (1 year, 12 months) each starting at years_train + 2
        * 1982, 1985, 1992, 1995, 2002, 2005
    * FCLU: sizeout=20 parameters (weights for FCLW 20x20 and FCLV 20x12 based off that)
    * ACT1: Tanh
    * RMSE train average (across batches) = 2.85
    * epochs: 36
        * One epoch short of getting the "lowest" RMSE train avg. of 2.23
    * Notes:
        * Trained with for loop over num_epochs (was able to get consistent loading/results)
        * While it doesn't overfit a single batch, it predicts nearly the same thing for each when minimizing the average RMSE for training

    