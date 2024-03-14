import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates
from datetime import datetime
from dateutil import relativedelta
import math

df_raw = pd.read_csv(os.getcwd()+"\\PhiladelphiaLandTemperatures.csv")
def load_data(filepath):
    df = (pd.read_csv(filepath)
          .dropna() # Remove rows with missing values
          .iloc[1:, :] # First row has a temp. but skips the sequence; that far back not useful
          .reset_index(drop=True)
          .rename(columns={"dt": "date", 
                           "AverageTemperature": "avg_temp", 
                           "AverageTemperatureUncertainty": "avg_temp_uncertainty", # May be interesting to plot avg. temp +/- uncertainty
                           }
          )
    )
    
    df['date'] = pd.to_datetime(df['date'])

    cols_sel = ['date', 'avg_temp', 'avg_temp_uncertainty']
    
    return df.loc[:, cols_sel]

df = load_data(os.getcwd()+"\\PhiladelphiaLandTemperatures.csv")

def get_slice(df_inp, start_date, end_date):

    slice = df_inp[(df_inp["date"] >= start_date) & (df["date"] <= end_date)]

    return slice

startdate="1990-01-01"
enddate="1992-12-31"
toDate_start=datetime.strptime(startdate, '%Y-%m-%d')
toDate_end=datetime.strptime(enddate, '%Y-%m-%d')

delta=relativedelta.relativedelta(toDate_end, toDate_start)
date_size=delta.months+(delta.years * 12)

df_slice = get_slice(df, startdate, enddate)

def train_test_split(df_slice_inp, train_size):
    train = (df_slice_inp.iloc[:train_size, :]["avg_temp"]
             .to_numpy()
             .reshape(-1, 1)
             )
    test = (df_slice_inp.iloc[train_size:, :]["avg_temp"]
            .to_numpy()
            .reshape(-1, 1)
    )

    return train, test

training_size=math.floor(2*date_size/3)
X, y = train_test_split(df_slice, training_size)
teststart=datetime.strftime(df_slice.iloc[training_size, :]["date"],'%Y-%m-%d')
testend=datetime.strftime(df_slice.iloc[-1, :]["date"],'%Y-%m-%d')


def make_RNN(X_inp, fcl_size, output_size):

    from InputLayer import InputLayer
    from FullyConnectedLayer import FullyConnectedLayer
    from TanhLayer import TanhLayer
    from LinearLayer import LinearLayer
    from SquaredError import SquaredError

    IL = InputLayer(X_inp)
    FCLU = FullyConnectedLayer(X_inp.shape[1], fcl_size)
    ACT1 = TanhLayer()
    FCLW = FullyConnectedLayer(fcl_size, fcl_size,)
    FCLV = FullyConnectedLayer(fcl_size, output_size)
    ACT2 = LinearLayer()
    SE = SquaredError()

    model = {"IL": IL, "FCLU": FCLU, "ACT1": ACT1, "FCLW": FCLW, "FCLV": FCLV, "ACT2": ACT2, "SE": SE}

    return model



rnn = make_RNN(X_inp = X, fcl_size=10, output_size=y.shape[0])


def train_RNN_inplace(model, X, y, learning_rate, epochs):

    training_dict = {"epoch": list(range(epochs)),
                     "squared_error": [],
                     "y_preds": [],}

    for epoch in range(epochs):
        # Forward
        for t in range(len(X)):
            IL_out = model["IL"].forward(X[t])
            if t > 0:
                FCLU_out = model["FCLU"].forward_with_feedback(IL_out, model["FCLW"].getPrevOut()[t-1])
            else:
                FCLU_out = model["FCLU"].forward(IL_out)
            ACT1_out = model["ACT1"].forward(FCLU_out)
            FCLW_out = model["FCLW"].forward(ACT1_out)
            FCLV_out = model["FCLV"].forward(ACT1_out)
            ACT2_out = model["ACT2"].forward(FCLV_out)

        # Predictions and Loss
        # training_dict["y_preds"].append(ACT2_out.reshape(y.shape))
        training_dict["y_preds"].append(ACT2_out)

        error = model["SE"].eval(y, ACT2_out.reshape(y.shape))
        training_dict["squared_error"].append(error)

        # Backward
        dhNext_dW = np.zeros((1, model["FCLV"].getWeights().shape[0])) # Same shape as what FCLV.backward(grad) is...
        
        for t in range(len(X)-1, -1, -1):
            grad = model["SE"].gradient(y, model["ACT2"].getPrevOut()[-1].reshape(y.shape)).reshape(1, -1)
            grad = model["ACT2"].backward(grad, t_inp=t)

            model["FCLV"].updateWeightsGradAccum(grad, t_inp=t)
            model["FCLV"].updateBiasesGradAccum(grad)

            grad = model["FCLV"].backward(grad) + dhNext_dW
            # if t == len(X)-1:
                # print(f"FCLV gradient shape is: {grad.shape}") # Verify gradient shape
            grad = model["ACT1"].backward(grad, t_inp=t)

            if t > 0:
                model["FCLW"].updateWeightsGradAccum(grad, t_inp=t-1) # Have to use t-1
                model["FCLW"].updateBiasesGradAccum(grad)

            model["FCLU"].updateWeightsGradAccum(grad, t_inp=t)
            model["FCLU"].updateBiasesGradAccum(grad)

            dhNext_dW = model["FCLW"].backward(grad)

        # Update weights   
        model["FCLU"].updateWeights(grad, eta=learning_rate)
        model["FCLV"].updateWeights(grad, eta=learning_rate)
        model["FCLW"].updateWeights(grad, eta=learning_rate)

    return pd.DataFrame(training_dict)



lr = 0.0001
rnn = make_RNN(X_inp = X, fcl_size=20, output_size=y.shape[0]) # Have to reinitialize
training_results = train_RNN_inplace(rnn, X, y, learning_rate=lr, epochs=60)

training_results.plot(x="epoch", y="squared_error")

graph_index_lst = [0, 11, 19, 22]
colors = ["red", "magenta", "purple"]
y_preds_graph = training_results["y_preds"].loc[graph_index_lst].to_numpy()

fig, ax = plt.subplots(figsize=(10, 6))
t_vals_x = np.arange(len(X))
t_vals_y = np.arange(len(X), len(X)+len(y))

actual_df = get_slice(df, teststart, testend)
actual=(actual_df.iloc[:, :]["avg_temp"]
             .to_numpy()
             .reshape(-1, 1)
             )


ax.plot(t_vals_x, X.flatten(), color="green", linestyle="solid", label="Training Sequence", linewidth=2, marker="o")
ax.plot(t_vals_y, y.flatten(), color="blue", linestyle="solid", label="Predict Sequence", linewidth=2, marker="o")
ax.plot(t_vals_y, actual.flatten(), color="black", linestyle="solid", label="Actual", linewidth=2, marker="x")

colors = ["red", "magenta", "pink", "purple"]
for ind, y_pred_seq, color in zip(graph_index_lst, y_preds_graph, colors):
    ax.plot(t_vals_y, y_pred_seq.flatten(), color = color, linestyle="dashed", label=f"{ind} epochs", linewidth=2, marker="x")

# Set the ticks as dates
start_date = pd.Timestamp(startdate)
end_date = start_date + pd.Timedelta(days=len(X) + len(y))
date_range = pd.date_range(start=start_date, periods=len(X) + len(y), freq='M')
ax.set_xticks(np.arange(0, len(X) + len(y), 1))
ax.set_xticklabels(date_range.strftime('%Y-%m'))
#ax.xaxis.set_major_locator(plt.MaxNLocator(18))

every_nth = 5
for n, label in enumerate(ax.xaxis.get_ticklabels()):
    if n % every_nth != 0:
        label.set_visible(False)

for label in ax.get_xticklabels(which='major'):
    label.set(rotation=30, horizontalalignment='right')


ax.set_xlabel("Time", fontsize=16)
ax.set_ylabel("y(t)", fontsize=16)
ax.set_title(f"Predictions at Different Epochs, $\eta$ = {lr}", 
             fontsize=16)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
ax.legend(fontsize=14)
ax.grid()
plt.show()