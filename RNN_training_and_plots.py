import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
# import pickle as pkl
np.random.seed(0)

###----------------- FUNCTIONS -----------------###
'''
Load the data from the file and return a dataframe with the 
date, average temperature, and average temperature uncertainty.
'''
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

'''
This function takes a dataframe and a date range and 
returns a slice of the dataframe that falls within the date range.
'''
def get_slice(df_inp, start_date, end_date):

    slice = df_inp[(df_inp["date"] >= start_date) & (df["date"] <= end_date)]

    return slice

'''
This function takes a dataframe and a list of years and returns a list of slices of the dataframe.
Each slice is a 3-year period starting from the year
'''
def get_slice_list(years, df_inp):
    slice_list = []
    for year in years:
        slice = get_slice(df_inp, f"{year}-01-01", f"{year+2}-12-31")
        slice_list.append(slice)
    return slice_list

'''
Take a dataframe slice and size for training set and return the train and test sets.
Temporal so takes the first train_size rows as the training set and the rest as the test set.
'''
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

'''
Take a list of dataframe slices and a size for the training set and return the train and test sets.
X_train will be a 3D array with dimensions (num_slices, train_size, 1)
y_train will be a 3D array with dimensions (num_slices, (slice_size - train_size), 1)
'''
def train_test_split_batches(df_slice_lst, train_size):
    X_lst = []
    y_lst = []

    for df_slice in df_slice_lst:
        X, y = train_test_split(df_slice, train_size)
        X_lst.append(X)
        y_lst.append(y)

    X_train = np.array(X_lst)
    y_train = np.array(y_lst)

    return X_train, y_train

'''
Returns a dictionary for a simple RNN model with the following layers:
    - InputLayer
    - FullyConnectedLayer (FCLU, weights size D x K)
    - TanhLayer
    - FullyConnectedLayer (FCLW, weights size K x K) - for feedback 
    - FullyConnectedLayer (FCLV, weights size K x output_size) - for output
        * Must be reshaped to dimensions of labels to calculate errors
    - LinearLayer
    - SquaredError

'''
def make_RNN(X_inp, fcl_size, output_size, act_func="tanh"):

    from Layers.InputLayer import InputLayer
    from Layers.FullyConnectedLayer import FullyConnectedLayer
    from Layers.TanhLayer import TanhLayer
    from Layers.ReLULayer import ReLULayer
    from Layers.LinearLayer import LinearLayer
    from LossFunctions.SquaredError import SquaredError

    IL = InputLayer(X_inp)
    FCLU = FullyConnectedLayer(X_inp.shape[1], fcl_size, random=False, randstate=0)
    if act_func == "relu":
        ACT1 = ReLULayer()
    else:
        ACT1 = TanhLayer()

    FCLW = FullyConnectedLayer(fcl_size, fcl_size, random=False, randstate=1)
    FCLV = FullyConnectedLayer(fcl_size, output_size, random=False, randstate=2)
    ACT2 = LinearLayer()
    SE = SquaredError()

    model = {"IL": IL, "FCLU": FCLU, "ACT1": ACT1, "FCLW": FCLW, "FCLV": FCLV, "ACT2": ACT2, "SE": SE}

    return model

'''
Save the model to a pickle file.
Automatically puts in a subdirectory called "Models"
'''
def save_model_pickle(model, filename):
    import pickle

    with open(f"./Models/{filename}", "wb") as f:
        pickle.dump(model, f)

'''
Load the model from a pickle given filepath
'''
def load_model_pickle(filepath):
    import pickle

    with open(filepath, "rb") as f:
        model = pickle.load(f)

    return model

'''
Train RNN model in place and return training data as a dataframe.
Expects X and y to be 3D arrays with dimensions (batches, slice_size, 1)
'''
def train_RNN_inplace_with_batches_for(model, X, y, learning_rate, epochs):

    training_dict = {"epoch": list(range(epochs)),
                     "squared_error_avg": [],
                    #  "y_preds": [],
                     }

    for epoch in range(epochs):
        error_batches = []
        y_preds_batches = []

        for batch in range(len(X)):
            # reset model at start of batch, otherwise cycling over t will use the first t's output as the previous output
            model["FCLU"].reset()
            model["FCLW"].reset()
            model["FCLV"].reset()
            model["ACT1"].setPrevIn([])
            model["ACT1"].setPrevOut([])
            model["ACT2"].setPrevIn([])
            model["ACT2"].setPrevOut([])
            
            # Forward
            for t in range(len(X[batch])):
                IL_out = model["IL"].forward(X[batch][t])
                if t > 0:
                    FCLU_out = model["FCLU"].forward_with_feedback(IL_out, model["FCLW"].getPrevOut()[t-1])
                else:
                    FCLU_out = model["FCLU"].forward(IL_out)
                ACT1_out = model["ACT1"].forward(FCLU_out)
                FCLW_out = model["FCLW"].forward(ACT1_out)
                FCLV_out = model["FCLV"].forward(ACT1_out)
                ACT2_out = model["ACT2"].forward(FCLV_out)

            # Predictions and Loss
            y_preds_batch = ACT2_out
            y_preds_batches.append(y_preds_batch)
            # training_dict["y_preds"].append(ACT2_out)

            error_batch = model["SE"].eval(y, ACT2_out.reshape(y[batch].shape))
            # training_dict["squared_error"].append(error_batch)
            error_batches.append(error_batch)

            # Backward
            dhNext_dW = np.zeros((1, model["FCLV"].getWeights().shape[0])) # Same shape as what FCLV.backward(grad) is...
            
            for t in range(len(X[batch])-1, -1, -1):
                grad = model["SE"].gradient(y[batch], (model["ACT2"].getPrevOut()[-1].reshape(y[batch].shape))).reshape(1, -1)
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
        training_dict["squared_error_avg"].append(np.mean(error_batches))

    return pd.DataFrame(training_dict)

'''
Given an input RNN as a dictionary, input data, and output labels, 
predict the output and return the predictions and RMSE.
'''
def predict_batches(model, X_inp, y_inp):
    rmse_batches = []
    y_preds_batches = []

    for batch in range(len(X_inp)):
        model["FCLU"].reset()
        model["FCLW"].reset()
        model["FCLV"].reset()
        model["ACT1"].setPrevIn([])
        model["ACT1"].setPrevOut([])
        model["ACT2"].setPrevIn([])
        model["ACT2"].setPrevOut([])
        
        for t in range(len(X_inp[batch])):
            IL_out = model["IL"].forward(X_inp[batch][t])
            if t > 0:
                FCLU_out = model["FCLU"].forward_with_feedback(IL_out, model["FCLW"].getPrevOut()[t-1])
            else:
                FCLU_out = model["FCLU"].forward(IL_out)
            ACT1_out = model["ACT1"].forward(FCLU_out)
            FCLW_out = model["FCLW"].forward(ACT1_out)
            FCLV_out = model["FCLV"].forward(ACT1_out)
            ACT2_out = model["ACT2"].forward(FCLV_out)


        y_pred_batch = ACT2_out
        # print(y_pred_batch) # Print output
        y_preds_batches.append(y_pred_batch)

        error_batch = model["SE"].eval(y_inp[batch], y_pred_batch.reshape(y_inp[batch].shape))
        rmse_batch = np.sqrt(error_batch)
        rmse_batches.append(rmse_batch)

    return y_preds_batches, rmse_batches

'''
Given a list of years to train on and a dataframe, 
return a list of non-overlapping slices for the validation sets
(all years not involved in training, starting from 1983).
'''
def all_val_slices(years_train_inp, df_inp):

    all_years_train = [year for year in years_train_inp]
    for year in years_train_inp:
        all_years_train.append(year+1)
        all_years_train.append(year+2)

    all_years_train = set(all_years_train)

    all_years = set(range(1983, 2012-1))
    val_years = all_years - all_years_train

    return get_slice_list(list(val_years), df_inp)

'''
Given an RNN model and a list of years used to train:
1) Predict outputs for validation sets
    * predict_batches() on X_val and y_val iterating over slices made from all_val_slices()
2) Calculate RMSEs for each validation set
3) Return a dataframe with columns input_range, predict_range, and rmse
'''
def get_df_val_RMSEs(model, years_train):
    
    val_slices = all_val_slices(years_train, df)
    val_testing_dict = {"input_range": [], 
                        "predict_range": [], 
                        "rmse": []}
    for val_slice in val_slices:
        years_val_input = val_slice["date"].dt.year.unique()[:2]
        years_val_predict = val_slice["date"].dt.year.unique()[2:]
        val_testing_dict["input_range"].append(f"{years_val_input[0]}-{years_val_input[-1]}")
        val_testing_dict["predict_range"].append(f"{years_val_predict[0]}")
    
    X_val, y_val = train_test_split_batches(val_slices, 24)
    
    
    y_pred_val, rmse_val = predict_batches(model, X_val, y_val)
    val_testing_dict["rmse"] = rmse_val

    return pd.DataFrame(val_testing_dict)

###----------------- Load Data -----------------###
df = load_data("./Data/PhiladelphiaLandTemperatures.csv")
years_train = [1980, 1983, 1990, 1993, 2000, 2003]

slices_train = get_slice_list(years_train, df)

X_train, y_train = train_test_split_batches(slices_train, 24)


###----------------- Compare Error vs. Epochs for Tanh and ReLU RNN -----------------###
lr_tanh = 1e-5
lr_relu = 1e-6
num_epochs=800

# Assume scaling by X_train[0] is sufficient for all batches
rnn_tanh_batches = make_RNN(X_inp = X_train[0], fcl_size=20, output_size=y_train[0].shape[0]) # Have to reinitialize
training_results_tanh_batches = train_RNN_inplace_with_batches_for(rnn_tanh_batches, X_train, y_train, learning_rate=lr_tanh, epochs=num_epochs)

rnn_relu_batches = make_RNN(X_inp = X_train[0], fcl_size=20, output_size=y_train[0].shape[0], act_func="relu") # Have to reinitialize
training_results_relu_batches = train_RNN_inplace_with_batches_for(rnn_relu_batches, X_train, y_train, learning_rate=lr_relu, epochs=num_epochs)

df_error_vs_epochs_compare = pd.concat([pd.DataFrame(training_results_tanh_batches).assign(act_func="Tanh (Learning Rate = 1e-5)"),
                                        pd.DataFrame(training_results_relu_batches).assign(act_func="ReLU (Learning Rate = 1e-6)")], axis=0)

act_funcs = df_error_vs_epochs_compare['act_func'].unique()

fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")

markers = ["o", "x"]

for act_func, marker in zip(act_funcs, markers):
    data = df_error_vs_epochs_compare[df_error_vs_epochs_compare['act_func'] == act_func]
    ax.plot(data['epoch'], data['squared_error_avg'], marker=marker, label=act_func)

# Format
ax.set_title('Average Squared Error vs. Epochs for RNN', fontsize=16)
ax.set_xlabel('Epochs', fontsize=14)
ax.set_ylabel('Average Squared Error', fontsize=14)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
ax.grid()
ax.legend(fontsize=12)
plt.show()

###----------------- Train "Optimal" Tanh -----------------###
lr = lr_tanh

# Assume scaling by X_train[0] is sufficient for all batches
rnn_tanh_batches = make_RNN(X_inp = X_train[0], fcl_size=20, output_size=y_train[0].shape[0])
training_results_tanh_batches = train_RNN_inplace_with_batches_for(rnn_tanh_batches, X_train, y_train, learning_rate=lr, epochs=100+1)
y_preds_train_tanh, rmses_train_tanh = predict_batches(rnn_tanh_batches, X_train, y_train)

###----------------- Train "Optimal" ReLU -----------------###
lr = lr_relu

# Assume scaling by X_train[0] is sufficient for all batches
rnn_relu_batches = make_RNN(X_inp = X_train[0], fcl_size=20, output_size=y_train[0].shape[0], act_func="relu")
training_results_relu_batches = train_RNN_inplace_with_batches_for(rnn_relu_batches, X_train, y_train, learning_rate=lr_relu, epochs=700+1)

y_preds_train_relu, rmses_train_relu = predict_batches(rnn_relu_batches, X_train, y_train)

###----------------- Plot Training Predictions and RMSE for "Optimal" Tanh and ReLU -----------------###
ax_coords = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]

fig, axs = plt.subplots(3, 2, figsize=(18, 18), facecolor="white")
for i in range(len(years_train)):
    year = years_train[i]
    train_slice = slices_train[i]
    X_batch = X_train[i]
    y_batch = y_train[i]
    ax_coord = ax_coords[i]
    y_pred_batch_tanh = y_preds_train_tanh[i]
    rmse_batch_tanh = rmses_train_tanh[i]
    y_pred_batch_relu = y_preds_train_relu[i]
    rmse_batch_relu = rmses_train_relu[i]

    t_vals_x = train_slice.iloc[:len(X_batch), :]["date"].dt.strftime('%Y-%m')
    t_vals_y = train_slice.iloc[len(X_batch):(len(X_batch)+len(y_batch)), :]["date"].dt.strftime('%Y-%m')

    axs[ax_coord].plot(t_vals_x, X_batch.flatten(), color="green", linestyle="solid", label="Training Sequence", linewidth=2, marker="o")
    axs[ax_coord].plot(t_vals_y, y_batch.flatten(), color="blue", linestyle="solid", label="Predict Sequence", linewidth=2, marker="o")

    axs[ax_coord].plot(t_vals_y, y_pred_batch_tanh.flatten(), color = "magenta", linestyle="dashed", label=f"Tanh Model Predictions", linewidth=2, marker="x")
    axs[ax_coord].plot(t_vals_y, y_pred_batch_relu.flatten(), color = "orange", linestyle="dashed", label=f"ReLU Model Predictions", linewidth=2, marker="*")


    axs[ax_coord].set_xlabel("Time", fontsize=14)
    axs[ax_coord].set_ylabel("Average Temperature ($^\circ$C)", fontsize=14)
    axs[ax_coord].set_title(f"Predictions for Training Set {year+2}, RMSE$_{{Tanh}}$={rmse_batch_tanh:.2f}, RMSE$_{{ReLU}}$={rmse_batch_relu:.2f}", 
                fontsize=16)
    new_xticks = (train_slice["date"].dt.strftime('%Y-%m')
                  .iloc[list(range(0, len(train_slice), 3))] # 3 month intervals
    )
    axs[ax_coord].set_xticks(new_xticks, labels=new_xticks.astype(str), rotation=60)
    axs[ax_coord].tick_params(axis="x", labelsize=14)
    axs[ax_coord].tick_params(axis="y", labelsize=14)
    axs[ax_coord].legend(fontsize=11, loc="lower left")
    axs[ax_coord].grid()

plt.subplots_adjust(hspace=0.5)
plt.show()

###----------------- Compare RMSEs on Validation Sets-----------------###
