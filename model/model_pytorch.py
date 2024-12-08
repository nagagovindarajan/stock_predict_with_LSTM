# -*- coding: UTF-8 -*-
"""
pytorch model
"""

import torch
from torch.nn import Module, LSTM, Linear
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class Net(Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.lstm = LSTM(
            input_size=config.input_size, 
            hidden_size=config.hidden_size,
            num_layers=config.lstm_layers, 
            batch_first=True, 
            dropout=config.dropout_rate,
            bidirectional=True # Use bidirectional LSTM
        )
        # Since we used bidirectional=True, hidden_size doubles
        self.linear = Linear(in_features=config.hidden_size*2, out_features=config.output_size)

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        linear_out = self.linear(lstm_out)
        return linear_out, hidden


def train(config, logger, train_and_valid_data):
    train_X, train_Y, valid_X, valid_Y = train_and_valid_data
    train_X, train_Y = torch.from_numpy(train_X).float(), torch.from_numpy(train_Y).float()
    train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=config.batch_size)

    valid_X, valid_Y = torch.from_numpy(valid_X).float(), torch.from_numpy(valid_Y).float()
    valid_loader = DataLoader(TensorDataset(valid_X, valid_Y), batch_size=config.batch_size)

    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print("The device is {}".format(device))

    model = Net(config).to(device)
    if config.add_train:
        model.load_state_dict(torch.load(config.model_save_path + config.model_name))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()

    # Learning rate scheduler: reduce LR if no improvement for 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    valid_loss_min = float("inf")
    bad_epoch = 0
    global_step = 0
    for epoch in range(config.epoch):
        logger.info("Epoch {}/{}".format(epoch+1, config.epoch))
        model.train()
        train_loss_array = []
        hidden_train = None
        for i, (_train_X, _train_Y) in enumerate(train_loader):
            _train_X, _train_Y = _train_X.to(device), _train_Y.to(device)
            optimizer.zero_grad()
            pred_Y, hidden_train = model(_train_X, hidden_train)

            if not config.do_continue_train:
                hidden_train = None
            else:
                h_0, c_0 = hidden_train
                h_0.detach_(), c_0.detach_()
                hidden_train = (h_0, c_0)

            loss = criterion(pred_Y, _train_Y)
            loss.backward()
            optimizer.step()
            train_loss_array.append(loss.item())
            global_step += 1

        model.eval()
        valid_loss_array = []
        hidden_valid = None
        with torch.no_grad():
            for _valid_X, _valid_Y in valid_loader:
                _valid_X, _valid_Y = _valid_X.to(device), _valid_Y.to(device)
                pred_Y, hidden_valid = model(_valid_X, hidden_valid)
                if not config.do_continue_train:
                    hidden_valid = None
                val_loss = criterion(pred_Y, _valid_Y)
                valid_loss_array.append(val_loss.item())

        train_loss_cur = np.mean(train_loss_array)
        valid_loss_cur = np.mean(valid_loss_array)
        logger.info("Train loss: {:.6f}, Valid loss: {:.6f}".format(train_loss_cur, valid_loss_cur))

        # Step the scheduler with validation loss
        scheduler.step(valid_loss_cur)

        if valid_loss_cur < valid_loss_min:
            valid_loss_min = valid_loss_cur
            bad_epoch = 0
            torch.save(model.state_dict(), config.model_save_path + config.model_name)
            logger.info("Model improved and saved at epoch {}!".format(epoch+1))
        else:
            bad_epoch += 1
            if bad_epoch >= config.patience:
                logger.info("Early stopping at epoch {} due to no improvement.".format(epoch+1))
                break

def predict(config, test_X):
    # Get test data
    test_X = torch.from_numpy(test_X).float()
    test_set = TensorDataset(test_X)
    test_loader = DataLoader(test_set, batch_size=1)

    # Load model
    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu")
    model = Net(config).to(device)
    model.load_state_dict(torch.load(config.model_save_path + config.model_name))   # Load model parameters

    # First define a tensor to save the prediction results
    result = torch.Tensor().to(device)

    # Prediction process
    model.eval()
    hidden_predict = None
    for _data in test_loader:
        data_X = _data[0].to(device)
        pred_X, hidden_predict = model(data_X, hidden_predict)
        # if not config.do_continue_train: hidden_predict = None    # Experiments show that whether it is continuous training mode or not, passing the hidden of the previous time_step to the next one works better
        cur_pred = torch.squeeze(pred_X, dim=0)
        result = torch.cat((result, cur_pred), dim=0)

    return result.detach().cpu().numpy()    # First remove gradient information, if on gpu, transfer to cpu, finally return numpy data

def forecast_future(config, start_sequence, n_future_steps=10):
    import torch.nn as nn

    if isinstance(start_sequence, np.ndarray):
        start_sequence = torch.from_numpy(start_sequence).float()

    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu")
    model = Net(config).to(device)
    model.load_state_dict(torch.load(config.model_save_path + config.model_name))
    model.eval()

    start_sequence = start_sequence.to(device)
    hidden_future = None
    future_preds = []
    current_input = start_sequence

    # If input_size and output_size differ, define a mapping layer
    if config.input_size != config.output_size:
        map_output_to_input = nn.Linear(config.output_size, config.input_size).to(device)

    for step in range(n_future_steps):
        pred_output, hidden_future = model(current_input, hidden_future)
        # pred_output: [batch_size=1, seq_len, output_size]
        next_pred = pred_output[:, -1:, :] # take the last time step predicted

        # If output_size != input_size, map output to input dimension
        if config.input_size != config.output_size:
            # Convert from [1, 1, output_size] -> [1, output_size]
            next_pred_converted = next_pred.squeeze(1) 
            # Map to input size
            next_pred_converted = map_output_to_input(next_pred_converted) 
            # Reshape back to [1, 1, input_size]
            next_pred_converted = next_pred_converted.unsqueeze(1) 
            # Now concatenate along time dimension
            current_input = torch.cat([current_input[:, 1:, :], next_pred_converted], dim=1)
        else:
            current_input = torch.cat([current_input[:, 1:, :], next_pred], dim=1)

        future_preds.append(next_pred.detach().cpu().numpy())

    future_preds = np.concatenate(future_preds, axis=1) # [1, n_future_steps, output_size]
    return future_preds[0]