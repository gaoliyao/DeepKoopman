import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import helperfns_torch
import networkarch_torch as net

def define_loss(x, y, g_list, model, params):
    """Define the (unregularized) loss functions for the training."""
    denominator_nonzero = 1e-5

    # autoencoder loss
    if params['relative_loss']:
        loss1_denominator = torch.mean(torch.mean(torch.square(x[0, :, :]), 1)) + denominator_nonzero
    else:
        loss1_denominator = 1.0

    mean_squared_error = torch.mean(torch.mean(torch.square(y[0] - x[0, :, :]), 1))
    loss1 = params['recon_lam'] * (mean_squared_error / loss1_denominator)

    # dynamics/prediction loss
    loss2 = torch.zeros(1, device=x.device)
    if params['num_shifts'] > 0:
        for j in range(params['num_shifts']):
            shift = params['shifts'][j]
            if params['relative_loss']:
                loss2_denominator = torch.mean(torch.mean(torch.square(x[shift, :, :]), 1)) + denominator_nonzero
            else:
                loss2_denominator = 1.0
            loss2 += params['recon_lam'] * (torch.mean(torch.mean(torch.square(y[j + 1] - x[shift, :, :]), 1)) / loss2_denominator)
        loss2 /= params['num_shifts']

    # K linear loss
    loss3 = torch.zeros(1, device=x.device)
    count_shifts_middle = 0
    if params['num_shifts_middle'] > 0:
        omegas = model.omega_net(g_list[0])
        next_step = net.varying_multiply(g_list[0], omegas, params['delta_t'], params['num_real'], params['num_complex_pairs'])
        for j in range(max(params['shifts_middle'])):
            if (j + 1) in params['shifts_middle']:
                if params['relative_loss']:
                    loss3_denominator = torch.mean(torch.mean(torch.square(g_list[count_shifts_middle + 1]), 1)) + denominator_nonzero
                else:
                    loss3_denominator = 1.0
                loss3 += params['mid_shift_lam'] * (torch.mean(torch.mean(torch.square(next_step - g_list[count_shifts_middle + 1]), 1)) / loss3_denominator)
                count_shifts_middle += 1
            omegas = model.omega_net(next_step)
            next_step = net.varying_multiply(next_step, omegas, params['delta_t'], params['num_real'], params['num_complex_pairs'])
        loss3 /= params['num_shifts_middle']

    # inf norm on autoencoder error and one prediction step
    if params['relative_loss']:
        Linf1_den = torch.norm(torch.norm(x[0, :, :], dim=1, p=float('inf')), p=float('inf')) + denominator_nonzero
        Linf2_den = torch.norm(torch.norm(x[1, :, :], dim=1, p=float('inf')), p=float('inf')) + denominator_nonzero
    else:
        Linf1_den = 1.0
        Linf2_den = 1.0

    Linf1_penalty = torch.norm(torch.norm(y[0] - x[0, :, :], dim=1, p=float('inf')), p=float('inf')) / Linf1_den
    Linf2_penalty = torch.norm(torch.norm(y[1] - x[1, :, :], dim=1, p=float('inf')), p=float('inf')) / Linf2_den
    loss_Linf = params['Linf_lam'] * (Linf1_penalty + Linf2_penalty)

    loss = loss1 + loss2 + loss3 + loss_Linf

    return loss1, loss2, loss3, loss_Linf, loss

def define_regularization(params, model, loss, loss1):
    """Define the regularization and add to loss."""
    if params['L1_lam']:
        loss_L1 = params['L1_lam'] * sum(p.abs().sum() for p in model.parameters())
    else:
        loss_L1 = torch.zeros(1, device=next(model.parameters()).device)

    loss_L2 = params['L2_lam'] * sum(p.pow(2.0).sum() for p in model.parameters() if 'bias' not in p.name())

    regularized_loss = loss + loss_L1 + loss_L2
    regularized_loss1 = loss1 + loss_L1 + loss_L2

    return loss_L1, loss_L2, regularized_loss, regularized_loss1

def try_net(data_val, params):
    """Run a random experiment for particular params and data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = net.create_koopman_net(params).to(device)
    
    max_shifts_to_stack = helperfns_torch.num_shifts_in_stack(params)

    optimizer = helperfns_torch.choose_optimizer(params, model)
    optimizer_autoencoder = helperfns_torch.choose_optimizer(params, model)

    csv_path = params['model_path'].replace('model', 'error').replace('pth', 'csv')
    print(csv_path)

    num_saved_per_file_pass = params['num_steps_per_file_pass'] // 20 + 1
    num_saved = int(np.floor(num_saved_per_file_pass * params['data_train_len'] * params['num_passes_per_file']))
    train_val_error = np.zeros([num_saved, 16])
    count = 0
    best_error = float('inf')

    data_val_tensor = torch.tensor(helperfns_torch.stack_data(data_val, max_shifts_to_stack, params['len_time']), dtype=params['dtype'], device=device)

    start = time.time()
    finished = False

    torch.save(model.state_dict(), params['model_path'])

    # TRAINING
    for f in range(params['data_train_len'] * params['num_passes_per_file']):
        if finished:
            break
        file_num = (f % params['data_train_len']) + 1

        if (params['data_train_len'] > 1) or (f == 0):
            data_train = np.loadtxt(f'./data/{params["data_name"]}_train{file_num}_x.csv', delimiter=',', dtype=np.float64)
            data_train_tensor = torch.tensor(helperfns_torch.stack_data(data_train, max_shifts_to_stack, params['len_time']), dtype=params['dtype'], device=device)
            num_examples = data_train_tensor.shape[1]
            num_batches = num_examples // params['batch_size']

        ind = torch.randperm(num_examples)
        data_train_tensor = data_train_tensor[:, ind, :]

        for step in range(params['num_steps_per_batch'] * num_batches):
            if params['batch_size'] < data_train_tensor.shape[1]:
                offset = (step * params['batch_size']) % (num_examples - params['batch_size'])
            else:
                offset = 0

            batch_data_train = data_train_tensor[:, offset:(offset + params['batch_size']), :]

            model.train()
            if (not params['been5min']) and params['auto_first']:
                optimizer_autoencoder.zero_grad()
                x, y, g_list = model(batch_data_train)
                loss1, _, _, _, _ = define_loss(batch_data_train, y, g_list, model, params)
                _, _, regularized_loss1, _ = define_regularization(params, model, loss1, loss1)
                regularized_loss1.backward()
                optimizer_autoencoder.step()
            else:
                optimizer.zero_grad()
                x, y, g_list = model(batch_data_train)
                loss1, loss2, loss3, loss_Linf, loss = define_loss(batch_data_train, y, g_list, model, params)
                loss_L1, loss_L2, regularized_loss, regularized_loss1 = define_regularization(params, model, loss, loss1)
                regularized_loss.backward()
                optimizer.step()

            if step % 20 == 0:
                model.eval()
                with torch.no_grad():
                    x, y, g_list = model(batch_data_train)
                    train_loss1, train_loss2, train_loss3, train_loss_Linf, train_loss = define_loss(batch_data_train, y, g_list, model, params)
                    train_loss_L1, train_loss_L2, train_regularized_loss, train_regularized_loss1 = define_regularization(params, model, train_loss, train_loss1)

                    x, y, g_list = model(data_val_tensor)
                    val_loss1, val_loss2, val_loss3, val_loss_Linf, val_loss = define_loss(data_val_tensor, y, g_list, model, params)
                    val_loss_L1, val_loss_L2, val_regularized_loss, val_regularized_loss1 = define_regularization(params, model, val_loss, val_loss1)

                if val_loss < (best_error - best_error * 1e-5):
                    best_error = val_loss.item()
                    torch.save(model.state_dict(), params['model_path'])
                    print(f"New best val error {best_error} (with reg. train err {train_regularized_loss.item()} and reg. val err {val_regularized_loss.item()})")

                train_val_error[count] = [
                    train_loss.item(), val_loss.item(),
                    train_regularized_loss.item(), val_regularized_loss.item(),
                    train_loss1.item(), val_loss1.item(),
                    train_loss2.item(), val_loss2.item(),
                    train_loss3.item(), val_loss3.item(),
                    train_loss_Linf.item(), val_loss_Linf.item(),
                    train_loss_L1.item(), val_loss_L1.item(),
                    train_loss_L2.item(), val_loss_L2.item()
                ]

                np.savetxt(csv_path, train_val_error, delimiter=',')
                finished, save_now = helperfns_torch.check_progress(start, best_error, params)
                count += 1
                if save_now:
                    train_val_error_trunc = train_val_error[:count, :]
                    helperfns_torch.save_files(model, csv_path, train_val_error_trunc, params)
                if finished:
                    break

            if step > params['num_steps_per_file_pass']:
                params['stop_condition'] = 'reached num_steps_per_file_pass'
                break

    # SAVE RESULTS
    train_val_error = train_val_error[:count, :]
    print(train_val_error)
    params['time_exp'] = time.time() - start
    model.load_state_dict(torch.load(params['model_path']))
    helperfns_torch.save_files(model, csv_path, train_val_error, params)

def main_exp(params):
    """Set up and run one random experiment."""
    helperfns_torch.set_defaults(params)

    if not os.path.exists(params['folder_name']):
        os.makedirs(params['folder_name'])

    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    
    data_val = np.loadtxt(f'./data/{params["data_name"]}_val_x.csv', delimiter=',', dtype=np.float64)
    try_net(data_val, params)

if __name__ == "__main__":
    # Set your parameters here
    params = {
        'data_name': 'your_data_name',
        'folder_name': 'results',
        # Add other necessary parameters
    }
    main_exp(params)