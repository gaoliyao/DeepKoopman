import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import helperfns_torch

def weight_variable(shape, var_name, distribution='tn', scale=0.1):
    if distribution == 'tn':
        return nn.Parameter(torch.randn(shape) * scale)
    elif distribution == 'xavier':
        scale = 4 * np.sqrt(6.0 / (shape[0] + shape[1]))
        return nn.Parameter(torch.rand(shape) * 2 * scale - scale)
    elif distribution == 'dl':
        scale = 1.0 / np.sqrt(shape[0])
        return nn.Parameter(torch.rand(shape) * 2 * scale - scale)
    elif distribution == 'he':
        scale = np.sqrt(2.0 / shape[0])
        return nn.Parameter(torch.randn(shape) * scale)
    elif distribution == 'glorot_bengio':
        scale = np.sqrt(6.0 / (shape[0] + shape[1]))
        return nn.Parameter(torch.rand(shape) * 2 * scale - scale)
    else:
        initial = np.loadtxt(distribution, delimiter=',', dtype=np.float64)
        if (initial.shape[0] != shape[0]) or (initial.shape[1] != shape[1]):
            raise ValueError(
                f'Initialization for {var_name} is not correct shape. Expecting {shape}, but find {initial.shape} in {distribution}.')
        return nn.Parameter(torch.tensor(initial))

def bias_variable(shape, var_name, distribution=''):
    if distribution:
        initial = np.genfromtxt(distribution, delimiter=',', dtype=np.float64)
        return nn.Parameter(torch.tensor(initial))
    else:
        return nn.Parameter(torch.zeros(shape))

class Encoder(nn.Module):
    def __init__(self, widths, dist_weights, dist_biases, scale, num_shifts_max):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(widths) - 1):
            self.layers.append(nn.Linear(widths[i], widths[i+1]))
            self.layers[-1].weight = weight_variable([widths[i], widths[i+1]], f'WE{i+1}', dist_weights[i], scale)
            self.layers[-1].bias = bias_variable([widths[i+1]], f'bE{i+1}', dist_biases[i])

    def forward(self, x, act_type, shifts_middle):
        y = []
        num_shifts_middle = len(shifts_middle)
        for j in range(num_shifts_middle + 1):
            if j == 0:
                shift = 0
            else:
                shift = shifts_middle[j - 1]
            if isinstance(x, list):
                x_shift = x[shift]
            else:
                x_shift = x[shift, :, :]
            y.append(self.forward_one_shift(x_shift, act_type))
        return y

    def forward_one_shift(self, x, act_type):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if act_type == 'sigmoid':
                x = torch.sigmoid(x)
            elif act_type == 'relu':
                x = torch.relu(x)
            elif act_type == 'elu':
                x = torch.elu(x)
        return self.layers[-1](x)

class Decoder(nn.Module):
    def __init__(self, widths, dist_weights, dist_biases, scale):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(widths) - 1):
            self.layers.append(nn.Linear(widths[i], widths[i+1]))
            self.layers[-1].weight = weight_variable([widths[i], widths[i+1]], f'WD{i+1}', dist_weights[i], scale)
            self.layers[-1].bias = bias_variable([widths[i+1]], f'bD{i+1}', dist_biases[i])

    def forward(self, x, act_type):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if act_type == 'sigmoid':
                x = torch.sigmoid(x)
            elif act_type == 'relu':
                x = torch.relu(x)
            elif act_type == 'elu':
                x = torch.elu(x)
        return self.layers[-1](x)

def form_complex_conjugate_block(omegas, delta_t):
    scale = torch.exp(omegas[:, 1] * delta_t)
    entry11 = scale * torch.cos(omegas[:, 0] * delta_t)
    entry12 = scale * torch.sin(omegas[:, 0] * delta_t)
    row1 = torch.stack([entry11, -entry12], dim=1)
    row2 = torch.stack([entry12, entry11], dim=1)
    return torch.stack([row1, row2], dim=2)

def varying_multiply(y, omegas, delta_t, num_real, num_complex_pairs):
    complex_list = []
    for j in range(num_complex_pairs):
        ind = 2 * j
        ystack = torch.stack([y[:, ind:ind + 2], y[:, ind:ind + 2]], dim=2)
        L_stack = form_complex_conjugate_block(omegas[j], delta_t)
        elmtwise_prod = ystack * L_stack
        complex_list.append(torch.sum(elmtwise_prod, 1))

    if complex_list:
        complex_part = torch.cat(complex_list, dim=1)

    real_list = []
    for j in range(num_real):
        ind = 2 * num_complex_pairs + j
        temp = y[:, ind]
        real_list.append(temp.unsqueeze(1) * torch.exp(omegas[num_complex_pairs + j] * delta_t))

    if real_list:
        real_part = torch.cat(real_list, dim=1)

    if complex_list and real_list:
        return torch.cat([complex_part, real_part], dim=1)
    elif complex_list:
        return complex_part
    else:
        return real_part

class OmegaNet(nn.Module):
    def __init__(self, params):
        super(OmegaNet, self).__init__()
        self.params = params
        self.complex_nets = nn.ModuleList([self._create_one_omega_net('OC', i) for i in range(params['num_complex_pairs'])])
        self.real_nets = nn.ModuleList([self._create_one_omega_net('OR', i) for i in range(params['num_real'])])

    def _create_one_omega_net(self, net_type, index):
        if net_type == 'OC':
            widths = self.params['widths_omega_complex']
        else:
            widths = self.params['widths_omega_real']
        
        layers = []
        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i+1]))
            layers[-1].weight = weight_variable([widths[i], widths[i+1]], f'{net_type}{index+1}_W{i+1}', 
                                                self.params['dist_weights_omega'][i], self.params['scale_omega'])
            layers[-1].bias = bias_variable([widths[i+1]], f'{net_type}{index+1}_b{i+1}', 
                                            self.params['dist_biases_omega'][i])
        return nn.Sequential(*layers)

    def forward(self, ycoords):
        omegas = []
        for j, net in enumerate(self.complex_nets):
            ind = 2 * j
            pair_of_columns = ycoords[:, ind:ind + 2]
            radius_of_pair = torch.sum(torch.square(pair_of_columns), dim=1, keepdim=True)
            omegas.append(net(radius_of_pair))
        
        for j, net in enumerate(self.real_nets):
            ind = 2 * self.params['num_complex_pairs'] + j
            one_column = ycoords[:, ind].unsqueeze(1)
            omegas.append(net(one_column))
        
        return omegas

class KoopmanNet(nn.Module):
    def __init__(self, params):
        super(KoopmanNet, self).__init__()
        self.params = params
        depth = int((params['d'] - 4) / 2)
        
        encoder_widths = params['widths'][0:depth + 2]
        self.encoder = Encoder(encoder_widths, params['dist_weights'][0:depth + 1],
                               params['dist_biases'][0:depth + 1], params['scale'],
                               helperfns_torch.num_shifts_in_stack(params))
        
        self.omega_net = OmegaNet(params)
        
        decoder_widths = params['widths'][depth + 2:]
        self.decoder = Decoder(decoder_widths, params['dist_weights'][depth + 2:],
                               params['dist_biases'][depth + 2:], params['scale'])

    def forward(self, x):
        g_list = self.encoder(x, self.params['act_type'], self.params['shifts_middle'])
        
        y = []
        encoded_layer = g_list[0]
        y.append(self.decoder(encoded_layer, self.params['act_type']))
        
        omegas = self.omega_net(encoded_layer)
        advanced_layer = varying_multiply(encoded_layer, omegas, self.params['delta_t'],
                                          self.params['num_real'], self.params['num_complex_pairs'])
        
        for j in range(max(self.params['shifts'])):
            if (j + 1) in self.params['shifts']:
                y.append(self.decoder(advanced_layer, self.params['act_type']))
            
            omegas = self.omega_net(advanced_layer)
            advanced_layer = varying_multiply(advanced_layer, omegas, self.params['delta_t'],
                                              self.params['num_real'], self.params['num_complex_pairs'])
        
        if len(y) != (len(self.params['shifts']) + 1):
            raise ValueError('length(y) not proper length: check create_koopman_net code and how defined params[shifts] in experiment')
        
        return x, y, g_list

def create_koopman_net(params):
    return KoopmanNet(params)