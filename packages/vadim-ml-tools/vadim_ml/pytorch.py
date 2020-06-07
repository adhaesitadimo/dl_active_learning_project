import torch
import torch.optim
import torch.nn.functional as F
from torch import nn
import tqdm
import numpy as np
import itertools
from sklearn.metrics import cohen_kappa_score
from vadim_ml.batch import batch_n
import time

def torchify(data, device=torch.device('cpu'), dtype=None):
    return torch.as_tensor(data, dtype=dtype, device=device)

def untorchify(data):
    try:
        return data.cpu().data.numpy()
    except AttributeError:
        return data

def multi_layer_perceptron(input_size, output_size, hidden_layers, activation=nn.Sigmoid, last_activation=True):
    sizes = np.linspace(input_size, output_size, hidden_layers+2)
    sizes = sizes.round().astype(int)
    input_sizes = sizes[:-1]
    output_sizes = sizes[1:]
    
    layers = []
    for input_size, output_size in zip(input_sizes, output_sizes):
        layers.append(nn.Linear(input_size, output_size))
        layers.append(activation())

    if not last_activation:
        del layers[-1]

    return nn.Sequential(*layers)

master_schedule = {
    'min_ema_improvement': 0,
    'ema_coeff': 0.8,
    'min_epochs': 10,
    'max_epochs': 10000,
    'max_seconds': 60 * 60 * 24,
    'learning_rate': 1e-3
}

class AbsoluteMaxPool(nn.Module):
    def __init__(self, output_dims, squeeze_dims):
        super().__init__()
        self.output_dims = output_dims
        self.squeeze_dims = squeeze_dims

    def forward(self, x):
        dims_to_squeeze = [- 1 - dim - output_dims for dim in range(self.squeeze_dims)]

        for max_dim in dims_to_squeeze:
            x = x.max(dim=-max_dim, keepdim=True)[0]
        for max_dim in dims_to_squeeze:
            x = x.squeeze(dim=-max_dim)
        return x.squeeze()

def convolution_layers(input_size, output_size, kernel_sizes, conv_dim=2, pad=False):
    Conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[conv_dim]
    layers = []

    sizes = np.linspace(input_size, output_size, len(kernel_sizes)+1)
    sizes = sizes.round().astype(int)
    input_sizes = sizes[:-1]
    output_sizes = sizes[1:]

    for ksize, isize, osize in zip(kernel_sizes, input_sizes, output_sizes):
        if pad:
            if type(ksize) == tuple:
                pad_size = tuple(int(ks / 2) for ks in ksize)
            else:
                pad_size = int(ksize / 2)
        else:
            pad_size = 0
        
        layers.append(Conv(isize, osize, ksize, padding=pad_size))
        layers.append(nn.LeakyReLU())

    del layers[-1]

    return nn.Sequential(*layers)

def run_schedule(update_model, validate_model, schedule=master_schedule, log=None):
    if not log:
        log = lambda x,y,z: None

    start_quality = validate_model()
    loss_history = []
    quality_history = [start_quality]
    ema_history = [start_quality]

    passive_agressive_comment = f'How do you expect a schedule with minimum {schedule["min_epochs"]} epochs and maximum {schedule["max_epochs"]} epochs to work?'
    assert schedule['min_epochs'] <= schedule['max_epochs'], passive_agressive_comment

    start_time = time.monotonic()

    for epoch in range(schedule['max_epochs']):
        # EMA (Exponential Moving Average) for early stopping
        if epoch >= schedule['min_epochs']:
            if ema_history[-1] != ema_history[0] and ema_history[-1] - ema_history[-2] < schedule['min_ema_improvement']:
                break
        if time.monotonic() - start_time > schedule['max_seconds']:
            break

        loss = update_model()
        quality = validate_model()

        loss_history.append(loss)
        quality_history.append(quality)
        ema_history.append(ema_history[-1] * schedule['ema_coeff'] + (1 - schedule['ema_coeff']) * quality)

        log(loss, quality, ema_history[-1])
        

    return loss_history, quality_history

identity = lambda x: x

class PytorchModel():
    def __init__(self, module, 
                 pred_f=identity,
                 target_f=identity,
                 loss_f=nn.MSELoss(), 
                 quality_f=lambda x, y: np.mean((np.array(list(x)) - (np.array(list(x)))) ** 2), 
                 input_type=torch.float,
                 batch = 100, 
                 optimizer=torch.optim.Adam):
        if type(batch) == int:
            batch_size = batch
            batch = lambda x: batch_n(x, batch_size=batch_size)
        self.batch = batch
        self.module = module.to('cpu')
        self.optimizer = optimizer
        self.loss_f = loss_f
        self.quality_f = quality_f
        self.pred_f = pred_f
        self.target_f = target_f
        self.input_type = input_type
        self.schedule = {**master_schedule}

    def update_model(self, X_train, y_train, opt, device):
        self.module.train() # enable dropout / batch_norm training behavior

        epoch_loss = 0

        permutation = np.random.permutation(len(X_train))

        for idx in permutation:
            X = torchify(X_train[idx], device, self.input_type)
            y = torchify(self.target_f(y_train[idx]), device)
            
            pred = self.module(X)
            loss = self.loss_f(pred, y)
            epoch_loss += loss.cpu().data.numpy()

            loss.mean().backward() # loss vectors are supported

            opt.step()
            opt.zero_grad()

        return epoch_loss

    def predict_all(self, batches, device):
        self.module.eval()
        
        for X in batches:
            X = torchify(X, device, self.input_type)
            y = self.module(X)
            y = untorchify(y)
            y = self.pred_f(y)
            yield y

    def predict(self, X, batched=False, device='cpu'):
        if not batched:
            X = self.batch(X)
        
        y = self.predict_all(X, device)

        if not batched:
            y = np.concatenate(list(y), axis=0)

        return y

    def test_model(self, X_test, y_test, device):
        y_pred = itertools.chain(*self.predict_all(X_test, device))
        y_test = itertools.chain(*y_test)

        return self.quality_f(y_pred, y_test)
        
    def fit(self, X_train, y_train, X_valid, y_valid, batched=False, device='cpu', log=None):
        if not batched:
            X_train, y_train, X_valid, y_valid = (list(self.batch(a)) for a in (X_train, y_train, X_valid, y_valid))
        self.module = self.module.to(device)
        opt = self.optimizer(self.module.parameters(), lr=self.schedule['learning_rate'])
        upd = lambda: self.update_model(X_train, y_train, opt, device)
        validate = lambda: self.test_model(X_valid, y_valid, device)
        ret = run_schedule(upd, validate, schedule=self.schedule, log=log)
        self.module = self.module.to('cpu')
        if device != 'cpu':
            torch.cuda.empty_cache()
        return ret

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state):
        return self.module.load_state_dict(dict(state))

class PytorchClassifier(PytorchModel):
    def __init__(self, module, outputs='scores', batch=100, optimizer=torch.optim.Adam, input_type=torch.float):
        loss_functions = {
            'scores': nn.CrossEntropyLoss(),
            'logprobabilities': nn.NLLLoss(),
            'probabilities': lambda y, target: - (y * F.one_hot(target).to(torch.float)).mean()
        }
        loss_f = loss_functions[outputs]

        pred_f = lambda y: [self.id_to_class[i] for i in np.argmax(y, axis=-1)]
        target_f = np.vectorize(lambda target: self.class_to_id[target])
        quality_f = lambda y1,y2: cohen_kappa_score(list(y1), list(y2))
        super().__init__(module, pred_f=pred_f, target_f=target_f, loss_f=loss_f, quality_f=quality_f, batch=batch, optimizer=optimizer, input_type=input_type)

    def fit(self, X_train, y_train, X_valid, y_valid, batched=False, device='cpu', log=None):
        self.class_to_id = {}
        self.id_to_class = []

        if batched:
            all_ys = itertools.chain(*y_train)
        else:
            all_ys = y_train

        for y in all_ys:
            if y not in self.class_to_id:
                self.class_to_id[y] = len(self.id_to_class)
                self.id_to_class.append(y)

        return super().fit(X_train, y_train, X_valid, y_valid, batched, device, log)

    def state_dict(self):
        return {
            'params': super().state_dict(),
            'classes': self.id_to_class
        }

    def load_state_dict(self, state):
        self.id_to_class = state['classes']
        self.class_to_id = {c: idx for idx, c in enumerate(self.id_to_class)}
        super().load_state_dict(state['params'])

def PytorchBinaryClassifier(module, outputs='scores', batch = 100, optimizer=torch.optim.Adam, input_type=torch.float):
    loss_functions = {
        'scores': lambda y, target: F.binary_cross_entropy(F.sigmoid(y), target.to(torch.float)),
        'probabilities': lambda y, target: F.binary_cross_entropy(y.to(torch.float), target.to(torch.float)),
        'logprobabilities': lambda y, target: F.binary_cross_entropy_with_logits(y.to(torch.float), target.to(torch.float))
    }
    loss_f = loss_functions[outputs]
    
    pred_functions = {
        'scores': lambda y: (y > 0).astype(int).reshape(-1),
        'probabilities': lambda y: (y > 0.5).astype(int).reshape(-1),
        'logprobabilities': lambda y: (y > np.log(0.5)).astype(int).reshape(-1)
    }

    pred_f = pred_functions[outputs]
    quality_f = lambda y1,y2: cohen_kappa_score(list(y1), list(y2))
    return PytorchModel(module, pred_f=pred_f, loss_f=loss_f, quality_f=quality_f, batch=batch, optimizer=optimizer, input_type=input_type)

class OneArgumentModule(nn.Module):
    """An adapter for pytorch modules that want several arguments"""

    def __init__(self, module):
        super().__init__()
        self.inner_module = module

    def forward(self, input):
        return self.inner_module(*input)