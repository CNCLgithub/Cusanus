# Several chunks cribbed from
# https://github.com/lucidrains/siren-pytorch/blob/master/siren_pytorch/siren_pytorch.py
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


class Sine(eqx.Module):

    w0: float

    def __init__(self, w0 = 1.):
        self.w0 = w0

    def __call__(self, x):
        return jnp.sin(self.w0 * x)

class Siren(eqx.Module):
    linear: eqx.nn.Linear
    bias: Array

    def __init__(self, dim_in:int, dim_out:int,
                 w0:float = 1., w_std:float = 1.,
                 bias = True, activation = True):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias = bias)
        # REVIEW
        # Initialization is important
        nn.init.uniform_(self.linear.weight, a = -w_std, b = w_std)
        if bias:
            nn.init.uniform_(self.linear.bias, a = -w_std, b = w_std)
        # REVIEW: get rid of conditional?
        self.activation = Sine(w0) if activation else nn.Identity()

    def __call__(self, x):
        x =  self.linear(x)
        y = Sine(self.sine_weight)(x)
        return y


class SirenNet(eqx.Module):
    '''SirenNet model.

    Args:
        in_features: Input size.
        hidden_features: Hidden layer size.
        out_features: Output layer size.
        num_layers: Number of layers.
        sine_weight: Sine activation weight for hidden layers.
        sine_weight_initial: Sine activation weight for first layer.
        c:
        use_bias: Flag for using biases or not.
        final_activation: Activation function of final layer.
    '''
    in_features: int
    hidden_features: int
    out_features: int
    num_layers: int
    layers: list
    last_layer: eqx.Module
    final_activation: callable

    def __init__(self,
                 key: jax.random.key,
                 in_features: int,
                 hidden_features: int,
                 out_features: int,
                 num_layers: int,
                 sine_weight: float = 1.0,
                 sine_weight_initial: float = 15.0,
                 c: float = 6.0,
                 use_bias: bool = True,
                 final_activation = jax.nn.sigmoid) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.final_activation = final_activation
        w_std = jnp.sqrt(c / hidden_features) / sine_weight
        layers = []
        for l in range(num_layers-1):
            key, _key = jax.random.split(key)
            layers.append(
                Siren(
                    key=_key,
                    in_features=in_features if l == 0 else hidden_features,
                    out_features=hidden_features,
                    sine_weight=sine_weight_initial if l == 0 else sine_weight,
                    w_std=1.0 / in_features if l == 0 else w_std,
                    use_bias=use_bias,
                )
            )
        self.layers = layers
        key, _key = jax.random.split(key)
        self.last_layer = eqx.nn.Linear(
            key=_key, in_features=hidden_features, out_features=out_features)

    def __call__(self, x: Array):
        '''Forward pass through the model.
        
        Args:
            x: Model input.
        
        Returns:
            y: Model output.
        '''
        for l in range(self.num_layers - 1):
            x = self.layers[l](x)
        y = self.last_layer(x)
        y = self.final_activation(y)
        return y


class ModulatedSirenNet(SirenNet):
    '''Class for modulated SirenNet.
    
    Wraps around SirenNet class and overrides the forward method. Note this
    changes arity of from 1 to 2:
    SirenNet.forward() -> ModulatedSirenNet.forward(x, phi).
    '''

    # pylint: disable=arguments-differ
    def __call__(self, x:Array, phi:Array):
        '''Forward pass through the model.

        Note:
            The overwritten forward method changes the arity of
            SirenNet.forward(x) from 1 to 2 in
            ModulateSirenNet.forward(x, phi).
        
        Args:
            x: Model input.
            phi: Latent modulations.

        Returns:
            y: Model output.
        '''
        for l in range(self.num_layers - 1):
            phi_ = phi[l]
            x = self.layers[l](x) + phi_
        y = self.last_layer(x)
        y = self.final_activation(y)
        return y
    # pylint: enable=arguments-differ


class LatentModulation(eqx.Module):
    '''Latent modulations for ModulatedSirenNets.
    
        Args:
            shape: The shape of the modulation.
            device: The device environment.
    '''
    latent_code: Array

    def __init__(self, shape: tuple[int, ...] | int, device):
        super().__init__()
        data = jnp.zeros(shape=shape, device=device)
        # Equinox will treat this as a parameter
        self.latent_code = data


class Modulator(eqx.Module):
    '''Class for MLP-based modulations.
    
    Args:
        key: jax.random.key(...)
        in_features: input dimensions
        out_features: output dimensions
        num_layers: number of hidden layers
    '''
    num_layers: int
    hidden_features: int
    layers: list
    in_features: int

    def __init__(
            self,
            key: jax.random.key,
            in_features: int,
            hidden_features:int,
            num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.layers = []
        # Add hidden layers
        for ind in range(num_layers):
            is_first = ind == 0
            # for skip connection
            dim = in_features if is_first else (hidden_features + in_features)
            key, _key = jax.random.split(key)
            layer = eqx.nn.Linear(
                key=_key,
                in_features=dim,
                out_features=hidden_features)
            self.layers.append(layer)

    def __call__(self, x):
        '''Forward pass through the model.
        
        Args:
            x: modulator input
        '''
        hiddens = []
        # set as first input
        theta = x
        for l in range(self.num_layers - 1):
            # pass through next layer in modulator
            y = self.layers[l](x)
            y = jax.nn.relu(y)
            # save layer output
            hiddens.append(y)
            # concat with latent code for next step
            # NOTE: Does not change theta size after first loop
            x = jnp.concatenate([y, theta], axis=-1)
        y = self.layers[-1](x)
        y = jax.nn.relu(y)
        hiddens.append(y)
        return hiddens


class ImplicitNeuralModule(eqx.Module):
    '''Implicit Neural Module

    Args:
        theta: Siren network
        psi: Modulation FC network
    '''
    hidden_features: int
    mod: int
    theta: ModulatedSirenNet
    psi: Modulator

    def __init__(self, theta: ModulatedSirenNet, psi: Modulator) -> None:
        super().__init__()
        # Siren Network - weights refered to as `theta`
        # optimized during outer loop
        self.theta = theta
        # Modulation FC network - refered to as psi
        # psi is initialize with default weights
        # and is not optimized
        self.psi = psi
        self.hidden_features = theta.hidden_features
        self.mod = psi.in_features
        assert psi.hidden_features == theta.hidden_features

    def __call__(self, qs:Array, m: Array) -> Array:
        '''Forward model call.

        Args:
            qs: Model (sirenet) input.
            m: Modulator input.
        '''
        phi = self.psi(m) # shift modulations
        return self.theta(qs, phi)
