from . import double_lstm
from . import simple_lstm
from . import simple_lstm_cnn


models_architectures = {
    'double_lstm': double_lstm.DoubleLSTM,
    'simple_lstm': simple_lstm.SimpleLSTM,
    'simple_lstm_cnn': simple_lstm_cnn.SimpleLSTMCnn,
}
