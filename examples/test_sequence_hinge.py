from discrete_skip_gram.sequence_model import hinge_targets, SequenceModel
from discrete_skip_gram.lstm import LSTM
import theano
import theano.tensor as T
import numpy as np
from keras.optimizers import RMSprop


def main():
    x = [[1, 2, 3, 4],
         [5, 2, 1, 0],
         [3, 2, 5, 0],
         [2, 1, 0, 0]]
    z = [[1, 1, 0],
         [2, 1, 0],
         [1, 0, 0],
         [1, 2, 1]]
    x = np.array(x).astype(np.int32)
    z = np.array(z).astype(np.int32)
    d = np.concatenate((x, z), axis=1)

    lstm = LSTM("x_lstm", k=5, hidden_dim=64)
    seq = SequenceModel("z_sequence", k=3, depth = 3, hidden_dim=128, latent_dim=64)
    x_input = T.imatrix("x_input")
    z_input = T.imatrix("z_input")
    pred = seq.hinge(z_input, lstm.call(x_input))
    targets = hinge_targets(z_input, k=4)
    loss = T.mean(T.nnet.relu(-targets * pred + 1), axis=None)
    opt = RMSprop(1e-3)
    params = lstm.params + seq.params
    updates = opt.get_updates(params, {}, loss)
    train_f = theano.function([x_input, z_input], [], updates=updates)

    zpred = seq.policy_hinge(lstm.call(x_input))
    predict_f = theano.function([x_input], zpred)
    loss_f = theano.function([x_input, z_input], [loss])

    for epoch in range(32):
        np.random.shuffle(d)
        _x = d[:, :x.shape[1]]
        _z = d[:, x.shape[1]:]
        print "Epoch: {}".format(epoch)
        _z_pred = predict_f(_x)
        for i in range(_x.shape[0]):
            _zp = predict_f(_x[i:i+1,:])
            print "{} -> {}/{} (target: {})".format(_x[i,:], _zp[0,:], _z_pred[i,:], _z[i,:])
        _loss = loss_f(_x, _z)
        print "Loss: {}".format(_loss)
        for _ in range(8):
            train_f(_x, _z)

if __name__ == "__main__":
    main()
