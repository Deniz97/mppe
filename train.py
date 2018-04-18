from __future__ import print_function
import os
import argparse
import keras

from server.py_rmpe_config import RmpeCocoConfig
from utils.data import DataIterator

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', help='Experiment directory')
    parser.add_argument('data', help='Dataset directory')
    parser.add_argument('--weights', default=None, help='Pretrained model weights')
    parser.add_argument('--init-epoch', default=0, help='Initial epoch', type=int)
    parser.add_argument('--batch-size', default=4, help='Batch size', type=int)
    parser.add_argument('--num-steps', default=-1, help='Batch size', type=int)
    parser.add_argument('--num-outs', default=5, help='Out size', type=int)
    return parser.parse_args(args)


def main(args=None):
    args = parse_args(args)

    batch_size = args.batch_size
    exp_dir = os.path.join('exp', args.exp_dir)
    os.makedirs(exp_dir)

    train_client = DataIterator(os.path.join(args.data, 'coco_train_dataset_480.h5'),
                                RmpeCocoConfig,
                                shuffle=True,
                                augment=True,
                                batch_size=batch_size,
                                num_out=args.num_outs)

    if args.num_steps < 0:
        train_steps = 117576 // batch_size
    else:
        train_steps = args.num_st

    train_di = train_client.gen()

    o = next(train_di)
    pretrained_model = args.weights
    if pretrained_model is not None:
        model = keras.models.load_model(args.weights, custom_objects=custom_objects, compile=True)
    else:
        print('Building model!')
        model = build_stuff_br_net(inter=False)
        losses = [eucl_loss(batch_size=batch_size)]*args.num_outs
        adam = keras.optimizers.adam(lr=1e-4, clipnorm=0.001)
        model.compile(loss=losses, optimizer=adam)

    checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(exp_dir,'weights.{epoch:02d}.h5'), verbose=1)
    csv_log = keras.callbacks.CSVLogger(os.path.join(exp_dir,'training_log.csv'), separator=',', append=False)


    model.fit_generator(generator=train_di,
                        steps_per_epoch=train_steps,
                        epochs=8,
                        callbacks=[checkpoint, csv_log],
                        verbose=1,
                        initial_epoch=args.init_epoch,
                        max_queue_size=20,
                        workers=4)


if __name__ == '__main__':
    main()