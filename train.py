from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import keras
import numpy as np
import generator
import model

sys.path.append('../')
from misc import toolkits

# ===========================================
#        Parse the argument
# ===========================================
import argparse
parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--resume', default='../saved_models/EndtoEnd/weights-54-0.961.h5', type=str)
parser.add_argument('--batch_size', default=64, type=int)
# parser.add_argument('--data_path', default='../dataset/stft_embedding/', type=str)
parser.add_argument('--data_path', default='/home/amir/Datasets/VoxCeleb/V1/ghosted_4096', type=str)
parser.add_argument('--test_path', default='/home/amir/Datasets/VoxCeleb/Test/ghosted_4096', type=str)
parser.add_argument('--multiprocess', default=50, type=int)
# set up network configuration.
parser.add_argument('--bottleneck_dim', default=512, type=int)
# set up learning rate, training loss and optimizer.
parser.add_argument('--epochs', default=56, type=int)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--warmup_ratio', default=0, type=float)
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
parser.add_argument('--ohem_level', default=3, type=int,
                    help='pick hard samples from (ohem_level * batch_size) proposals, must be > 1')
global args
args = parser.parse_args()

def main():
    params = {    'mp_pooler': args.multiprocess,
                  'n_classes': 2,
                  'batch_size': args.batch_size
                  }
    trnlist, trnlb = toolkits.get_datalist(args, path='../meta_data/VoxCeleb1_training.txt', suffix='.npy')
    trn_gen = generator.DataGenerator(trnlist, trnlb, **params)

    testlist, testlb = toolkits.get_testlist(args, path='../meta_data/veri_test.txt', suffix='.npy')
    test_gen = generator.TestGenerator(testlist, testlb, **params)

    ete_model = model.end_to_end_capsule_model(input_dim=(2, 4096))
    if args.resume:
        ete_model.load_weights(args.resume, by_name=True)
        print('model loaded successfully.')

    ete_model.summary()

    normal_lr = keras.callbacks.LearningRateScheduler(step_decay)
    tbcallbacks = keras.callbacks.TensorBoard(log_dir='../log', histogram_freq=0, write_graph=True, write_images=False,
                                              update_freq=args.batch_size * 16)
    callbacks = [keras.callbacks.ModelCheckpoint(os.path.join('../saved_models','EndtoEnd', 'weights-{epoch:02d}-{val_acc:.3f}.h5'),
                                                 monitor='val_acc',
                                                 mode='max',
                                                 save_best_only=True),
                 normal_lr, tbcallbacks]

    ete_model.fit_generator(trn_gen,
                            steps_per_epoch=len(trn_gen),
                            callbacks=callbacks,
                            epochs=args.epochs,
                            validation_data=test_gen,
                            validation_steps=len(test_gen))

def step_decay(epoch):
    '''
    The learning rate begins at 10^initial_power,
    and decreases by a factor of 10 every step epochs.
    '''
    half_epoch = args.epochs // 2
    stage1, stage2, stage3 = int(half_epoch * 0.5), int(half_epoch * 0.8), half_epoch
    stage4 = stage3 + stage1
    stage5 = stage4 + (stage2 - stage1)
    stage6 = args.epochs

    if args.warmup_ratio:
        milestone = [2, stage1, stage2, stage3, stage4, stage5, stage6]
        gamma = [args.warmup_ratio, 1.0, 0.1, 0.01, 1.0, 0.1, 0.01]
    else:
        milestone = [stage1, stage2, stage3, stage4, stage5, stage6]
        gamma = [1.0, 0.1, 0.01, 1.0, 0.1, 0.01]

    lr = 0.005
    init_lr = args.lr
    stage = len(milestone)
    for s in range(stage):
        if epoch < milestone[s]:
            lr = init_lr * gamma[s]
            break
    print('Learning rate for epoch {} is {}.'.format(epoch + 1, lr))
    return np.float(lr)

if __name__ == '__main__':
    main()
