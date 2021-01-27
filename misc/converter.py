import os
import numpy as np
import toolkits
import utils as ut
from tqdm import tqdm
datapath = "/home/amir/Datasets/VoxCeleb/Test/wav"
target = "/home/amir/Datasets/VoxCeleb/Test/ghosted_512"


def main():

    # gpu configuration
    toolkits.initialize_GPU()
    import model as model
    convertlist = np.loadtxt('../meta_data/Test_list.txt', dtype=str, usecols=[1])

    convertlist = np.array([os.path.join(datapath, i) for i in convertlist])


    # ==================================
    #       Get Model
    # ==================================
    # construct the data generator.
    params = {'dim': (257, None, 1),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 5994,
              'sampling_rate': 16000,
              'normalize': True,
              }

    network_eval = model.ghostvlad_model_resnet(input_dim=params['dim'], mode='eval')

    network_eval.load_weights(os.path.join('../../saved_models/ghostvlad_weights.h5'), by_name=True)

    print('==> start converting.')

    for ID in tqdm(convertlist):
        specs = ut.load_data(ID, win_length=params['win_length'], sr=params['sampling_rate'],
                             hop_length=params['hop_length'], n_fft=params['nfft'],
                             spec_len=params['spec_len'], mode='eval')
        specs = np.expand_dims(np.expand_dims(specs, 0), -1)

        v = network_eval.predict(specs)
        target_path = ID.replace(datapath, target)
        if not os.path.isdir(os.path.dirname(target_path)):
            try:
                os.makedirs(os.path.dirname(target_path))
            except FileExistsError:
                pass
        np.save(target_path, v[0])


if __name__ == "__main__":
    main()
