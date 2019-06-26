import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='read pickle')
    parser.add_argument('--import-path', type=str, default="mi_values_dict", metavar='N',
                        help='none')
    args = parser.parse_args()
    pickle_in = open(args.import_path+".pickle","rb")
    mi_values = pickle.load(pickle_in)
    epochs = np.arange(1, 2 + 1)
    values = {}
    for key in mi_values.keys():
        values[key] = np.array(mi_values[key])#.reshape(180)
        print(values[key].shape)
    plt.figure(1)
    for key in mi_values.keys():
        print(len(mi_values[key]), len(mi_values[key][1]))
    for i in mi_values.keys():
        #print(i, example_dict[i])
        plt.plot(epochs, np.max(values[i], axis=1), label=i)
    plt.legend()
    plt.ylabel('MI(x, L_i + eps)')
    plt.xlabel('MINE epochs')
    plt.savefig('%s.png' % args.import_path)
    plt.show()


if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    main()