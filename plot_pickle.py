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
    example_dict = pickle.load(pickle_in)
    epochs = np.arange(1, 50 + 1)
    plt.figure(1)
    for i in example_dict.keys():
        #print(i, example_dict[i])
        plt.plot(example_dict[i], epochs, label=i)
    plt.legend()
    plt.ylabel('epochs')
    plt.xlabel('MINE_values')
    plt.show()
    plt.savefig('result.png')


if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    main()