import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='read pickle')
    parser.add_argument('--import-path', type=str, default="mi_values_dict", metavar='N',
                        help='none')
    parser.add_argument('--title', type=str, default="", metavar='N',
                        help='none')
    args = parser.parse_args()
    pickle_in = open(args.import_path+".pickle","rb")
    mi_values = pickle.load(pickle_in)
    epochs = np.arange(1, 30 + 1)
    values = {}
    values = np.array(mi_values['relu3T'])
    print(values.shape)
    plt.figure(1)
    for epoch in range(len(values)):
        plt.plot(epochs, values[epoch][:30], label='Run %s' % (epoch + 1))
    plt.legend()
    plt.ylabel('MI(Y, relu3)')
    plt.xlabel('MINE epochs')
    plt.savefig('%s+%s_T.png' % (args.import_path, args.title))
    plt.show()
   

if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    main()