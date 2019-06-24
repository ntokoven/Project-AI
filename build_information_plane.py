import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt


def build_information_plane(MI):
    plt.figure(2)
    epochs = len(MI['maxP1'])
    epochs = np.arange(1, epochs + 1)
    layers = ['maxP1','maxP2','relu3','sm1']
    #print(MI)
    for layer in layers:    
        #for epoch in range(len(MI[layer])):
        plt.scatter(np.max(MI[layer][:]), np.max(MI[layer+'T'][:]), label=layer)
    plt.legend()
    plt.ylabel('MI(T, Y)')
    plt.xlabel('MI(X, T)')
    plt.savefig('information_plane.png')
    plt.show()
    



def main():
    parser = argparse.ArgumentParser(description='read pickle')
    parser.add_argument('--import-path', type=str, default="mi_values_dict", metavar='N',
                        help='none')
    args = parser.parse_args()
    pickle_in = open(args.import_path+".pickle","rb")
    mi_values = pickle.load(pickle_in)
    build_information_plane(mi_values)
if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    main()