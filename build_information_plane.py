import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def build_information_plane(MI, path):
    plt.figure(2, figsize=(20,10))
    epochs = len(MI['maxP1'])
    epochs = np.arange(1, epochs + 1)
    layers = ['maxP1','maxP2','relu3','sm1']
    step = 0
    #print(MI)
    for layer in layers:    
        print(step*0.125, (step+1)*0.125, '\n\n')
        colors = cm.rainbow(np.linspace(step*0.125, (step+1)*0.125, len(MI[layer])))
        print(len(colors),'\n')
        print(len(MI[layer]))
        for epoch in range(len(MI[layer])):
            plt.scatter(np.max(MI[layer][epoch]), np.max(MI[layer+'T'][epoch]), label=layer, color = colors[epoch])
        step += 2
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel('MI(T, Y)')
    plt.xlabel('MI(X, T)')
    plt.savefig(path+'_information_plane.png')
    plt.show()
    



def main():
    parser = argparse.ArgumentParser(description='read pickle')
    parser.add_argument('--import-path', type=str, default="mi_values_dict", metavar='N',
                        help='none')
    args = parser.parse_args()
    pickle_in = open(args.import_path+".pickle","rb")
    mi_values = pickle.load(pickle_in)
    for key in mi_values.keys():
        print(len(mi_values[key]), len(mi_values[key][0]))
    #print(mi_values['sm1T'])
    build_information_plane(mi_values, args.import_path)
if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    main()