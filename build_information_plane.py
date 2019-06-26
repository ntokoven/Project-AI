import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict

def build_information_plane(MI, path):
    #plt.figure(2, figsize=(12,6.5))
    epochs = len(MI['maxP1'])
    epochs = np.arange(1, epochs + 1)
    layers = ['maxP1','maxP2','relu3','sm1']
    step = 0
    #print(MI)
    for layer in layers:    
        #print(step*0.125, (step+1)*0.125, '\n\n')
        colors = cm.rainbow(np.linspace(step*0.125, (step+1)*0.125, 1))[0]
        #print(len(colors),'\n')
        #print(len(MI[layer]))
        values = defaultdict(list)
        num = 1
        for epoch in range(len(MI[layer])):
            #values[layer].append(np.max(MI[layer][epoch]))
            #values[layer+'T'].append(np.max(MI[layer+'T'][epoch]))
            values[layer].append(np.median(MI[layer][epoch][-5]))
            values[layer+'T'].append(np.median(MI[layer+'T'][epoch][-5]))
            #plt.scatter(np.max(MI[layer][epoch]), np.max(MI[layer+'T'][epoch]), label=layer, color = colors[epoch])
            if num == 1:
                plt.scatter(np.median(MI[layer][epoch][-5]), np.median(MI[layer+'T'][epoch][-5]), label=layer, color = colors)#[epoch])
            else:
                plt.scatter(np.median(MI[layer][epoch][-5]), np.median(MI[layer+'T'][epoch][-5]), color = colors)#[epoch])
            plt.annotate(num, (np.median(MI[layer][epoch][-5]), np.median(MI[layer+'T'][epoch][-5])))
            num += 1
        #plt.plot(values[layer], values[layer+'T'], label=layer, color = colors[epoch])
        #print(layer, epoch, MI[layer+'T'][epoch], '\n')
        step += 2
    plt.legend(loc='best')#center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel('MI(T, Y)')
    plt.xlabel('MI(X, T)')
    plt.savefig(path+'_information_plane.png')
    plt.show()
    



def main():
    parser = argparse.ArgumentParser(description='read pickle')
    parser.add_argument('--import-path', type=str, default="mi_values_dict", metavar='N',
                        help='none')
    parser.add_argument('--import-path-second', type=str, default="", metavar='N',
                        help='none')
    args = parser.parse_args()
    pickle_in = open(args.import_path+".pickle","rb")
    mi_values = pickle.load(pickle_in)
    flag = False
    if args.import_path_second != '':
        flag = True
    

    def replace_nan(mi_values):
        for key in mi_values.keys():
            mi_values[key] = np.array(mi_values[key])

            for epoch in range(len(mi_values[key])):
                #print(np.where(np.isnan(mi_values[key][epoch])))
                nan_ind = np.where(np.isnan(mi_values[key][epoch]))
                if len(nan_ind[0]) != 0:
                    last_nan = nan_ind[0][0]
                    mi_values[key][epoch][nan_ind] = np.mean(mi_values[key][epoch][-(last_nan - 5):-last_nan])
        return mi_values

    mi_values = replace_nan(mi_values)

    if flag:
        pickle_in = open(args.import_path_second+".pickle","rb")
        mi_values_second = pickle.load(pickle_in)
        mi_values_second = replace_nan(mi_values_second)
        for layer in ['maxP1', 'maxP2', 'relu3', 'sm1']:
            mi_values[layer] = mi_values_second[layer]

        #print(len(mi_values[key]), len(mi_values[key][0]))
    #print(mi_values['sm1T'])
    build_information_plane(mi_values, args.import_path)
if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    main()