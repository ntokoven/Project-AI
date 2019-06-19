import pickle
import argparse
def main():
    parser = argparse.ArgumentParser(description='read pickle')
    parser.add_argument('--import_path', type=str, default="mi_values_dict", metavar='N',
                        help='none')
    args = parser.parse_args()
    pickle_in = open(args.import_path+".pickle","rb")
    example_dict = pickle.load(pickle_in)
    for i in example_dict.keys():
        print(i,example_dict[i])

if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    main()