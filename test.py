import sys
import os
import argparse
import model
import torch
import skimage
import re
from skimage import io
from os.path import join, isfile
from os import listdir

#custom imports
import visualize
import load_data
from visualize import TestVisualizer

def get_files(argv):
    if argv.image_dir != '':
        root_dir = argv.image_dir
        files = [root_dir+f for f in os.listdir(root_dir) if isfile(join(root_dir, f))]
    else:
        files = [argv.image_path]

    return files


def main(argv):
    # my code here
    files = get_files(argv)


    for f in files:
        if not torch.cuda.is_available():
            print("The model needs to be loaded to a GPU")

        device = torch.device('cuda')

        net = model.ResNet101().to('cuda')
        net.eval()

        with torch.no_grad():

            net.load_state_dict(torch.load(argv.model_path))
            image = io.imread(f)
            image = torch.from_numpy(image.transpose(2,0,1))
            width, height = image.size()[2], image.size()[1]


            pred_labels = net(image.unsqueeze(0).float().to('cuda')).tolist()[0]
            pred_labels[0] /= float(width)
            pred_labels[1] /= float(height)

            print(f)
            print("X: {0:.4f}".format(pred_labels[0]))
            print("Y: {0:.4f}".format(pred_labels[1]))

            if argv.visual_dir != '':
                if argv.labels_file != '':
                    actual_label = list(load_data.read_txt(f, argv.labels_file)[0, 1:3].astype('float').reshape(-1,2).squeeze())

                    testv = TestVisualizer(f, actual_label, pred_labels)
                    print("Testing error: {:3f}".format(testv.get_error()))
                else:
                    testv = TestVisualizer(f, pred_labels)

                testv.save_image("./"+argv.visual_dir+"/")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #single image outputs
    parser.add_argument('--image_path', type=str, default='./data/test/120.jpg', help='test image directory')
    parser.add_argument('--model_path', type=str, default='./models/resnet32.pt' , help='path for model to load')
    parser.add_argument('--labels_file', type=str , default='', help='labels file for visualized images')

    #args for directory-based outputs
    parser.add_argument('--image_dir', type=str, default='', help='image directory if provided')
    parser.add_argument('--visual_dir', type=str , default='', help='directory to save visualized images to')

    args = parser.parse_args()
    print(args)
    main(args)

