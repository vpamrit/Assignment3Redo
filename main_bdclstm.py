import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torchvision
import torchio
from plain_dice import dice_coeff

from losses import DICELossMultiClass, DICELoss
from seg_losses import DiceLoss
from load_data import SpleenDatasetBuilder
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as tr

from CLSTM import BDCLSTM
from models import *

# %% import transforms

SAVE_VALID_IMAGES = True
UNET_MODEL_FILE = 'unetsmall-100-10-0.001'
SAVE_EPOCHS = [5, 10, 15]

# %% Training settings
parser = argparse.ArgumentParser(description='UNet+BDCLSTM')
parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--train', action='store_true', default=False,
                    help='Argument to train model (default: False)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--mom', type=float, default=0.99, metavar='MOM', help='SGD momentum')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training (default: False)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N', help='batches to wait before logging training status')
parser.add_argument('--test-dataset', action='store_true', default=False,
                    help='test on smaller dataset (default: False)')
parser.add_argument('--size', type=int, default=128, metavar='N',
                    help='imsize')
parser.add_argument('--drop', action='store_true', default=False,
                    help='enables drop')
parser.add_argument('--data-folder', type=str, default='./data/Training/', metavar='str',
                    help='folder that contains data (default: test dataset)')


args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
if args.cuda:
    print("We are on the GPU!")

DATA_FOLDER = args.data_folder
CLASSES = [1,6,7,8,9,11]

# %% Loading in the Dataset
slice_size = 240
dset_train = SpleenDatasetBuilder(DATA_FOLDER, (1, 1)).dataset
dset_valid = SpleenDatasetBuilder(DATA_FOLDER, (0, 4)).dataset

train_loader = DataLoader(dset_train, batch_size=args.batch_size, num_workers=4)


# %% Loading in the models
unet = UNetSmall(num_classes=(len(CLASSES) + 1))
#unet.load_state_dict(torch.load(UNET_MODEL_FILE))
model = BDCLSTM(input_channels=32, hidden_channels=[32], num_classes=(len(CLASSES) + 1))

if args.cuda:
    unet.cuda()
    model.cuda()

# Setting Optimizer
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom)
criterion = DiceLoss()

# Define Training Loop
def train(epoch, counter):
    model.train()
    for batch_idx, subjects_batch in enumerate(train_loader):

        image, mask = subjects_batch['t1'][torchio.DATA], subjects_batch['label'][torchio.DATA]

        print(image.shape)
        print(mask.shape)

        #split into three depth images
        image1 = image[:, :, :, :, 0].cuda()
        image2 = image[:, :, :, :, 1].cuda()
        image3 = image[:, :, :, :, 2].cuda()

        mask = mask[:, :, :, :, 1].cuda()

        print(image1.shape)
        print(mask.shape)


        image1, image2, image3, mask = Variable(image1), \
            Variable(image2), \
            Variable(image3), \
            Variable(mask)

        optimizer.zero_grad()

        map1 = unet(image1, return_features=True)
        map2 = unet(image2, return_features=True)
        map3 = unet(image3, return_features=True)

        output = model(map1, map2, map3)

        #print("Mask size is {} Output size is {}".format(mask.size(), output.size()))
        #need to ignore padding border here (for loss)
        loss = criterion(output, mask)

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(image1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    #here we can compute the average dice coefficient on the remaining dataset
    valid_loader = DataLoader(dset_valid, batch_size=1, num_workers=4)
    model.eval()

    total_loss = 0
    dice_total = 0

    print('Computing validation loss...')

    for batch_idx, (image1, image2, image3, mask) in enumerate(valid_loader):
        with torch.no_grad():
            image1, image2, image3, mask = image1.cuda(), \
                 image2.cuda(), \
                 image3.cuda(), \
                 mask.cuda()

            map1 = unet(image1, return_features=True)
            map2 = unet(image2, return_features=True)
            map3 = unet(image3, return_features=True)

            output = model(map1, map2, map3)

            loss = criterion(output, mask)
            total_loss += loss.item()

            #force the output to 0 and 1
            pure_output = (output[:,1,:,:].round() > 0).float()
            dice_total += dice_coeff(pure_output, mask[:, 1, :, :])

            if SAVE_VALID_IMAGES and epoch in SAVE_EPOCHS:
                for i in range(output.size()[0]):
                    pil_img = torchvision.transforms.functional.to_pil_image((output[i, 1, :, :]*125).squeeze_().cpu())
                    mask_img = torchvision.transforms.functional.to_pil_image((mask[i, 1, :, :]*125).squeeze_().cpu())

                    pil_img.save('./gen/' + str(epoch) + '/gen_img_' + str(counter) + '.png')
                    mask_img.save('./gen/'+ str(epoch) + '/mask_img_' + str(counter) + '.png')
                    counter += 1

    print('Validation Epoch: Loss {}, Avg Loss {}\n'.format(total_loss, total_loss / len(valid_loader.dataset)))
    print('Dice Coeff Avg {}'.format(dice_total / len(valid_loader.dataset)))

    return counter

def test(train_accuracy=False):
    test_loss = 0

    if train_accuracy == True:
        loader = train_loader
    else:
        loader = test_loader

    for (image1, image2, image3, mask) in loader:
        if args.cuda:
            image1, image2, image3, mask = image1.cuda(), \
                image2.cuda(), \
                image3.cuda(), \
                mask.cuda()

        image1, image2, image3, mask = Variable(image1, volatile=True), \
            Variable(image2, volatile=True), \
            Variable(image3, volatile=True), \
            Variable(mask, volatile=True)
        map1 = unet(image1, return_features=True)
        map2 = unet(image2, return_features=True)
        map3 = unet(image3, return_features=True)

        # print(image1.type)
        # print(map1.type)

        output = model(map1, map2, map3)
        test_loss += criterion(output, mask).data[0]

    test_loss /= len(loader)
    if train_accuracy:
        print(
            '\nTraining Set: Average Dice Coefficient: {:.4f}\n'.format(test_loss))
    else:
        print(
            '\nTest Set: Average Dice Coefficient: {:.4f}\n'.format(test_loss))


if args.train:
    counter = 0
    for i in range(args.epochs):
        counter = train(i, counter)
        #test()

    torch.save(model.state_dict(),
               'bdclstm-{}-{}-{}'.format(args.batch_size, args.epochs, args.lr))
else:
    model.load_state_dict(torch.load('bdclstm-{}-{}-{}'.format(args.batch_size, args.epochs,args.lr)))
    test()

print("Complete")
