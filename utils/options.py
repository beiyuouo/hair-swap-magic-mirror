import argparse


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--epochs', type=int, default=20, help="epochs of training")
    args.add_argument('--batch_size', type=int, default=16, help="batch size")
    args.add_argument('--start_lr', type=float, default=0.001, help="learning rate")
    args.add_argument('--milestones', type=str, default='10,20,30', help='Milestones for LR decreasing')


    args.add_argument('--model', type=str, default='cnn', help='model name')
    args.add_argument('--model_path', type=str, default='./log', help='path to save model')
    args.add_argument('--model_chkp', type=str, default=10, help='check point')

    args.add_argument('--dataset', type=str, default='mnist', help='dataset name')
    args.add_argument('--data_path', type=str, default='./data', help='path to dataset')
    args.add_argument('--trainxml', type=str, default='./data/training.xml', help='path to xml file')
    args.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    args.add_argument('--backend', type=str, default='resnet34', help='Feature extractor')
    args.add_argument('--snapshot', type=str, default=None, help='Path to pretrained weights')
    args.add_argument('--crop_x', type=int, default=256, help='Horizontal random crop size')
    args.add_argument('--crop_y', type=int, default=256, help='Vertical random crop size')

    args.add_argument('--alpha', type=float, default=0.4, help='Coefficient for classification loss term')
    args.add_argument('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
    args.add_argument('--result_path', type=str, default='./result', help='path to save result')

    args = args.parse_args()
    return args
