import argparse


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--model', type=str, default='./log', help='model name (seg, vae, gan)')
    args.add_argument('--dataset', type=str, default='mnist', help='dataset name')
    args.add_argument('--trainxml', type=str, default='./seg_data/training.xml', help='path to xml file')
    args.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    args.add_argument('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
    args.add_argument('--model_path', type=str, default='./log', help='path to save model')
    args.add_argument('--model_chkp', type=str, default=10, help='check point')
    args.add_argument('--result_path', type=str, default='./result', help='path to save result')

    # Segmentation
    args.add_argument('--seg_epochs', type=int, default=50, help="epochs of training")
    args.add_argument('--seg_batch_size', type=int, default=8, help="batch size")
    args.add_argument('--seg_data_path', type=str, default='./seg_data', help='path to dataset')
    args.add_argument('--seg_model', type=str, default='PSPNet', help='Model name')
    args.add_argument('--seg_lr', type=float, default=0.001, help="learning rate")
    args.add_argument('--milestones', type=str, default='10,20,30', help='Milestones for LR decreasing')

    args.add_argument('--seg_backend', type=str, default='resnet34', help='Feature extractor')
    args.add_argument('--seg_snapshot', type=str, default=None, help='Path to pretrained weights')
    args.add_argument('--seg_crop_x', type=int, default=256, help='Horizontal random crop size')
    args.add_argument('--seg_crop_y', type=int, default=256, help='Vertical random crop size')
    args.add_argument('--seg_alpha', type=float, default=0.4, help='Coefficient for classification loss term')

    args.add_argument('--vae_epochs', type=int, default=20, help="epochs of training")
    args.add_argument('--vae_batch_size', type=int, default=16, help="batch size")
    args.add_argument('--vae_data_path', type=str, default='./vae_data', help='path to dataset')
    args.add_argument('--vae_zsize', type=int, default=256, help='Vertical random crop size')
    args.add_argument('--vae_lr', type=float, default=0.0005, help="learning rate")

    args = args.parse_args()
    return args
