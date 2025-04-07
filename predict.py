import torch
from torchvision import transforms
import time
from functions.lf_utils import *
from model import Generator
# from model_odd import Generator
# from model_all import Generator
import argparse


def main(args):
    normalize_fn = normalize
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ])

    # load image

    valid_lf_extras = args.valid_lf
    valid_lf_extras = get_lf_extra(valid_lf_extras, args.valid_path, args.Nnum, normalize_fn=normalize_fn)

    # create model
    model = Generator(data_size=args.data_size,n_slice=args.n_slice,Nnum=args.Nnum).to(device)

    # load model weights

    model_weight_path = args.model_weight_path


    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        valid_lf_extras = torch.Tensor(valid_lf_extras)
        valid_lf_extras = valid_lf_extras.permute(2, 0, 1)
        valid_lf_extras = valid_lf_extras.unsqueeze(0).type(torch.cuda.FloatTensor)
        # for i in range(60):
        time_star = time.time()
        output = model(valid_lf_extras).to(device)
        time_end = time.time()
        # print(time_end - time_star)
        output = output.permute(0, 2, 3, 1).cpu()


    output = output.numpy()
    output[output<-1] = -1

    write3d(output, args.save_path, bitdepth=16, norm_max=True)
    print(time_end - time_star)


if __name__ == '__main__':

        parser = argparse.ArgumentParser()
        # parser.add_argument('--n_slice', type=int, default=61)
        parser.add_argument('--n_slice', type=int, default=61)
        parser.add_argument('--Nnum', type=int, default=11)
        # parser.add_argument('--Nnum', type=int, default=13)
        # parser.add_argument('--data_size', type=tuple, default=(490, 490))
        parser.add_argument('--data_size', type=tuple, default=(682, 682))
        parser.add_argument('--valid_path', type=str, default='./to_predict/')
        parser.add_argument('--model_weight_path', type=str, default='weights/M40x_Nnum11_n_slice61_tubulins600.pth')
        parser.add_argument('--valid_lf', type=str, default='x40tubulinsNnum13.tif')
        parser.add_argument('--save_path', type=str, default='result/x40tubulinsNnum13.tif')
        opt = parser.parse_args()
        main(opt)





















