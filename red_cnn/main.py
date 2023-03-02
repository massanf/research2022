import os
import argparse
import types
from torch.backends import cudnn
from red_cnn.loader import get_loader
from red_cnn.solver import Solver


def main(
    device: str,
    mode="train",
    load_mode=0,
    saved_path="./red_cnn/npy_img/",
    save_path="./red_cnn/save/",
    test_patient="sample1002",
    result_fig=True,
    norm_range_min=-1024.0,
    norm_range_max=3072.0,
    trunc_min=-160.0,
    trunc_max=240.0,
    transform=False,
    patch_n=10,
    patch_size=64,
    batch_size=16,
    num_epochs=100,
    print_iters=20,
    decay_iters=3000,
    save_iters=1000,
    test_iters=1000,
    lr=1e-5,
    num_workers=7,
    multi_gpu=False,
):
    cudnn.benchmark = True

    args = types.SimpleNamespace()

    args.device = device
    args.mode = mode
    args.load_mode = load_mode
    args.saved_path = saved_path
    args.save_path = save_path
    args.test_patient = test_patient
    args.result_fig = result_fig
    args.norm_range_min = norm_range_min
    args.norm_range_max = norm_range_max
    args.trunc_min = trunc_min
    args.trunc_max = trunc_max
    args.transform = transform
    args.patch_n = patch_n
    args.patch_size = patch_size
    args.batch_size = batch_size
    args.num_epochs = num_epochs
    args.print_iters = print_iters
    args.decay_iters = decay_iters
    args.save_iters = save_iters
    args.test_iters = test_iters
    args.lr = lr
    args.num_workers = num_workers
    args.multi_gpu = multi_gpu

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("Create path : {}".format(save_path))

    if result_fig:
        fig_path = os.path.join(save_path, "fig")
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
            print("Create path : {}".format(fig_path))

    data_loader = get_loader(
        mode=mode,
        load_mode=load_mode,
        saved_path=saved_path,
        test_patient=test_patient,
        patch_n=(patch_n if mode == "train" else None),
        patch_size=(patch_size if mode == "train" else None),
        transform=transform,
        batch_size=(batch_size if mode == "train" else 1),
        num_workers=num_workers,
    )

    solver = Solver(args, data_loader)
    if mode == "train":
        solver.train()
    elif mode == "test":
        solver.test()


# if __name__ == "__main__":
def __init__(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--load_mode", type=int, default=0)
    # parser.add_argument('--data_path', type=str, default='./AAPM-Mayo-CT-Challenge/')
    # parser.add_argument('--saved_path', type=str, default='./npy_img/')
    parser.add_argument("--saved_path", type=str, default="./red_cnn/npy_img/")
    parser.add_argument("--save_path", type=str, default="./red_cnn/save/")
    parser.add_argument("--test_patient", type=str, default="sample1002")
    parser.add_argument("--result_fig", type=bool, default=True)

    parser.add_argument("--norm_range_min", type=float, default=-1024.0)
    parser.add_argument("--norm_range_max", type=float, default=3072.0)
    parser.add_argument("--trunc_min", type=float, default=-160.0)
    parser.add_argument("--trunc_max", type=float, default=240.0)

    parser.add_argument("--transform", type=bool, default=False)
    # if patch training, batch size is (--patch_n * --batch_size)
    parser.add_argument("--patch_n", type=int, default=10)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--print_iters", type=int, default=20)
    parser.add_argument("--decay_iters", type=int, default=3000)
    parser.add_argument("--save_iters", type=int, default=1000)
    parser.add_argument("--test_iters", type=int, default=1000)

    parser.add_argument("--lr", type=float, default=1e-5)

    parser.add_argument("--device", type=str)
    parser.add_argument("--num_workers", type=int, default=7)
    parser.add_argument("--multi_gpu", type=bool, default=False)

    args = parser.parse_args(args)
    main(args)
