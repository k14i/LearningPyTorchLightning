import pytorch_lightning as pl

from arguments import get_argparser
from mnist_conv_net import MNISTConvNet


def main(args=None) -> None:
    if not args:
        args = get_argparser().parse_args()
    
    model = MNISTConvNet(conv1_in_channels=args.conv1_in_channels, conv1_out_channels=args.conv1_out_channels, conv1_kernel_size=args.conv1_kernel_size, conv1_stride=args.conv1_stride,
        conv2_in_channels=args.conv2_in_channels, conv2_out_channels=args.conv2_out_channels, conv2_kernel_size=args.conv2_kernel_size, conv2_stride=args.conv2_stride,
        pool1_kernel_size=args.pool1_kernel_size, dropout1_p=args.dropout1_p, dropout2_p=args.dropout2_p,
        fullconn1_in_features=args.fullconn1_in_features, fullconn1_out_features=args.fullconn1_out_features, fullconn2_in_features=args.fullconn2_in_features, fullconn2_out_features=args.fullconn2_out_features,
        adadelta_lr=args.adadelta_lr, adadelta_rho=args.adadelta_rho, adadelta_eps=args.adadelta_eps, adadelta_weight_decay=args.adadelta_weight_decay,
        dataset_root=args.dataset_root, dataset_download=args.dataset_download,
        dataloader_mean=args.dataloader_mean, dataloader_std=args.dataloader_std, dataloader_batch_size=args.dataloader_batch_size, dataloader_num_workers=args.dataloader_num_workers)
    
    trainer = pl.Trainer(progress_bar_refresh_rate=args.progress_bar_refresh_rate, max_epochs=args.max_epochs)
    trainer.fit(model)
    trainer.test(model)
    # trainer.predict(model)


if __name__ == "__main__":
    # argparser = get_argparser()
    # args = argparser.parse_args([
    #     "--max-epochs", str(1),
    #     "--adadelta-lr", str(0.5),
    # ])
    # main(args)

    main()

