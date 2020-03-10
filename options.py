import argparse

def parser():
    """ Returns a dictionarty of all the options for training """

    # Models
    parser = argparse.ArgumentParser("Arguments for training model")

    # Networks
    parser.add_argument(
        "--pose_network",
        type=str,
        choices=["Simple_pose"],
        default="Simple_pose",
        help="The architecture used for the pose network")
    parser.add_argument(
        "--encoder_network",
        type=str,
        choices=["ResNet", "U_net"],
        default="U_net",
        help="The architecture used for the encoder network")
    parser.add_argument(
        "--decoder_network",
        type=str,
        choices=["Simple_decoder"],
        default="Simple_decoder",
        help="The architecture used for the decoder network")

    # Config
    parser.add_argument(
        "--batch_size",
        type=int,
        default=12,
        help="The size of each batch")
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=20,
        help="The total number of epochs")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=6,
        help="Number of CPUS")
    parser.add_argument(
        "--width",
        type=int,
        default=600,
        help="")
    parser.add_argument(
        "--height",
        type=int,
        default=600,
        help="")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="")
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="")
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="")
    parser.add_argument(
        "--no_gpu",
        action="store_true",
        help="Use only CPU if flag is passed")

    # Dataset
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Shapes3D_loader",
        choices=["Shapes3D_loader"],
        help="The name of the dataset to use")
    parser.add_argument(
        "--dataset_folder",
        type=str,
        required=True,
        help="")
    parser.add_argument(
        "--frames",
        type=str,
        default=['a', 'b', 'c'],
        nargs=3,
        help="")

    # Loggin and saving
    parser.add_argument(
        "--log_frequency",
        type=int,
        default=250,
        help="Number of batches between each log")
    parser.add_argument(
        "--save_frequency",
        type=int,
        default=1,
        help="Number of epochs between each save")
    parser.add_argument(
        "--model_name",
        default="unnamed_model",
        type=str,
        help="Name of the model for logging purposes")
    parser.add_argument(
        "--save_folder",
        type=str,
        default=".",
        help="Folder where to save and log the model")

    args = parser.parse_args()
    return vars(args)
