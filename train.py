from train import train
from options import args

if __name__ == '__main__':
    # Get parameters
    opts = args()

    # Train
    train(**opts)
