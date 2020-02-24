from train import train
from options import parser

if __name__ == '__main__':
    # Get parameters
    opts = parser()

    # Train
    model = SfM(**opts)
    model.trai()
