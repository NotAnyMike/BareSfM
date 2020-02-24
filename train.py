from trainer import SfM
from options import parser

if __name__ == '__main__':
    # Get parameters
    opts = parser()

    # Train
    model = SfM(**opts)
    model.train()
