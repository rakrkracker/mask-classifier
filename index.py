from webcam import startCameraFeed
from mask_pred import load_model


# Main program
def main():
    model = load_model()
    feed = startCameraFeed(model)


if __name__ == "__main__":
    main()
