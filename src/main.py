import os
from train import train_baseline_convolution_model

if __name__ == "__main__":
    should_train_model = True
    model_path = "/home/linuxu/afeka/computer-vision/baseline_cnn_trained.pth"
    if should_train_model or not os.path.exists(model_path):
        cnn_model = train_baseline_convolution_model()
        print("traine basedline convloution completed")
    else:
        print("no need to train cnn baseline, it exist")
