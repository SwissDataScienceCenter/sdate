
class Base_Loss():
    mean = 0
    std = 1
    reflect = False
    scale = 1

    def __init__(self):
        pass

    @staticmethod
    def process_data(img):
        img -= Base_Loss.mean
        img /= Base_Loss.std
        img *= Base_Loss.scale
        if Base_Loss.reflect:
            img *= -1;
        return img

    @staticmethod
    def compute_train_loss(model, img, device):
        pass


    @staticmethod
    def compute_test_loss(args, model, img, device):
        pass