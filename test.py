from PIL import Image
from timm.data import create_transform

import models  # load models
from imagenet_classes import imagenet_classes


def main():
    # can change different model name
    model = models.mambaout_femto(pretrained=True)
    model.eval()

    transform = create_transform(
        input_size=224,
        crop_pct=model.default_cfg['crop_pct']
    )

    image = Image.open('dog.jpg')
    input_image = transform(image).unsqueeze(0)

    pred = model(input_image)
    print(f'Prediction: {imagenet_classes[int(pred.argmax())]}.')


if __name__ == '__main__':
    main()
