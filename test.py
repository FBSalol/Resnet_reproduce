import argparse
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

import torchvision.transforms as transforms
import cv2
import os
import json
import resnet

def get_argparse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,default='./dataset/show',help='path to dataset')
    parser.add_argument('--checkpoint',type=str,default='./checkpoint/best.pth',help='checkpoint path')
    parser.add_argument('--num_classes',type=int,default=5,help='number of classes')

    return parser

def main():
    args = get_argparse().parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])

    image_path = [os.path.join(args.data_path,f) for f in os.listdir(args.data_path) if f.endswith('.jpg')]
    image_path = image_path[:10]

    class_indict = json.load(open("./class_indices.json"))

    model = resnet.ResNet_18()
    num_ftrs = model.fc.in_features  # 获取全连接层的输入特征数量
    model.fc = torch.nn.Linear(num_ftrs, args.num_classes)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    model.eval()

    for idx, image_path in enumerate(image_path):

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = data_transform(img).unsqueeze(0)
        img = img.to(device)

        with torch.no_grad():
            output = model(img)
            predict = torch.softmax(output, dim=1)
            pred = torch.argmax(predict,dim=1).cpu().numpy()

        # print_res = "class: {} prob: {:.3}".format(class_indict[str(pred)],
        #                                            predict[pred].cpu().numpy())
        class_name = class_indict[str(pred[0])]  # 获取类别名称
        prob = predict[0, pred[0]].item()  # 获取预测概率
        print_res = "class: {} prob: {:.3f}".format(class_name, prob)
        print(f"Image: {image_path}, {print_res}")
        print("prediction:", predict)


if __name__ == '__main__':
    main()