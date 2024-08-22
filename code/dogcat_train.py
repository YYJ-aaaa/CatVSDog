import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from resnet import create_model, model_info
from datasets import DogCat
from tensorboardX import SummaryWriter
import os
from datetime import datetime

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch Training Example')
parser.add_argument('--arch' , default='resnet18', help='resnet18 | resnet34 | resnet50 | resnet101 | resnet152') #resnet的结构选择
parser.add_argument('--model', help='model file path') #resnet的结构选择
parser.add_argument('--train_dataset',required=True, help='layer of resnet') #resnet的结构选择
parser.add_argument('--trainval_dataset',required=True, help='layer of resnet') #resnet的结构选择
parser.add_argument('--output', default='/output', help='folder to output images and model checkpoints') #输出结果保存路径
args = parser.parse_args()

# 超参数设置
EPOCH = 3   #遍历数据集次数
pre_epoch = 0
BATCH_SIZE = 16      #批处理尺寸(batch_size)
LR = 0.01        #学习率


#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# 准备数据集并预处理
transforms_dogcat = transforms.Compose([
    # transforms.Resize(256),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
print(args.train_dataset, args.trainval_dataset)

trainset = DogCat(data_dir=args.train_dataset, train=True, transform=transforms_dogcat) #训练数据集
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取
trainvalset =  DogCat(data_dir=args.trainval_dataset, train=True, transform=transforms_dogcat) #训练数据集
trainvalloader = torch.utils.data.DataLoader(trainvalset, batch_size=100, shuffle=True, num_workers=2)
# 分类标签
classes = ('dog', 'cat')

# 模型定义-ResNet
# 未传入预训练模型的，将pretrained设置为False
if args.model is not None:
    pretrained = True
else:
    pretrained = False
print(pretrained, args.model)
net = create_model(args.arch, pretrained=pretrained, model_path=args.model, num_classes=2)

#查看模型信息
model_info(net)

if pretrained:
    # 在预训练的情况下固定除了全连接层所有层的权重，反向传播时将不会计算梯度（可根据需要自行设置需要训练的网络层）
    # 将其余层设置为Flase更少的计算量 速度快 而且一般收敛会更快
    for param in net.parameters():
        param.requires_grad = False
    for param in net.fc.parameters():
        param.requires_grad = True
#再次查看模型信息
model_info(net)

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
optimizer = optim.Adam(net.parameters(), lr=LR) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
lrScheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

# 训练
if __name__ == "__main__":
    print("Start Training, %s!" % args.arch)
    #生成图模型样例
    dummy_input = torch.rand(4, 3, 224, 224)
    #指定 Tensorboard 存储路径  文件夹名称：时间_网络结构名称
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join(
        '/output', 'logs', current_time + '_' + args.arch)
    with SummaryWriter(logdir) as writer:
        writer.add_graph(net, (dummy_input,))
        net.to(device)
        for epoch in range(pre_epoch, EPOCH):
            print('\nEpoch: %d' % (epoch + 1))
            net.train()
            sum_loss = 0.0
            correct = 0.0
            total = 0.0
            for i, data in enumerate(trainloader):
                # 准备数据
                length = len(trainloader)
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                # forward + backward
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # 每训练1个batch打印一次loss和准确率
                sum_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.data).cpu().sum().float()
                #记录损失以及训练精度
                print('[epoch:%d, iter:%d] curr_Loss: %0.3f Loss: %.03f | Acc: %.3f%% '
                      % (epoch + 1, (i + 1 + epoch * length), loss.item(), sum_loss / (i + 1), 100. * correct / total))
                writer.add_scalar('scalar/train_loss', sum_loss / (i + 1), (i + 1 + epoch * length))
                writer.add_scalar('scalar/train_acc', 100. * correct / total, (i + 1 + epoch * length))

            #查看学习率
            for p in optimizer.param_groups:
                output = []
                for k, v in p.items():
                    if k == 'lr':
                        print('%s:%s'%(k, v))
                        writer.add_scalar('scalar/learning_rate', v, (epoch + 1))
            lrScheduler.step()
            # 记录参数信息
            for name, param in net.named_parameters():
                if 'bn' not in name:
                    writer.add_histogram(name, param, epoch + 1)
            #writer.add_image('Image', x, n_iter)
            writer.add_text('Text', 'text logged at step:' + str(epoch + 1), epoch + 1)

            #每训练完一个epoch测试一下准确率
            print("Waiting Test!")
            with torch.no_grad():
                correct = 0
                total = 0
                for data in trainvalloader:
                    net.eval()
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    # 取得分最高的那个类 (outputs.data的索引号)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().float()

                print('test acc is %.3f%%' % (100 * correct / total))
                acc = 100. * correct / total
                writer.add_scalar('scalar/trainval_acc', acc, epoch + 1)
                # 将每次测试结果实时写入acc.txt文件中
        print('Saving model......')
        torch.save(net.state_dict(), '%s/final_net.pth' % (args.output))
        print("Training Finished, TotalEPOCH=%d" % EPOCH)

