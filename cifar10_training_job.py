import numpy as np
import matplotlib.pyplot as plt
import sagemaker
import torchvision.transforms as transforms
import torchvision
import torch

from sagemaker.pytorch import PyTorch
from sagemaker.s3 import S3Uploader

region = sagemaker.Session().boto_region_name
print("AWS Region: {}".format(region))

role = sagemaker.get_execution_role()
print("RoleArn: {}".format(role))

sagemaker_session = sagemaker.Session()
bucket = sagemaker_session.default_bucket()
prefix = "pytorch-cnn-cifar10-example"

trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
)

print(len(trainset))

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True
)

images, labels = next(iter(trainloader))
plt.imshow(torchvision.utils.make_grid(images).permute(1, 2, 0) / 2 + 0.5);
plt.title(' '.join(trainset.classes[label] for label in labels)); 
plt.show();

inputs = S3Uploader.upload("data", "s3://{}/{}/data".format(bucket, prefix))

estimator = PyTorch(
    entry_point="cifar10.py",
    role=role,
    framework_version="1.4.0",
    py_version="py3",
    instance_count=1,
    instance_type="ml.c5.xlarge",
)

estimator.fit(inputs)

predictor = estimator.deploy(initial_instance_count=1, instance_type="ml.c5.xlarge")

testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
)

test_loader = torch.utils.data.DataLoader(
    testset,
    batch_size=4,
    shuffle=True
)

# get some test images
images, labels = next(iter(test_loader))
plt.imshow(torchvision.utils.make_grid(images).permute(1, 2, 0) / 2 + 0.5); 
plt.title(' '.join(testset.classes[label] for label in labels)); plt.show()

# print images, labels, and predictions
outputs = predictor.predict(images.numpy())

_, predicted = torch.max(torch.from_numpy(np.array(outputs)), 1)

print("Predicted:   ", " ".join("%4s" % classes[predicted[j]] for j in range(4)))

predictor.delete_endpoint()
