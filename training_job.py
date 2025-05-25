import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.s3 import S3Uploader


sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()


env = {
    'SAGEMAKER_REQUIREMENTS': 'requirements.txt'
}

# https://github.com/aws/deep-learning-containers/blob/master/available_images.md
estimator = PyTorch(
    entry_point="train.py",
    source_dir="source",
    role=role,
    env=env,
    framework_version="2.6.0",
    py_version="py312",
    instance_count=1,
    instance_type="ml.p3.2xlarge",
)

estimator.fit()

# Epoch 1/1
# Epoch 1, x 100, Loss: 0.1620
# completed 100 of 737 at 2025-05-25 18:51:37.254128
# Epoch 1, x 200, Loss: 0.1491
# completed 200 of 737 at 2025-05-25 18:53:04.663023
# Epoch 1, x 300, Loss: 0.1867
# completed 300 of 737 at 2025-05-25 18:54:32.175528
# Epoch 1, x 400, Loss: 0.2250
# completed 400 of 737 at 2025-05-25 18:55:59.790855
# Epoch 1, x 500, Loss: 0.0949
# completed 500 of 737 at 2025-05-25 18:57:27.643377
# Epoch 1, x 600, Loss: 0.1743
# completed 600 of 737 at 2025-05-25 18:58:55.377408
# Epoch 1, x 700, Loss: 0.1535
# completed 700 of 737 at 2025-05-25 19:00:23.029317

# 2025-05-25 19:00:58 Uploading - Uploading generated training modelEpoch 1, Loss: 0.1128, completed in 645.89 seconds
# Model saved to /opt/ml/model/model.pth
# 2025-05-25 19:00:56,166 sagemaker-training-toolkit INFO     Waiting for the process to finish and give a return code.
# 2025-05-25 19:00:56,166 sagemaker-training-toolkit INFO     Done waiting for a return code. Received 0 from exiting process.
# 2025-05-25 19:00:56,167 sagemaker-training-toolkit INFO     Reporting training SUCCESS

# 2025-05-25 19:01:16 Completed - Training job completed
# Training seconds: 1109
# Billable seconds: 1109
