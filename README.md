ErrorMessage "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.22 GiB. GPU 0 has a total capacity of
15.77 GiB of which 959.44 MiB is free. Process 12447 has 14.83 GiB memory in use. Of the allocated memory 14.01 GiB
is allocated by PyTorch, and 472.05 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory 
is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation 
for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)"


sagemaker-user@default:~$ python train.py --data-path CUB_200_2011/images --ann-file annotations.json --batch-size 16 --epochs 1 --use-amp --image-size 512 --accumulation-steps 2
loading annotations into memory...
Done (t=0.05s)
creating index...
index created!
Starting training for 1 epochs...
Batch size: 16, Accumulation steps: 2
Effective batch size: 32
Epoch 1/1, Batch 100/736, Loss: 0.6650, Speed: 1.5 batch/s, ETA: 7.1min
Epoch 1/1, Batch 200/736, Loss: 0.3992, Speed: 1.6 batch/s, ETA: 5.5min
Epoch 1/1, Batch 300/736, Loss: 0.4225, Speed: 1.7 batch/s, ETA: 4.4min
Epoch 1/1, Batch 400/736, Loss: 0.4406, Speed: 1.7 batch/s, ETA: 3.3min
Epoch 1/1, Batch 500/736, Loss: 0.4480, Speed: 1.7 batch/s, ETA: 2.3min
Epoch 1/1, Batch 600/736, Loss: 0.4464, Speed: 1.7 batch/s, ETA: 1.3min
Epoch 1/1, Batch 700/736, Loss: 0.4521, Speed: 1.7 batch/s, ETA: 0.4min
Epoch 1 completed in 7.2 minutes
Total training completed in 0.1 hours
