import os

epochs_grid = [10,20,40,80]
lr_grid = [1e-5,5e-5,1e-4,5e-4,1e-3]
weight_decay_grid = [5e-2,1e-2,5e-3,1e-3]
batch_grid = [3,6,12,24]
num = 0
for epoch in epochs_grid:
    for batch in batch_grid:
        for lr in lr_grid:
            for weight in weight_decay_grid:
                num += 1
                cmd = 'python train.py --num_epochs=%i --batch_size=%i ' \
                      '--weight_decay=%f --learning_rate=%f --output_dir=%i' %(epoch,batch,weight,lr,num)
                os.system(cmd)



