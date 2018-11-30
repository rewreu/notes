
## How batch normalization works


```python
import torch
m = torch.nn.BatchNorm2d(1)
inputs = torch.rand(2,1,2,2)

output = m(inputs)
print("input tensor:\n",inputs,"\n\noutputTensor:\n",output)

mu = torch.mean(inputs)
var = torch.var(inputs,unbiased=False)
dom = (var+1e-5)**0.5
print("result\n",(inputs - mu)/dom*m.weight)
```

    input tensor:
     tensor([[[[0.5603, 0.2043],
              [0.6890, 0.5484]]],
    
    
            [[[0.0924, 0.7808],
              [0.2247, 0.7926]]]]) 
    
    outputTensor:
     tensor([[[[ 0.2358, -0.9031],
              [ 0.6477,  0.1978]]],
    
    
            [[[-1.2613,  0.9417],
              [-0.8378,  0.9792]]]], grad_fn=<ThnnBatchNormBackward>)
    result
     tensor([[[[ 0.2358, -0.9031],
              [ 0.6477,  0.1978]]],
    
    
            [[[-1.2613,  0.9417],
              [-0.8378,  0.9792]]]], grad_fn=<ThMulBackward>)


## The mean and variance are calculated accross (N, H, W)
### (N, C, H, W) stands for (batch, channel, height, width)


```python
import torch
m = torch.nn.BatchNorm2d(3)
inputs = torch.rand(2,3,2,2)

output = m(inputs)
print("input tensor:\n",inputs,"\n\noutputTensor:\n",output)

mu1 = torch.mean(inputs[:,0,:,:])
var1 = torch.var(inputs[:,0,:,:],unbiased=False)
dom1 = (var1+1e-5)**0.5
print("result_1\n",(inputs[:,0,:,:] - mu1)/dom1*m.weight[0])
```

    input tensor:
     tensor([[[[0.8439, 0.5570],
              [0.0911, 0.0757]],
    
             [[0.1199, 0.0140],
              [0.4575, 0.5617]],
    
             [[0.5839, 0.6828],
              [0.6673, 0.0501]]],
    
    
            [[[0.3866, 0.5515],
              [0.0540, 0.8977]],
    
             [[0.4628, 0.4546],
              [0.4669, 0.2009]],
    
             [[0.3349, 0.2194],
              [0.5329, 0.3284]]]]) 
    
    outputTensor:
     tensor([[[[ 0.2412,  0.0731],
              [-0.1998, -0.2089]],
    
             [[-1.0942, -1.6148],
              [ 0.5668,  1.0793]],
    
             [[ 0.0497,  0.0807],
              [ 0.0759, -0.1174]]],
    
    
            [[[-0.0267,  0.0699],
              [-0.2216,  0.2727]],
    
             [[ 0.5929,  0.5526],
              [ 0.6130, -0.6957]],
    
             [[-0.0282, -0.0644],
              [ 0.0338, -0.0302]]]], grad_fn=<ThnnBatchNormBackward>)
    result_1
     tensor([[[ 0.2412,  0.0731],
             [-0.1998, -0.2089]],
    
            [[-0.0267,  0.0699],
             [-0.2216,  0.2727]]], grad_fn=<ThMulBackward>)

