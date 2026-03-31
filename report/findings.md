# SadTalker

## Image vs Video input
There is no reason to use video for this model:
    for face generation the SadTalker uses the first frame of the video, 
    so the result will be the same as for image;  
The only reason to use the video is for --ref_pose or --ref_eyeblink params.  
Also, in "full image" mode the SadTalker takes the first frame of the video for the background.  
Since we will remove background and use face only, we can use an image as input.  

## Params for face generation

- `--size` has only two options: 256 and 512. This is a pretty small image, so the quality will be not that good;
- `--expression 0.7` looks a bit better than default value (1.0), as the person on video moves more natural;
- `--preprocess extcrop` looks a bit better than `--preprocess crop`, as face cropped somewhat larger

## CPU + GPU

It looks like the model is not optimized for GPU on Apple Silicon.  
The source code has options like: auto, cpu and cuda.  
Cuda is not supported on Apple Silicon. Instead of cuda 'mps' should be used which is not supported by the model.

```python
    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda"
    else:
        args.device = "cpu"
```