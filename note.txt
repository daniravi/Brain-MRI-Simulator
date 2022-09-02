# 4D-DaniNet

tensorboard --logdir /home/dravi/Desktop/CurrentWork/Alzhaimer/FaceAging/DaniNet/save_DaniNet-V2_AL/100/ --port 6006


#weights must be normalized to better understand the importance
            
# 0)similarity with the image # the sum need to be equal to 1 (low value->population average vs high value->personality)
# 1)realistic image (smaller is more realistic structures)   (realistic structure vs number of epoch)
# 1)reduce this value when artifacts appear, increase to make training on subject more sharp
# 2)smoothing in progression (0 very smooth , 1 major freedom to be different), (temporal smoothing vs progression)
# 3)pixel loss  (progression)
# 4)regional loss (progression-> reliability of progression prior)

# when G-loss is close to 0 the generator create image realistic
# when G-loss is high it creates sharp image
        