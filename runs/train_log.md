# 训练记录

- exp18: 使用yolov5x对`depth`进行训练，效果非常差，map不到0.2
- exp19: 使用yolov5l对`infrared`进行训练，还好，快到0.4
- exp20: 使用yolov5l对`color`进行训练，
- exp21: 模型写错了, try again
- exp22: 使用yolov5x, epoch=100, 最经典的训练方案，当作一个baseline
- exp23: 使用yolov5x, epoch=100, 对infrared进行训练，希望下一步可以进行模型融合
- exp31: 使用mixup: 它给的框架代码居然有bug... 
	- 中间调好了mixup, 调高了mixup的超参数，调低了一点点马赛克, 希望可以有效果，如果效果不佳，下一次使用albumentations手动融合ball和light以及手动放大ball的照片，增强训练
> mixup 居然真的可以有效提高ball和light的预测准确率
- exp32: 调高了mixup, 调高了copy-paste(不知道能干啥..), 调高了epoch数量, batch_size 调到16
- exp33: 32又提高了，但我感觉跟mixup关系不大，因为ball, sign预测准确率的提高并不明显，我试着调低lrf，然后再次调高epoch到150，看看效果
