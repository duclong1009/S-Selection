def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(42)

with open("cifar100_bright/x.npy", "rb") as f:
    import numpy as np
    train_x = np.load(f)
import torchvision
train_dataset = torchvision.datasets.CIFAR100('./cifar100', train=True, download=True)
import torchvision   

# img = Image.fromarray(np.transpose(train_x[0],(1,2,0)) * 255, "RGB")
# img.save('my.png')
# img.show()
# import numpy as np
# train_dataset = torchvision.datasets.CIFAR100('./cifar100', train=True, download=True)
# breakpoint()
img_enhancer = ImageEnhance.Brightness(train_dataset[0][0])

factor = 1
enhanced_output = img_enhancer.enhance(factor)
enhanced_output.save("output/original-image.png")

factor = 0.5
enhanced_output = img_enhancer.enhance(factor)
enhanced_output.save("output/dark-image.png")

factor = 3
enhanced_output = img_enhancer.enhance(factor)
enhanced_output.save("output/bright-image.png")