with open("cifar100_bright/x.npy", "rb") as f:
    import numpy as np
    train_x = np.load(f)
    
with open("cifar100_bright/x_3.npy", "rb") as f:
    import numpy as np
    train_x_2 = np.load(f)

breakpoint()