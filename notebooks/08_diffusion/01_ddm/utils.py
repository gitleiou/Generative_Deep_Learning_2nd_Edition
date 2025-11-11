import matplotlib.pyplot as plt

#value=0~1, （639，64,64,64,3）的数据集中采样第一个batch
def sample_batch(dataset): 
    batch = dataset.take(1).get_single_element() #得到一个batch的数据
    if isinstance(batch, tuple):
        batch = batch[0] #如果有label等特征，这里只需要第[0]个特征，即图像数据
    return batch.numpy() #tf张量-->numpy数组


def display(
    images, n=10, size=(20, 3), cmap="gray_r", as_type="float32", save_to=None
):
    """
    Displays n random images from each one of the supplied arrays.
    """
    if images.max() > 1.0:
        images = images / 255.0
    elif images.min() < 0.0:
        images = (images + 1.0) / 2.0

    plt.figure(figsize=size)
    for i in range(n):
        _ = plt.subplot(n//10, 10, i + 1)
        plt.imshow(images[i].astype(as_type), cmap=cmap)
        plt.axis("off")

    if save_to:
        plt.savefig(save_to)
        print(f"\nSaved to {save_to}")

    plt.show()
