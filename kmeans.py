import numpy as np

def kmeans(img):

    log_img = img[0].numpy()
    [rows, cols, chls] = log_img.shape
    log_img_one = log_img[:,:,0]
    pre_img = log_img_one.reshape((rows * cols, 1))
    k=2
    list_img = []
    while len(list_img) < k:
        n = np.random.randint(0,  pre_img.shape[0],  1)
        if n not in list_img:
            list_img.append(n[0])
    pre_point = np.array([np.min(pre_img),np.max(pre_img)])
    c=0
    while True:
        distance = [np.sum(np.sqrt((pre_img - i) ** 2)   , axis=1)   for i in pre_point]
        now_point = np.argmin(distance, axis=0)
        now_piont_distance = np.array(  list([np.average(pre_img[now_point == i], axis=0) for i in range(k)])  )
        c+=0
        if np.sum(now_piont_distance - pre_point) < 1e-7 or c>50:
        # if c>50:
            break
        else:
            pre_point = now_piont_distance

    labels=now_point
    res = labels.reshape((rows, cols))
    ima = np.zeros([rows, cols, chls])
    ima[:, :, 0] = res
    return ima




