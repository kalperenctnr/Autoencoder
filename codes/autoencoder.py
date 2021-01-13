import h5py as h5
import numpy as np
from matplotlib import pyplot as plt

file = h5.File("assign3_data1.h5", 'r')
data = np.array(file["data"])  # 10240x3x16x16

# Preprocessing data in order to accerelerate the learning process
n = np.shape(data)[0]

# Gray scaling it by using luminosity model
gray_sc = 0.2126 * data[:, 0, :, :] + 0.7152 * data[:, 1, :, :] + 0.0722 * data[:, 2, :, :]
print(data[0][0][0][0])
for gray_img in gray_sc:
    # Normalization
    gray_img = gray_img - np.mean(gray_img)

    std_third = np.std(gray_img) * 3

    # Clipping to [-3std, 3std] interval
    greater_third = gray_img > std_third
    lower_third = gray_img < -std_third

    gray_img[greater_third] = std_third
    gray_img[lower_third] = -std_third

    # Mapping [-3std, 3std] to [0.1, 0.9]
    gray_img = (0.4 / (std_third + 0.00000000001)) * gray_img + 0.5

for i in range(20):
    plt.figure()

    for j in range(10):
        # Plotting RGB images randomly
        rand = np.random.randint(0, n)
        plt.subplot(5, 4, 2 * j + 1)
        plt.title("RGB Img: " + str(i * 10 + (j + 1)))
        img = np.swapaxes(data[rand], 0, 2)
        img = np.swapaxes(img, 0, 1)
        plt.axis("off")
        plt.imshow(img)

        # PLotting gray scaled image
        plt.subplot(5, 4, 2 * (j + 1))
        plt.title("Gray Img: " + str(i * 10 + (j + 1)))
        plt.axis("off")
        plt.imshow(gray_sc[rand], cmap='gray')
    plt.tight_layout()


def display_weights(weights):
    num = weights.shape[-1]
    reshaped_to_2d = np.reshape(weights, (num, 16, 16))
    row = np.ceil(np.sqrt(num))
    plt.figure()
    for i in range(num):
        plt.subplot(row, row, i + 1)
        plt.axis("off")
        plt.imshow(reshaped_to_2d[i], cmap='gray')
    plt.tight_layout()
    return


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sig_der(x):
    sig = sigmoid(x)
    return (sig) * (1 - sig)


L_hid = 64
data = gray_sc


def aeCost(w_e, data, params):
    (w_1, w_2, b_1, b_2) = w_e
    L_1, L_hid, lam, beta, p = params

    # Forward Pass
    x = data.reshape(10240, 256)
    d = x
    v_1 = x @ w_1 + b_1
    a_1 = sigmoid(v_1)
    v_2 = a_1 @ w_2 + b_2
    o = sigmoid(v_2)
    pe = np.mean(a_1, axis=0)

    # Backward Pass
    local_grad_2 = (d - o) * sig_der(v_2)
    dw_2 = -a_1.T @ local_grad_2 + lam * w_2
    db_2 = -np.ones((1, 10240)) @ local_grad_2

    transient = (w_2 @ local_grad_2.T).T
    local_grad_1 = sig_der(v_1) * transient
    dKL_trans = (-p * np.reciprocal(pe) + (1 - p) * np.reciprocal(1 - pe))
    dKL = beta * sig_der(v_1) * dKL_trans
    dw_1 = -x.T @ local_grad_1 + lam * w_1 + -x.T @ dKL
    db_1 = -np.ones((1, 10240)) @ local_grad_1

    kl_term = beta * np.sum(p * (np.log(p) - np.log(pe)) + (1 - p) * (np.log(1 - p) - np.log(1 - pe)))
    J = (1 / (2 * 10240)) * np.sum((o - d) * (o - d)) + (lam / 2) * (np.sum(w_1 * w_1) + np.sum(w_2 * w_2)) + kl_term

    J_grad = (dw_1, dw_2, db_1, db_2)
    return J, J_grad


def predict(data, w_e, params):
    (w_1, w_2, b_1, b_2) = w_e
    L_1, L_hid, lam, beta, p = params

    # Forward Pass
    x = data.reshape(10, 256)
    v_1 = x @ w_1 + b_1
    a_1 = sigmoid(v_1)
    v_2 = a_1 @ w_2 + b_2
    o = sigmoid(v_2)
    return np.reshape(o, (10, 16, 16))


def fit(data, beta=0.1, p=0.2, lam=0.0005, L_hid=64, l_r=0.1, epoch=500):
    w0 = np.sqrt(6 / (256 + L_hid))
    N = 10240
    # Weight Initializations by uniform xavier initialization
    w_1 = np.random.uniform(-w0, w0, (256, L_hid))
    w_2 = np.random.uniform(-w0, w0, (L_hid, 256))
    b_1 = np.random.uniform(-w0, w0, (1, L_hid))
    b_2 = np.random.uniform(-w0, w0, (1, 256))

    cost = []
    for m in range(epoch):
        w_e = (w_1, w_2, b_1, b_2)
        params = (256, 64, lam, beta, p)
        J, J_grad = aeCost(w_e, data, params)

        (dw_1, dw_2, db_1, db_2) = J_grad
        w_1 -= (1 / N) * l_r * dw_1
        b_1 -= (1 / N) * l_r * db_1
        w_2 -= (1 / N) * l_r * dw_2
        b_2 -= (1 / N) * l_r * db_2
        cost.append(J)

    plt.plot(cost)
    plt.show()
    return w_e, params


w_e, params = fit(data, beta=0.2, p=0.5, epoch=2000, l_r=0.05)
a = np.random.randint(0, 10240, size=10)
test = data[a]
output = predict(test, w_e, params)
fig, axs = plt.subplots(5, 4)
for i in range(5):
    for j in range(2):
        axs[i, 2 * j].imshow(output[2 * i + j], cmap='gray')
        axs[i, 2 * j].set_title("Input: " + str(2 * i + j + 1))
        axs[i, 2 * j].axis("off")

        axs[i, 2 * j + 1].imshow(test[2 * i + j], cmap='gray')
        axs[i, 2 * j + 1].set_title("Output: " + str(2 * i + j + 1))
        axs[i, 2 * j + 1].axis("off")
plt.tight_layout()
(w_1, w_2, b_1, b_2) = w_e
display_weights(w_1)