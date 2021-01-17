import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def draw_base_loss(file_path):
    file = open(file_path, "r")
    loss = []
    while True:
        data_line = file.readline()
        if data_line == "":
            break
        x = [a for a in data_line.split(" ")]
        loss.append(float(x[9]))
    x = [i for i in range(len(loss))]
    plt.plot(x, loss)

    plt.title("RNN模型Loss", fontsize=24)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.show()


def draw_gan_loss(file_path):
    file = open(file_path, "r")
    refiner_loss = []
    discriminative_loss = []
    while True:
        data_line = file.readline()
        if data_line == "":
            break
        x = [a for a in data_line.split(" ")]
        refiner_loss.append(float(x[9]))
        discriminative_loss.append(float(x[13]))
    x = [i for i in range(len(refiner_loss))]
    plt.plot(x, refiner_loss, label="Refiner Loss")
    plt.plot(x, discriminative_loss, label="Discriminative Loss")

    plt.title("GAN模型Loss", fontsize=24)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend(prop={"size": 16})
    plt.show()


if __name__ == '__main__':
    draw_base_loss("E:/rnn_log.txt")
