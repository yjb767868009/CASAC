import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def draw_pretrain_loss(file_path):
    file = open(file_path, "r")
    train_loss = []
    test_loss = []
    while True:
        data_line = file.readline()
        if data_line == "":
            break
        x = [a for a in data_line.split(" ")]
        if "Loss" not in x:
            continue
        train_loss.append(float(x[9]))
        test_loss.append(float(x[13]))
    x = [i for i in range(len(train_loss))]
    plt.plot(x, train_loss, label="train loss")
    plt.plot(x, test_loss, label="test loss")

    plt.title("Pretrain Loss", fontsize=24)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend(fontsize=16)
    plt.show()

def draw_train_loss(file_path):
    file = open(file_path, "r")
    train_loss = []
    test_loss = []
    while True:
        data_line = file.readline()
        if data_line == "":
            break
        x = [a for a in data_line.split(" ")]
        if "Loss" not in x:
            continue
        train_loss.append(float(x[9]))
        test_loss.append(float(x[13]))
    x = [i for i in range(len(train_loss))]
    plt.plot(x, train_loss, label="train loss")
    plt.plot(x, test_loss, label="test loss")

    plt.title("Train Loss", fontsize=24)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend(fontsize=16)
    plt.show()

if __name__ == '__main__':
    draw_pretrain_loss("D:/NSM/log1.txt")
    draw_train_loss("D:/NSM/log2.txt")
