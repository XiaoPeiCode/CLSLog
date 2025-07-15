import time
from torch import nn
import torch
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
# from modules.AutoEncoder import AE, train

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(Encoder, self).__init__()
        layer_dims = [input_dim] + hidden_dims
        self.layers = nn.Sequential(
            nn.Linear(layer_dims[0], layer_dims[1]), nn.ReLU(),
            nn.Linear(layer_dims[1], layer_dims[2]), nn.ReLU(),
            nn.Linear(layer_dims[2], layer_dims[3]),  nn.ReLU())

    def forward(self, input_data):
        hidden = self.layers(input_data)
        return hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dims):
        super(Decoder, self).__init__()
        layer_dims = hidden_dims + [output_dim]
        self.layers = nn.Sequential(
            nn.Linear(layer_dims[0], layer_dims[1]),  nn.ReLU(),
            nn.Linear(layer_dims[1], layer_dims[2]),  nn.ReLU(),
            nn.Linear(layer_dims[2], layer_dims[3]),nn.Tanh()
        )


    def forward(self, input_data):
        output = self.layers(input_data)
        return output

class AE(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(AE, self).__init__()
        self.hidden_dims = hidden_dims
        self.criterion = nn.MSELoss(reduction="none")
        self.input_dim = input_dim
        # self.embedder = Embedder(vocab_size, embedding_dim)
        # self.bertembedder = BERTEmbedder()
        #
        # self.rnn = nn.LSTM(
        #     input_size=embedding_dim,
        #     hidden_size=self.hidden_size,
        #     num_layers=num_layers,
        #     batch_first=True,
        #     # bidirectional=(self.num_directions == 2),
        # )
        self.encoder = Encoder(input_dim, hidden_dims)
        # self.clustering_layer = nn.Linear(hidden_dims[-1], num_clusters)
        # Use BERTEmbedder here
        self.decoder = Decoder(input_dim, list(reversed(hidden_dims)))



    def forward(self, input_data):


        # 都不用只用AE
        representation = input_data
        encoded = self.encoder(input_data)
        decoded = self.decoder(encoded)

        # 假设 representation 可能是二维或三维
        if representation.dim() == 3:
            # 如果 representation 是三维，按照最后两个维度求平均
            pred = self.criterion(representation, decoded).mean(dim=(-1, -2))
        elif representation.dim() == 2:
            # 如果 representation 是二维，按照最后一个维度求平均
            pred = self.criterion(representation, decoded).mean(dim=-1)

        # pred = self.criterion(representation, decoded).mean(dim=(-1,-2))
        # pred = self.criterion(representation, representation).mean(dim=-1)
        # pred should be (n_sample),loss,should be (1)

        loss = pred.mean()
        # loss = pred.mean()
        # y_pred存储的是每个输入对应的loss
        return_dict = {"loss": loss,
                       "y_pred": pred,
                       "encoded":encoded,
                       "rep":encoded}
        return return_dict


def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    y_preds = []
    epoch_time_start = time.time()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        res_dict = model(inputs)
        loss = torch.mean(res_dict["loss"])
        y_pred = res_dict["y_pred"]
        loss.backward()
        optimizer.step()

        y_preds.extend(y_pred.tolist())
        total_loss += loss.item()

    epoch_loss = total_loss / len(train_loader)
    # loss_history.append_loss(epoch_loss)
    # epoch_time_elapsed = time.time() - epoch_time_start

    return epoch_loss, y_preds


def infer_AE(model, data_loader, device):
    model.eval()
    losses = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            res_dict = model(inputs)
            loss = res_dict["y_pred"]
            losses.extend(loss.tolist())
    return losses

if __name__ == '__main__':
    
    # 超参数设置
    input_dim = 784  # 假设输入是28x28的图像
    hidden_dims = [256, 128, 64]  # 编码器的隐藏层维度
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据准备（这里以随机数据为例，实际使用时替换为真实数据）
    # 假设输入数据是二维的 (n_samples, input_dim)
    n_samples = 10000
    x_data = torch.rand(n_samples, input_dim)
    y_data = torch.zeros(n_samples)  # 自编码器不需要目标值，但可以用作占位符
    dataset = TensorDataset(x_data, y_data)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 模型、优化器初始化
    model = AE(input_dim, hidden_dims).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    for epoch in range(num_epochs):
        epoch_loss, y_preds = train(model, train_loader, optimizer, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    infer(model, train_loader, device)
    # 保存模型
    torch.save(model.state_dict(), "autoencoder_model.pth")
    print("模型已保存到 autoencoder_model.pth")
        


# def offline_train(config,dataloader):
    
#     train_dataloader,test_dataloader,al_dataloader = dataloader

#     model = AE(input_dim=768, hidden_dims=config["AE"]["hidden_dims"])

#     model = model.to(device)
#     if torch.cuda.device_count() > 1:
#         print(f"torch.cuda.device_count():{torch.cuda.device_count()}")
#         model = nn.DataParallel(model)
#     print(model)

#     #
#     model_path = f'./result/model_save/{config["dataset"]["dataset_name"]}_offline.pt'
#     loss_history = LossHistory.LossHistory("./loss/")
#     optimizer = optim.Adam(model.parameters(), lr=config["AE"]["lr"])
#     epochs = config["train"]["epoch"]
#     a_ratio = config["anomaly_detection"]["ratio"]
#     c_ratio = config["contrastive"]["ratio"]

#     # train
#     for epoch in tqdm(range(config["train"]["epoch"]), desc="Training: "):
#         # logging.info(f"--------Epoch {epoch + 1}/{epochs}-------")
#         train_loss, y_preds = train(model, train_dataloader, optimizer, device)
#         loss_history.append_loss(train_loss)

#         # 获得代表向量
#         representative_vector,distances = get_representative_vector(model,train_dataloader)
#         distances = distances.tolist()
#         # distance_threshold = get_loss_threshold(distances,config["anomaly_detection"]["ratio"])
#         # detect_anomalies(model,representative_vector,test_dataloader,threshold=distance_threshold)
#         # loss_threshold = get_loss_threshold(y_preds, config["anomaly_detection"]["ratio"])
#         # loss_threshold2 = get_loss_threshold(y_preds, config["contrastive"]["ratio"])
#         # test_th(model,y_preds,test_dataloader)

#         thresholds = np.arange(0, 1.1, 0.1)
#         score,th_pre = test_rep_vector(model,test_dataloader,representative_vector,distances,thresholds)

#         print("*****test_th*******")
#         test_th(model,y_preds,test_dataloader,thresholds)
#         logging.info(
#             "Epoch {}/{}, training loss: {:.5f}".format(epoch, epochs, train_loss)
#         )
#         print("")
#     print("**********train end ************")