import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from scipy.stats import norm

class MmoeGclocModule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_tasks, num_experts, num_epochs, batch_size, lr):
        super(MmoeGclocModule, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.threshold = 0.5
        self.eps = 2
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.zero = torch.tensor(0.00000001, dtype = torch.float32, device = self.device)
        self.experts = nn.ModuleList([
                        nn.Sequential(
                            nn.Linear(input_size, hidden_size, dtype = torch.float32, device = self.device),
                            nn.ReLU(),
                            nn.Linear(hidden_size, hidden_size, dtype = torch.float32, device = self.device),
                            nn.ReLU(),
                            nn.Linear(hidden_size, hidden_size, dtype = torch.float32, device = self.device),
                            nn.ReLU(),
                            nn.Linear(hidden_size, hidden_size, dtype = torch.float32, device = self.device),
                            nn.ReLU(),
                            nn.Linear(hidden_size, hidden_size, dtype = torch.float32, device = self.device),
                            nn.ReLU(),
                            nn.Dropout(p = 0.3, inplace = False)
                        )
                        for _ in range(num_experts)
        ])
        self.gates = nn.ModuleList([
                      nn.Sequential(
                        nn.Linear(input_size, num_experts, dtype = torch.float32, device = self.device),
                        nn.Softmax()
                      )
                      for _ in range(num_tasks)
        ])
        self.towers = nn.ModuleList([
                       nn.Sequential(
                        nn.Linear(hidden_size, output_size[i], dtype = torch.float32, device = self.device)
                       )
                       for i in range(len(output_size))
        ])
    
    def forward(self, X):
        X = torch.tensor(X, dtype = torch.float32, device = self.device)
        expert_out = [self.experts[i](X) for i in range(self.num_experts)]
        gate_out = [self.gates[i](X) for i in range(self.num_tasks)]
        gated_out = []
        for i in range(self.num_tasks):
            if expert_out[0].dim() > 1:
                tmp = [expert_out[j] * torch.unsqueeze(gate_out[i][:, j], dim = -1) for j in range(self.num_experts)]
            else:
                tmp = [expert_out[j] * gate_out[i][j] for j in range(self.num_experts)]
            tmp = sum(tmp) 
            gated_out.append(tmp)
        task_out = []
        for i in range(self.num_tasks):
            task_out.append(self.towers[i](gated_out[i]))
        return task_out
    
    def cross_loss(self, pred, label):
        label = torch.tensor(label, dtype = torch.long, device = self.device)
        cross_loss = nn.CrossEntropyLoss()
        if pred[2] is None:
            loss = cross_loss(pred[0], label[:, 0]) + cross_loss(pred[1], label[:, 1])
        else:
            loss = cross_loss(pred[0], label[:, 0]) + cross_loss(pred[1], label[:, 1]) + cross_loss(pred[2], label[:, 2])
        return torch.mean(loss)
    
    def gc_loss(self, prob_maps, label):
        label = torch.tensor(label, dtype = torch.float32, device = self.device)
        target_maps = []
        for n in range(len(prob_maps)):
            loc = label[n, : 2].cpu().numpy()
            scale = 1.
            target_map = np.array(
                [[norm.pdf(i, loc[0], scale) * norm.pdf(j, loc[1], scale) for j in range(self.output_size[1])] for i in range(self.output_size[0])]
            )
            target_map[target_map <= self.zero.cpu().numpy()] = self.zero.cpu().numpy()
            target_map = target_map / np.sum(target_map)
            target_maps.append(target_map)
        target_maps = torch.tensor(np.array(target_maps), dtype = torch.float32, device = self.device)
        target_maps = torch.where(target_maps <= self.zero, self.zero, target_maps)

        loss = F.kl_div(prob_maps.log(), target_maps, reduction = 'batchmean')
        return loss
                
    def data_iter(self, data, label):
        indices= torch.randperm(data.shape[0])
        num_step = data.shape[0] // self.batch_size
        for i in range(num_step):
            b = i * self.batch_size
            e = b + self.batch_size
            yield data[indices[b : e]], label[indices[b : e]]
    
    def fit(self, data, label):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr = self.lr)
        torch.autograd.set_detect_anomaly(True)
        for i in range(self.num_epochs):
            err = []
            for x, y in self.data_iter(data, label):
                pred = self.forward(x)
                pro_maps = self.prob_map(pred)
                l = self.gc_loss(pro_maps, y)
                err.append(l.item())
                print('loss: ', l.item())
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
            err = np.mean(err)
            print('The loss of %d-th epoch is: %.2f' % (i, err))
    
    def predict_test(self, X, y):
        return self.cluster_predict_test(X, y)

    def predict(self, X):
        return self.cluster_predict(X)
    
    def cluster_predict_test(self, X, y):
        self.eval()
        with torch.no_grad():
            pred = self.forward(X)
            prob_maps = self.prob_map(pred)
            l = self.gc_loss(prob_maps, y)
            clusters = self.dbscan(prob_maps)
            pred = self.cluster_summary(clusters)
            return pred, prob_maps, clusters, l

    def cluster_predict(self, X):
        self.eval()
        with torch.no_grad():
            pred = self.forward(X)
            prob_maps = self.prob_map(pred)
            clusters = self.dbscan(prob_maps)
            pred = self.cluster_summary(clusters)
            return pred
    
    def cluster_summary(self, clusters):
        pred = []
        for i in range(len(clusters)):
            max_cluster = self.find_max_cluster(clusters[i])
            x, y = 0., 0.
            s = 0.
            for j in range(len(clusters[i][max_cluster])):
                s += clusters[i][max_cluster][j][2]
            for j in range(len(clusters[i][max_cluster])):
                x += (clusters[i][max_cluster][j][0] * (clusters[i][max_cluster][j][2]) / s)
                y += (clusters[i][max_cluster][j][1] * (clusters[i][max_cluster][j][2]) / s)
            z = 0.
            pred.append([x, y, z])
        return np.array(pred)
    
    def find_max_cluster(self, clusters):
        max_n, max_p = 0, 0.
        index = None
        for i in range(len(clusters)):
            if len(clusters[i]) > max_n:
                max_n = len(clusters[i])
                p = 0.
                for j in range(len(clusters[i])):
                    p += clusters[i][j][2]
                max_p = p
                index = i
            elif len(clusters[i]) == max_n:
                p = 0.
                for j in range(len(clusters[i])):
                    p += clusters[i][j][2]
                if p > max_p:
                    max_p = p
                    index = i
        return index
    
    def dbscan(self, prob_maps):
        clusters = []
        for i in range(prob_maps.shape[0]):
            clusters.append(self._dbscan(prob_maps[i]))
        return clusters
    
    def _dbscan(self, prob_map):
        clusters = []
        threshold = torch.median(prob_map)
        _prob_map = prob_map.clone().detach()
        for x in range(_prob_map.shape[0]):
            for y in range(_prob_map.shape[1]):
                if _prob_map[x, y] >= threshold:
                    cluster = []
                    neighbors = [(x, y, _prob_map[x, y].cpu().item())]
                    while len(neighbors) > 0:
                        _neighbors = []
                        for i in range(len(neighbors)):
                            _prob_map[neighbors[i][0], neighbors[i][1]] = 0.
                            cluster.append(neighbors[i])
                            xb = 0 if neighbors[i][0] - self.eps < 0 else neighbors[i][0] - self.eps
                            xe = _prob_map.shape[0] - 1 if neighbors[i][0] + self.eps >= _prob_map.shape[0] \
                                 else neighbors[i][0] + self.eps
                            yb = 0 if neighbors[i][1] - self.eps < 0 else neighbors[i][1] - self.eps
                            ye = _prob_map.shape[1] - 1 if neighbors[i][1] + self.eps >= _prob_map.shape[1] \
                                 else neighbors[i][1] + self.eps
                            for _x in range(xb, xe + 1):
                                for _y in range(yb, ye + 1):
                                    if _prob_map[_x, _y] >= threshold and x != _x and y != _y:
                                        _neighbors.append((_x, _y, _prob_map[_x, _y].cpu().item()))
                                        _prob_map[_x, _y] = 0.
                        neighbors.clear()
                        neighbors = _neighbors
                    clusters.append(cluster)
        return clusters

    def prob_map(self, task_out):
        '''
        Calculate the probability map.
        NOTE: only support 2-D map up to now.
        @param task_out: the output of task towers.
        '''
        x = F.softmax(task_out[0], dim = -1)
        y = F.softmax(task_out[1], dim = -1)
        if x.dim() < 2:
            prob = torch.mm(x, y)
            total = torch.sum(prob)
            return torch.divide(prob, total)
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
            y = y.reshape((y.shape[0], 1, y.shape[1]))
            prob_maps = torch.bmm(x, y)
            prob_maps = prob_maps + self.zero
            return prob_maps
