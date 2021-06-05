import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from model.losses import NTXentLoss

from tqdm import tqdm

class MultiSimCLR(nn.Module):
    def __init__(self, classes):
        # def __init__(self, base_model, out_dim, num_classes, net_name):
        super(MultiSimCLR, self).__init__()

        model = models.resnet18(pretrained=True)
        self.feature_in = 512
        self.feature_mid = 128
        self.encoder = nn.Sequential(*list(model.children())[:-1])
        self.h = nn.Sequential(nn.Linear(in_features=self.feature_in, out_features=self.feature_mid, bias=True),
                               nn.ReLU())

        # projection MLP
        self.projetion = nn.Sequential(
            nn.Linear(in_features=self.feature_mid, out_features=self.feature_mid, bias=True),  # 0
            nn.ReLU(),
            nn.Linear(in_features=self.feature_mid, out_features=self.feature_mid, bias=True),  # 3
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.feature_mid, out_features=2, bias=True),  # 3
        )
        # self.fc = nn.Sequential(
        #                 nn.Linear(in_features=self.feature_in, out_features=2, bias=True),  # 04
        #             )

    def forward(self, x):
        enc_out = self.encoder(x)
        h_proj = self.h(enc_out.view(-1, self.feature_in))
        simclr_out = self.projetion(h_proj)
        ce_out = self.fc(h_proj)
        return h_proj, simclr_out, ce_out

features = None

def _get_features_hook(self, input, output):
    global features
    features = output

class SimCLR(object):
    def __init__(self, args, classes, max_epoch=30):
        self.args = args
        self.num_class = 2
        self.device = self.args.device
        self.max_epoch = max_epoch
        self.model = MultiSimCLR(classes).to(self.device)
        # batch size needed..
        self.simclr_criterion = NTXentLoss(self.args, self.device, args.batch_size, temperature=args.temperature, use_cosine_similarity=True)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model.to(self.device)

    def _get_trg_layer(self):
        if self.args.loss == 'ce_with_mu':
            trg_layer = self.model.fc[1]
        elif (self.args.loss == 'ce_simclr_with_mu') | (self.args.loss == 'ce_simclr_negative_with_mu'):
            trg_layer = self.model.h[1]
        else:
            raise NotImplementedError
        return trg_layer

    def _mu_loss(self, in_feature, label):
        in_feature = in_feature.to(torch.float32)
        zero = torch.FloatTensor([0.0]).to(in_feature.device)

        class_mu = torch.cat(
            [torch.mean(in_feature[label == l], dim=0).unsqueeze(0) for l in range(0, self.num_class)],
            dim=0)
        pairwise_distance = torch.cdist(class_mu, class_mu, p=2)
        distance = pairwise_distance[
            torch.tril(torch.ones_like(pairwise_distance), diagonal=-1) == 1]  # pick lower triangular

        if self.args.flag_margin:
            term2 = self.args.M - distance
            term2 = torch.where(term2 >= zero, term2, zero)
        else:
            term2 = -distance
            term2 *= self.args.lambda_loss_mu

        loss_mu = torch.mean(term2)
        return loss_mu

    def _step(self, model, xis, xjs, labels, flag_train = True):
        # get the representations and the projections
        if 'with_mu' in self.args.loss:
            trg_layer = self._get_trg_layer()
            handle = trg_layer.register_forward_hook(_get_features_hook)
        ris, zis, outis = model(xis)  # [N,C]
        if 'with_mu' in self.args.loss:
            handle.remove()
        # get the representations and the projections
        rjs, zjs, outjs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)
        if flag_train:
            simclr_loss = self.simclr_criterion(zis, zjs)
#             simclr_loss *= 0.01*(self.args.lambda_loss_mu)
            simclr_loss *= self.args.lambda_loss_var
            org_loss = self.criterion(outis, labels)
            loss = simclr_loss + org_loss
            if 'with_mu' in self.args.loss:
                mu_loss = self._mu_loss(features, labels)
                loss += mu_loss
            else:
                mu_loss = None
        else:
            simclr_loss, org_loss, mu_loss, loss = None, None, None, None

        return simclr_loss, org_loss, mu_loss, loss, outis

    def train(self, dataloader, summary):

        train_loader, valid_loader, test_loader= dataloader

        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-5)
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)  # original

        model_pth = os.path.join(self.args.result_path, "models")
        log_pth = os.path.join(self.args.result_path, "logs")
        os.makedirs(model_pth, exist_ok=True)
        os.makedirs(log_pth, exist_ok=True)

        max_acc = -1
        best_loss = 10000000
        best_acc_epoch = -1
        best_loss_epoch = -1

        running_loss, running_loss_simclr, running_loss_ce, running_loss_mu = 0.0, 0.0, 0.0, 0.0
        iter_for_summary = 0.0
        total = 0
        correct = 0

        for epoch_counter in range(self.max_epoch):
            # simclr_losses = 0
            # org_losses = 0
            # mu_losses = 0
            # losses = 0
            for ii, input in enumerate(tqdm(train_loader)):
                iter_for_summary+=1
                (xis, xjs), labels = input
                optimizer.zero_grad()

                xis = xis.to(self.device)
                xjs = xjs.to(self.device)
                labels = labels.to(self.device)

                simclr_loss, org_loss, mu_loss, loss, outputs = self._step(self.model, xis, xjs, labels)

                # simclr_losses += simclr_loss
                # org_losses += org_loss
                # if 'with_mu' in self.args.loss:
                #     mu_losses += mu_loss
                # losses += loss.item()
                loss.backward()

                running_loss_ce += org_loss.item()
                running_loss_simclr += simclr_loss.item()
                if 'with_mu' in self.args.loss:
                    running_loss_mu += mu_loss.item()
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                optimizer.step()

            # if scheduler is not None:
            #     scheduler.step()

            if (iter_for_summary != 0) & (iter_for_summary % 10 == 0):
                summary.add_scalar('loss/loss_tot', (running_loss / 10), iter_for_summary)
                summary.add_scalar('loss/loss_ce', (running_loss_ce / 10), iter_for_summary)
                summary.add_scalar('loss/loss_simclr', (running_loss_simclr / 10), iter_for_summary)
                if 'with_mu' in self.args.loss:
                    summary.add_scalar('loss/loss_mu', (running_loss_mu / 10), iter_for_summary)

                running_loss, running_loss_simclr, running_loss_ce, running_loss_mu = 0.0, 0.0, 0.0, 0.0
                train_acc = 100 * (correct / total)
                summary.add_scalar('acc/train_accuracy', train_acc, iter_for_summary)
                correct = 0
                total = 0

            # print("End of epoch {}, Byol Loss {:.4f}, Org Loss {:.4f}, Total Loss {:.4f}  ".format(epoch_counter,
            #                                                                                        simclr_losses/(ii+1), org_losses/(ii+1),
            #                                                                                        losses/(ii+1)))
            # validate the model if requested

            val_loss, val_acc = self._validate(self.model, valid_loader, flag_test=False)

            summary.add_scalar('acc/val_accuracy', val_acc, epoch_counter)

            if val_acc > max_acc:
                max_acc = val_acc
                best_file = os.path.join(self.args.result_path, "models", "best_{}.pt".format(self.args.seed))
                torch.save(self.model.state_dict(), best_file)

            if val_loss < best_loss:
                best_loss = val_loss
                best_file = os.path.join(self.args.result_path, "models", "best_loss_{}.pt".format(self.args.seed))
                torch.save(self.model.state_dict(), best_file)

        ###############
        # test
        ###############
        log_file = os.path.join(self.args.result_path, "logs", "test_log_{}.txt".format(self.args.seed))
        with open(log_file, "w") as file:
            file.write("")

        _, test_acc = self._validate(self.model, test_loader, flag_test=True)

        table = 'Test acc: {}'.format(test_acc)
        with open(log_file, "a") as file:
            file.write(table)

    def _validate(self, model, valid_loader, flag_test = False):
        total = 0
        correct = 0
        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            for ii, input in enumerate(tqdm(valid_loader)):
                (xis, xjs), labels = input
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)
                labels = labels.to(self.device)

                simclr_loss, org_loss, mu_loss, loss, outputs = self._step(self.model, xis, xjs, labels, flag_train = ~flag_test)
                if not flag_test:
                    valid_loss += loss.item()

                predicted_value, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                counter += 1
            if not flag_test:
                valid_loss /= counter
            acc = correct / total
            if not flag_test:
                print('Val Accuracy: {}, Val pred Loss: {} '.format(100 * acc, valid_loss))
            else:
                print('Test Accuracy: {} '.format(100 * acc))
        model.train()
        return valid_loss, acc * 100

    def predict(self, input):
        total = 0
        correct = 0
        # validation steps
        with torch.no_grad():
            self.model.eval()
            input = input.to(self.device)
            _,_, outputs = self.model(input)
        predicted_value, predicted = torch.max(outputs.data, dim=-1)

        return predicted, predicted_value