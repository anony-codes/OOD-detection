import os
import torch
import torch.nn.functional as F

class SimCLR(object):

    def __init__(self, args, classes):
        self.args = args
        self.max_epoch = args.max_epoch

        self.model = MultiSimCLR(self.args, classes).to(self.device)

        self.simclr_criterion = NTXentLoss(self.device, args.batch, temperature=0.5, use_cosine_similarity=True)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.model.to(self.device)


    def _step(self, model, xis, xjs, labels):

        # get the representations and the projections
        ris, zis, outis = model(xis)  # [N,C]
        # get the representations and the projections
        rjs, zjs, outjs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        simclr_loss = self.simclr_criterion(zis, zjs)
        org_loss = self.criterion(outis, labels)
        loss = simclr_loss + org_loss

        return simclr_loss, org_loss, loss, outis

    def train(self, dataloader, save_dir):

        train_loader, valid_loader = dataloader

        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-5)  # original

        # if self.args.nn == "wideresnet":
        #     optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr,
        #                           momentum=self.args.mom, nesterov=True, weight_decay=0.0005)
        #     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
        # else:
        #     optimizer = torch.optim.Adam(self.model.parameters(), 3e-4, weight_decay=eval(self.config['weight_decay']))
        #
        #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
        #                                                            last_epoch=-1)

        max_acc = -1
        # best_loss = 10000000
        # best_acc_epoch = -1
        # best_loss_epoch = -1

        running_loss, running_loss_simclr, running_loss_ce = 0.0, 0.0, 0.0

        for epoch_counter in range(self.max_epoch):
            simclr_losses = 0
            org_losses = 0
            losses = 0
            for ii, input in enumerate(tqdm(train_loader)):
                (xis, xjs), labels = input
                optimizer.zero_grad()

                xis = xis.to(self.device)
                xjs = xjs.to(self.device)
                labels = labels.to(self.device)

                simclr_loss, org_loss, loss, outputs = self._step(self.model, xis, xjs, labels)

                simclr_losses += simclr_loss
                org_losses += org_loss
                losses += loss.item()
                loss.backward()

                running_loss_ce += org_loss.item()
                running_loss_simclr += simclr_loss.item()
                running_loss += loss.item()

                optimizer.step()

            # if scheduler is not None:
            #     scheduler.step()

            if (ii != 0) & (ii % 10 == 0):
                summary.add_scalar('loss/loss_tot', (running_loss / 10), ii)
                summary.add_scalar('loss/loss_ce', (running_loss_ce / 10), ii)
                summary.add_scalar('loss/loss_simclr', (running_loss_simclr / 10), ii)

                running_loss, running_loss_simclr, running_loss_ce = 0.0, 0.0, 0.0


            # print("End of epoch {}, Byol Loss {:.4f}, Org Loss {:.4f}, Total Loss {:.4f}  ".format(epoch_counter,
            #                                                                                        simclr_losses/(ii+1), org_losses/(ii+1),
            #                                                                                        losses/(ii+1)))
            # validate the model if requested
            val_loss, val_acc = self._validate(self.model, valid_loader)

            summary.add_scalar('acc/val_accuracy', val_acc, epoch)

            if val_acc > max_acc:
                max_acc = val_acc
                best_acc_epoch = epoch_counter
                self.save_model(os.path.join(save_dir, "best_acc.pth"), self.args, optimizer)

            # if val_loss < best_loss:
            #     best_loss = val_loss
            #     best_loss_epoch = epoch_counter
            #     self.save_model(os.path.join(save_dir, "best_loss.pth"), self.args, optimizer)

    def _validate(self, model, valid_loader):
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

                simclr_loss, org_loss, loss, outputs = self._step(self.model, xis, xjs, labels)
                valid_loss += loss.item()

                predicted_value, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                counter += 1
            valid_loss /= counter
            acc = correct / total
            print('Val Accuracy: {}, Val pred Loss: {} '.format(100 * acc, valid_loss))
        model.train()
        return valid_loss, acc * 100