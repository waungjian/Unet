# 模型拟合的训练、测试函数
import torch.utils.data
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(dl,model,loss_fn,opt):
    num_batch = len(dl)
    size = len(dl.dataset)
    train_loss,train_acc = 0,0
    for img,anno in dl:
        x,y = img.to(device),anno.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred,y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            train_loss += loss.item()
            y_pred = torch.argmax(y_pred,dim=1)
            train_acc += (y_pred == y).type(torch.float).sum().item()
    train_loss /= num_batch
    train_acc /= (size*128*128)

    return train_loss,train_acc


def test(dl, model, loss_fn):
    num_batch = len(dl)
    size = len(dl.dataset)
    test_loss, test_acc = 0, 0
    for img, anno in dl:
        x, y = img.to(device), anno.to(device)
        with torch.no_grad():
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            y_pred = torch.argmax(y_pred, dim=1)
            test_acc += (y_pred == y).type(torch.float).sum().item()
    test_loss /= num_batch
    test_acc /= (size * 128 * 128)

    return test_loss, test_acc

def fit(Epoch,train_dl,test_dl,model,loss_fn,opt):
    model_state_dict = copy.deepcopy(model.state_dict())
    best_acc = 0
    train_loss = []
    test_loss = []
    train_acc =[]
    test_acc = []
    for i in range(Epoch):
        epoch_train_loss,epoch_train_acc = train(train_dl,model,loss_fn,opt)
        epoch_loss,epoch_acc = test(test_dl,model,loss_fn)
        if epoch_acc > best_acc:
            model_state_dict = model.state_dict()
            best_acc = epoch_acc
            torch.save(model_state_dict,"net_param.pth")
        print("Epoch {}: train_loss : {} train_acc : {}"
              .format(i,epoch_train_loss,epoch_train_acc))
        print("Epoch {}: test_loss : {} test_acc : {}"
              .format(i, epoch_loss, epoch_acc))
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)
        test_loss.append(epoch_loss)
        test_acc.append(epoch_acc)

    return train_loss,test_loss,train_acc,test_acc




