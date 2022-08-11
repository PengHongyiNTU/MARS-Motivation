from tqdm import tqdm
import wandb
import torch


def local_train(id, local_model, local_loader, optimizer, 
                    criterion, local_epoch, device):
 
    local_model.to(device)
    local_model.train()
    print(f'Client {id} was selected. Start Local Training.')
    for e in range(local_epoch):
        running_loss = 0.0
        pbar = tqdm(local_loader)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = local_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix_str(f'Epoch: {e} Loss: {running_loss:.3f} ')
    return local_model.state_dict(), running_loss
        
    
def aggregate(params_dict):
    with torch.no_grad():
        # Average the parameters in params_dict
        num_clients = len(params_dict)
        global_params = next(iter(params_dict.values()))
        for i in range(1, num_clients):
            params = list(params_dict.values())[i]
            for param_name in params.keys():
                global_params[param_name] += params[param_name]
        for param_name in global_params.keys():
            dtype = global_params[param_name].dtype
            global_params[param_name] = (global_params[param_name] / num_clients).type(dtype)
        return global_params
        
    
if __name__ == "__main__":
    from torch import nn 
    model = torch.nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 1))
    x = torch.zeros(100, 10)
    y = torch.ones(100, 1)
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01) 
    criterion = nn.CrossEntropyLoss()
    from Utils import get_best_gpu
    device = get_best_gpu()
    local_train(
        0, model, loader, optimizer, criterion, local_epoch=10, device=device, log_freq=10
    )
    params_dict = {
        0: model.state_dict(),
        1: model.state_dict(),
    }
    global_params = aggregate(params_dict)
    for param_name in global_params.keys():
        diff = torch.norm(global_params[param_name] - model.state_dict()[param_name])
        print(f'{param_name}: {diff}')
    # All good
    