from tqdm import tqdm
import torch
from Utils import compute_layerwise_diff
import copy 
def local_train(id, local_model, global_model_params, 
                local_loader, 
                optimizer, 
                criterion, local_epoch, device):
    local_model.load_state_dict(global_model_params)
    local_model.to(device)
    local_model.train()
    print(f'Client {id} was selected. Start Local Training.')
    for e in range(local_epoch):
        running_loss = 0.0
        pbar = tqdm(local_loader)
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = local_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # running_loss = running_loss / len(local_loader)
            pbar.set_postfix_str(f'Local Round: {e+1} Loss: {running_loss:.3f} ')
    local_model.cpu()
    local_params = copy.deepcopy(local_model.state_dict())
    return local_params 

    
        
    
def aggregate(params_dict):
    params_list = list(params_dict.values())
    with torch.no_grad():
        # Average the parameters in params_dict
        num_clients = len(params_list)
        global_params = {}
        for key in params_list[0].keys():
            weights = torch.stack([params_list[i][key] for i in range(num_clients)])
            avg_weights = torch.sum(weights, dim=0) / num_clients
            global_params[key] = avg_weights
            
        return global_params
        
    
if __name__ == "__main__":
    from torch import nn 
    model = torch.nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 2))
    trained_model = copy.deepcopy(model)
    x = torch.randn(100, 10)
    y = torch.ones(100, ).long()
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    optimizer = torch.optim.SGD(trained_model.parameters(), lr=0.1) 
    criterion = nn.CrossEntropyLoss()
    from Utils import get_best_gpu
    device = get_best_gpu()
    # device = 'cpu'
    local_train(
        0, trained_model, model.state_dict(), loader, 
        optimizer, criterion, local_epoch=10, device=device
    )
    params_dict = {
        0: model.state_dict(),
        1: trained_model.state_dict(),
    }
    global_params = aggregate(params_dict)
    for param_name in global_params.keys():
        diff = torch.norm(global_params[param_name] - model.state_dict()[param_name])
        print(f'{param_name}: {diff}')
        diff = torch.norm(global_params[param_name] - trained_model.state_dict()[param_name])
        print(f'{param_name}: {diff}')
    model.load_state_dict(global_params)
    # All good
    