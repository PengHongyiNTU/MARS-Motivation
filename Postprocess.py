import torch
import os
import pandas as pd
from Federate import Simulator
import copy
import numpy as np
import re
class ResultParser:
    def __init__(self, main_dir):
        self.main_dir = main_dir
        self.results_dir = os.path.join(main_dir, 'results')
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        self.cfg = torch.load(os.path.join(main_dir, 'cfg.pth'))
        self.sim = Simulator(cfg=self.cfg)
        print('Read Config ...')
        print('Preparing model and data ...')
        self.model = self.sim.make_model()
        self.local_model = copy.deepcopy(self.model)
        self.trainset, self.testset, self.client_data_idx_map = self.sim.make_data()
        torch.manual_seed(self.cfg['seed'])
        np.random.seed(self.cfg['seed'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def get_layer2compare(self, params_name):
        if self.cfg['model'] == 'ResNet18' or self.cfg['model'] == 'ResNet34':
            pat = 'layer\d+.*bn2'
        elif self.cfg['model'] == 'ResNet50' or self.cfg['model'] == 'ResNet101' or self.cfg['model'] == 'ResNet152':
            pat = 'layer\d+.*bn3'
        params_idx = [ i for i, name in enumerate(params_name) if re.match(pat, name)]
        params_name = [name for name in params_name if re.match(pat, name)]
        return params_idx, params_name
        
    def get_cka(self):
        print('Start calculating layerwise cka similarity')
        if os.path.exists(os.path.join(self.results_dir, 'cka_list.pth')):
            # self.cka_df = .read_csv(os.path.join(self.results_dir, 'cka.csv'))
            self.cka_list = torch.load(os.path.join(self.results_dir, 'cka_list.pth'))['cka_list']
            return self.cka_list
        else:
            cka_list = []
            torch.cuda.empty_cache()
            comparing_dataset = torch.utils.data.Subset(
                self.testset, np.random.choice(range(len(self.testset)), 
                                               1024, replace=False))
            comparing_loader = torch.utils.data.DataLoader(
                comparing_dataset, batch_size=512, shuffle=False)
            with torch.no_grad():
                for epoch in range(1, self.cfg['epoch']+1):
                    cka_matrix = []
                    print(f'Comparing CKA {epoch}/{self.cfg["epoch"]}')
                    global_params_dir = f'global_model_{epoch}.pth'
                    global_params = torch.load(os.path.join(self.main_dir+'/global', global_params_dir))
                    self.model.load_state_dict(global_params)
                    local_params_dir = f'local_model_{epoch}.pth'
                    local_params_dict = torch.load(os.path.join(self.main_dir+'/local', local_params_dir))
                    for client_id, params in local_params_dict.items():
                        self.local_model.load_state_dict(params)
                        from Utils import compute_cka_similarity
                        if self.cfg['model'].startswith('ResNet'):
                            """
                            params_name = list(self.model.state_dict().keys())
                            idx2compare, name2compare = self.get_layer2compare(
                                params_name
                            )
                            print(params_name)
                            print(name2compare)
                            """
                            name2compare = None
                            
                        else:
                            name2compare = None
                        result = compute_cka_similarity(self.model, 
                                                        self.local_model, 
                                                        comparing_loader, 
                                                        self.device, 
                                                        layer2compare=name2compare)
                        print(result)
                        cka = result['CKA'].numpy()
                        row = cka.diagonal().tolist()
                        print(row)
                        cka_matrix.append(row)
                    cka_list.append(np.array(cka_matrix))
                    print(cka_list)
                self.cka_list = np.array(cka_list)
                print(self.cka_list)
                torch.save({'cka_list': self.cka_list}, os.path.join(
                    self.results_dir, 'cka_list.pth'))
                return self.cka_list
        
      
        
        
    def get_l2diff(self):
        print('Start calculating layerwise weight difference')
        if os.path.exists(os.path.join(self.results_dir, 'l2diff_list.pth')):
            self.l2diff_list = torch.load(os.path.join(self.results_dir, 'l2diff_list.pth'))['l2diff_list']
            return self.l2diff_list
        else:
            l2diff_list = []
            for epoch in range(1, self.cfg['epoch']+1):
                l2_matrix = []
                global_params_dir = f'global_model_{epoch}.pth'
                global_params = torch.load(os.path.join(self.main_dir+'/global', global_params_dir))
                local_params_dir = f'local_model_{epoch}.pth'
                local_params_dict = torch.load(os.path.join(self.main_dir+'/local', local_params_dir))
                for client_id, params in local_params_dict.items():
                    from Utils import compute_layerwise_diff
                    row = list(compute_layerwise_diff(global_params, params).values())
                    l2_matrix.append(row)
                l2diff_list.append(np.array(l2_matrix))
            self.l2diff_list = np.array(l2diff_list)
            torch.save({
                'l2diff_list': self.l2diff_list}, os.path.join(self.results_dir, 'l2diff_list.pth'))
            return self.l2diff_list
        
            
                
                    
                
            
        


if __name__ == "__main__":
    parser = ResultParser(main_dir='experiments_1')
    cka_list = parser.get_cka()
    #l2_diff_list = parser.get_l2diff()
    print(cka_list)
    # print(l2_diff_list)
    