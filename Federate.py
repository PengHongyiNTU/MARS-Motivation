import yaml # for reading the config fil
import wandb

class Simulator:
    def __init__(self, cfg_path):
        self.cfg = yaml.safe_load(cfg_path)
        self.init_wandb()
    
    def run(self, project_name):
        self.cfg = wandb.config
        with wandb.init(project=project_name):
            model = self.cfg['model'] 
            self.make_models()
            self.make_data()
            self.train()
            self.test()
            self.evaluate()
            self.aggregate()
            self.save()
    
    def init_wandb(self):
        wandb.login()
        
    
    def make_models(self):
        pass

    def make_data(self):
        pass
    
    

