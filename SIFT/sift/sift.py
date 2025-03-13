import torch
import torch.nn as nn
import numpy as np
import random

class SIFT():
    def __init__(self, model, sparse_module, sparse_rate, exception=[], grad_acc=1, gradient_checkpointing=False, random_indices=False) -> None:
        self.model = model
        self.total_num = 0
        self.gradient_checkpointing = gradient_checkpointing
        
        #self.r = r
        self.sparse_rate = sparse_rate

        ## Parameters need to be trained sparsely
        self.sparse_module = sparse_module
        ## Parameters need to be trained normally 
        self.exception = exception
        
        ## Mapping: Parameter --> Sparse Parameter
        self.sparse_mapping = dict()
        ## For convenience, we record the gradient accumulation step for each parameter.
        self.grad_acc_count = dict()
        self.grad_acc = grad_acc
        
        self.if_get_idx = dict()
        self.record = dict()

        self.random_indices = random_indices
        
        ## Record all the trainable parameters(the initial parameter that need be updated sparsely).
        self.named_trainable_parameters_list = list()
        ## Record all the parameters that need be cacualated in optimizer.
        self.named_parameters_in_optimizer_list = list()
        
        self.register_sparse_param()
    
    def register_sparse_param(self):
        """Register a sparse param for each param that need be updated sparsely
            and get the sparse grad by using backward hook
        """
        for name, param in self.model.named_parameters():
            ## select the parameters needed to be trained sparsely
            self.total_num += param.numel()
            if any([module in name for module in self.sparse_module]):
                
                param.requires_grad = True
                ## set the number of trainable components of the parameter according to the sparse rate

                in_features, out_features = param.shape
                train_num = min(int(self.sparse_rate * param.numel()) + 1, param.numel())

                # more reliable way: number of trainable parameters is the same as in LoRA
                #train_num = (out_features + in_features) * self.r

                sparse_param = nn.Parameter(param.new_zeros(train_num), requires_grad=True)
                sparse_param.grad = sparse_param.new_zeros(train_num)
                sparse_param.train_num = train_num
                
                ## pick the components that have top-k maximun absolute values 
                #sparse_idx = torch.flatten(abs(param.data)).topk(train_num).indices
                ## Random pick
                sparse_idx = torch.tensor(random.sample(list(range(param.numel())), train_num), dtype=torch.int)
                
                sparse_param.idx = torch.stack(torch.unravel_index(sparse_idx, param.shape))
                ## help the initial parameter to find the sparse parameter 
                self.sparse_mapping[name] = sparse_param
                self.grad_acc_count[name] = 0
                self.if_get_idx[name] = False
                self.record[name] = []
                
                # ## register a backward hook to get the 'sparse' grad
                param.register_hook(self.get_sparse_grad())
                
                ## register it in the model so the framework can recognized the sparse param as a 'normal' param 
                setattr(self.model, name.replace('.', '_') + '_sparse', sparse_param)
                ## (name, p)
                self.named_trainable_parameters_list.append((name, param))
                ## (named_sparse, sparse p)
                self.named_parameters_in_optimizer_list.append((name + '_sparse', sparse_param))
                
            elif self.exception and any([item in name for item in self.exception]):
                param.requires_grad = True
                self.named_trainable_parameters_list.append((name, param))
                self.named_parameters_in_optimizer_list.append((name, param))
            elif self.gradient_checkpointing and name == next(self.model.named_parameters())[0]:
                param.requires_grad = True
            else:
                param.requires_grad = False
            
            # ## gradient caculate after backward hook, we use following codes to ensure the first sparse module can get the sparse grad as we expect.
            # ## the first parameter in the model, the last parameter in backward propagation
            # m = list(self.model.modules())[1]
            # m.register_full_backward_hook(self.get_sparse_grad())
            
    
    ## keep consistent with model.named_parameters()
    def named_trainable_parameters(self):
        return iter(self.named_trainable_parameters_list)
    
    def trainable_parameters(self):
        return iter(p for _, p in self.named_trainable_parameters_list)
    
    def named_parameters_in_optimizer(self):
        return iter(self.named_parameters_in_optimizer_list)
    
    def parameters_in_optimizer(self):
        return iter(p for _, p in self.named_parameters_in_optimizer_list)
    
    def get_trainable_num(self):
        return sum(p.numel() for p in self.parameters_in_optimizer())
    
    def print_trainable_parameters(self):
        print(
            f"trainable params: {self.get_trainable_num():,d} || all params: {self.total_num:,d} || trainable%: {100 * self.get_trainable_num() / self.total_num}"
        )
    
    def set_trainer(self, trainer):
        self.trainer = trainer
        self.grad_acc = trainer.args.gradient_accumulation_steps
        
    def get_sparse_grad(self):
        """use closure function to access the param in the backward hook
        """
        def hook(x):
            with torch.no_grad():

                for name, param in self.named_trainable_parameters():
                    if not (name in self.sparse_mapping.keys()) or param.grad is None:
                        continue

                    # print(name)
                    sparse_param = self.sparse_mapping[name]
                    grad = param.grad.to(sparse_param)

                    ## clean the init grad
                    param.grad = None
                    # if self.trainer.state.epoch ==0.:
                    if not self.if_get_idx[name]:
                        self.if_get_idx[name] = True
                        if not self.random_indices:
                            sparse_idx = torch.flatten(abs(grad)).topk(sparse_param.train_num).indices.cpu().numpy()
                        else:
                            sparse_idx = np.random.choice(param.numel(), sparse_param.train_num, replace=False)
                        if name == list(self.sparse_mapping.keys())[-1]:
                            print('switch idx')
                            # self.trainer.create_optimizer()
                            # print(sparse_param.idx)
                        sparse_param.idx = np.stack(np.unravel_index(sparse_idx, param.shape))
                        return

                    # ##if you are interested in grad proportion, uncomment following code
                    '''
                    grad_norm = torch.norm(grad).cpu().numpy().item()
                    sparse_grad_norm = torch.norm(grad[sparse_param.idx]).cpu().numpy().item()
                    grad_proportion = sparse_grad_norm/grad_norm*100
                    self.record[n].append((grad_norm, grad_proportion))
                    # print(f"{n} grad proportion: {grad_proportion:.2f}")
                    '''

                    ## get the sparse grad
                    if sparse_param.grad != None:
                        sparse_param.grad += grad[sparse_param.idx[0], sparse_param.idx[1]]
                    else:
                        sparse_param.grad = grad[sparse_param.idx[0], sparse_param.idx[1]]

                    self.grad_acc_count[name] += 1
                    if self.grad_acc_count[name] == self.grad_acc:
                        ## update the initial param sparsely
                        delta = param.data + torch.sparse_coo_tensor(sparse_param.idx, sparse_param, param.shape).to(param)
                        param.data.copy_(delta)
                        sparse_param.zero_()
                        self.grad_acc_count[name] = 0
                        # print('sparse update!')
                            
                            
        return hook
                    
            