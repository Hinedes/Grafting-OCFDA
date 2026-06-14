import torch
import torch.nn as nn


def _random_flat_indices(num_elements: int, train_num: int, device: torch.device) -> torch.Tensor:
    if train_num <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if train_num >= num_elements:
        return torch.arange(num_elements, dtype=torch.long, device=device)

    selected = torch.empty(0, dtype=torch.long, device=device)
    while selected.numel() < train_num:
        remaining = train_num - selected.numel()
        sample_count = min(num_elements, max(remaining + remaining // 10 + 16, remaining))
        sample = torch.randint(0, num_elements, (sample_count,), dtype=torch.long, device=device)
        selected = torch.unique(torch.cat([selected, sample]))
    return selected[:train_num]


def _flat_to_parameter_indices(flat_indices: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    return torch.stack(torch.unravel_index(flat_indices.to(torch.long), shape))

class SIFT():
    def __init__(self, model, sparse_module, sparse_rate: float, exception=[], grad_acc=1, gradient_checkpointing=False, random_indices=False) -> None:
        assert 0.0 <= sparse_rate <= 1.0, "sparse_rate should be a ratio between 0 and 1"

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
                sparse_idx = _random_flat_indices(param.numel(), train_num, param.device)
                sparse_param.idx = _flat_to_parameter_indices(sparse_idx, param.shape)
                ## help the initial parameter to find the sparse parameter 
                self.sparse_mapping[name] = sparse_param
                self.grad_acc_count[name] = 0
                self.if_get_idx[name] = False
                self.record[name] = []
                
                # ## register a backward hook to get the 'sparse' grad
                param.register_hook(self.get_sparse_grad(name, param))
                
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
        
    def get_sparse_grad(self, name, param):
        """use closure function to access the param in the backward hook
        """
        def hook(grad):
            with torch.no_grad():
                sparse_param = self.sparse_mapping[name]
                grad = grad.to(device=sparse_param.device, dtype=sparse_param.dtype)

                # if self.trainer.state.epoch ==0.:
                if not self.if_get_idx[name]:
                    self.if_get_idx[name] = True
                    if not self.random_indices:
                        sparse_idx = torch.flatten(abs(grad).float()).topk(sparse_param.train_num).indices
                    else:
                        sparse_idx = _random_flat_indices(param.numel(), sparse_param.train_num, grad.device)
                    sparse_param.idx = _flat_to_parameter_indices(sparse_idx, param.shape).to(param.device)
                    return torch.zeros_like(grad)

                # ##if you are interested in grad proportion, uncomment following code
                '''
                grad_norm = torch.norm(grad).cpu().numpy().item()
                sparse_grad_norm = torch.norm(grad[sparse_param.idx[0], sparse_param.idx[1]]).cpu().numpy().item()
                grad_proportion = sparse_grad_norm/grad_norm*100
                self.record[n].append((grad_norm, grad_proportion))
                # print(f"{n} grad proportion: {grad_proportion:.2f}")
                '''

                idx = sparse_param.idx.to(grad.device)
                sparse_grad = grad[tuple(idx)]

                ## get the sparse grad
                if sparse_param.grad is not None:
                    sparse_param.grad = sparse_param.grad + sparse_grad
                else:
                    sparse_param.grad = sparse_grad.clone()

                self.grad_acc_count[name] += 1
                if self.grad_acc_count[name] == self.grad_acc:
                    ## update the initial param sparsely
                    sparse_delta = torch.sparse_coo_tensor(
                        sparse_param.idx.to(param.device),
                        sparse_param.detach().to(device=param.device, dtype=param.dtype),
                        size=param.shape,
                        dtype=param.dtype,
                        device=param.device,
                    ).to_dense()
                    param.data.add_(sparse_delta)
                    sparse_param.data.zero_()
                    self.grad_acc_count[name] = 0
                    # print('sparse update!')
            return torch.zeros_like(grad)

        return hook
                    
            
