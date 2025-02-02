import torch
import torch.nn as nn
import numpy as np
import random


from src.mask import prepare_super_mask


class Super:
    def __init__(self, model, tokenizer, outliers_ratio, sparse_module, exception=[], grad_acc=1,
                 gradient_checkpointing=False) -> None:
        self.model = model
        self.total_num = 0
        self.gradient_checkpointing = gradient_checkpointing

        self.outliers_ratio = outliers_ratio

        # Parameters need to be trained sparsely
        self.sparse_module = sparse_module
        # Parameters need to be trained normally
        self.exception = exception

        # Mapping: Parameter --> Sparse Parameter
        self.sparse_mapping = dict()
        # For convenience, we record the gradient accumulation step for each parameter.
        self.grad_acc_count = dict()
        self.grad_acc = grad_acc

        self.if_get_idx = dict()
        self.record = dict()

        # Record all the trainable parameters(the initial parameter that need be updated sparsely).
        self.named_trainable_parameters_list = list()
        # Record all the parameters that need be calculated in optimizer.
        self.named_parameters_in_optimizer_list = list()

        device = torch.device("cpu")
        # if hasattr(model.hf_device_map):
        #    device = model.hf_device_map["lm_head"]
        prepare_super_mask(model, tokenizer, dev=device, outliers_ratio=outliers_ratio)

        self.register_sparse_param()

    def register_sparse_param(self):
        """Register a sparse param for each param that need be updated sparsely
            and get the sparse grad by using backward hook
        """
        for name, layer in self.model.named_parameters():
            # select the parameters needed to be trained sparsely
            self.total_num += layer.numel()
            if any([module in name for module in self.sparse_module]) and hasattr(layer, 'wanda_topk_indices'):

                layer.requires_grad = True
                # set the number of trainable components of the parameter according to the sparse rate
                train_num = min(int(self.outliers_ratio * layer.numel()) + 1, layer.numel())
                sparse_param = nn.Parameter(layer.new_zeros(train_num), requires_grad=True)
                sparse_param.grad = sparse_param.new_zeros(train_num)
                sparse_param.train_num = train_num

                sparse_idx = layer.wanda_topk_indices

                sparse_param.idx = torch.stack(torch.unravel_index(sparse_idx, layer.shape))
                ## help the initial parameter to find the sparse parameter
                self.sparse_mapping[name] = sparse_param
                self.grad_acc_count[name] = 0
                self.if_get_idx[name] = False
                self.record[name] = []

                # ## register a backward hook to get the 'sparse' grad
                layer.register_hook(self.get_sparse_grad())

                # register it in the model so the framework can recognise the sparse param as a 'normal' param
                setattr(self.model, name.replace('.', '_') + '_sparse', sparse_param)
                # (name, p)
                self.named_trainable_parameters_list.append((name, layer))
                # (named_sparse, sparse p)
                self.named_parameters_in_optimizer_list.append((name + '_sparse', sparse_param))

            elif self.exception and any([item in name for item in self.exception]):
                layer.requires_grad = True
                self.named_trainable_parameters_list.append((name, layer))
                self.named_parameters_in_optimizer_list.append((name, layer))
            elif self.gradient_checkpointing and name == next(self.model.named_parameters())[0]:
                layer.requires_grad = True
            else:
                layer.requires_grad = False

            # ## gradient caculate after backward hook, we use following codes to ensure the first sparse module can get the sparse grad as we expect.
            # ## the first parameter in the model, the last parameter in backward propagation
            # m = list(self.model.modules())[1]
            # m.register_full_backward_hook(self.get_sparse_grad())

    # keep consistent with model.named_parameters()
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
                for name, layer in self.named_trainable_parameters():
                    if not (name in self.sparse_mapping.keys()) or layer.grad is None:
                        return

                    # print(name)
                    sparse_param = self.sparse_mapping[name]
                    grad = layer.grad.to(sparse_param)

                    ## clean the init grad
                    layer.grad = None
                    # if self.trainer.state.epoch ==0.:
                    if not self.if_get_idx[name]:
                        self.if_get_idx[name] = True

                        sparse_idx = torch.flatten(abs(grad)).topk(sparse_param.train_num).indices.cpu().numpy()
                        sparse_param.idx = np.stack(np.unravel_index(sparse_idx, layer.shape))

                        ## reset optimizer state
                        # for s in self.trainer.optimizer.state[p].values():
                        #     s.zero_()
                        if name == list(self.sparse_mapping.keys())[-1]:
                            print('switch idx')
                            # self.trainer.create_optimizer()
                            # print(sparse_param.idx)
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
                        sparse_param.grad += grad[sparse_param.idx]
                    else:
                        sparse_param.grad = grad[sparse_param.idx]

                    self.grad_acc_count[name] += 1
                    if self.grad_acc_count[name] == self.grad_acc:
                        ## update the initial param sparsely
                        delta = layer.data + torch.sparse_coo_tensor(sparse_param.idx, sparse_param, layer.shape).to(
                            layer)
                        layer.data.copy_(delta)
                        sparse_param.zero_()
                        self.grad_acc_count[name] = 0
                        # print('sparse update!')

        return hook
