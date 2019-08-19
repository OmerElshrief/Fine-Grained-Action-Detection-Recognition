def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    else:
      for param in model.parameters():
            param.requires_grad = True
      
    
      
def create_optimizer(model , r = 0.0001):
  params_to_update = model.parameters()
  print("Params to learn:")
  
  params_to_update = []
  for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
        
  return  optim.SGD(params_to_update, lr=r,weight_decay=1e-6)

      
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
      
class Transform(nn.Module):
  def __init__(self):
    super(Transform, self).__init__()
  
  def forward(self,x):
    return transforms(x)
        