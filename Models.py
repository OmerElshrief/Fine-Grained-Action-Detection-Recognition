import torchvision.models as models
vgg16 = models.vgg16(pretrained=True)

def prepare_models(spatial,temporal,spatial2,temporal2,path_to_weights_dir):
      weights = torch.load(path_to_weights_dir+"/model1.pi")
  spatial.load_state_dict(weights)
  weights = torch.load(path_to_weights_dir+"/model_temporal.pt")
  temporal.load_state_dict(weights)
#   weights = torch.load(path_to_weights_dir+"/sub_back_patial_model12.pi")
#   spatial2.load_state_dict(weights)
#   weights= torch.load(path_to_weights_dir+"/model_temporal_background_subtracted1.pt")
#   temporal2.load_state_dict(weights)
  del weights
  torch.cuda.empty_cache()

  return spatial,temporal


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class backbone_feature_extractor (nn.Module):
  def __init__(self,features_extraction_output,softmax_output = False, classification = False,  Dropout = 0.5):
    
        super().__init__()
        
        self.features_extractor = vgg16.features
        set_parameter_requires_grad(self.features_extractor,True)
#         set_parameter_requires_grad(self.features_extractor[24:31],False)
        self.flatten = Flatten()
        self.classifier = nn.Sequential(nn.Linear(in_features = features_extraction_output ,out_features=4096, bias=True ),
                                        nn.ReLU(),
                                        nn.Dropout(Dropout),
                                        nn.Linear(in_features=4096, out_features=1024, bias=True),
                                        nn.ReLU(),
                                        nn.Dropout(Dropout),
                                        nn.Linear(in_features=1024, out_features=6, bias=True)
                                       )
        set_parameter_requires_grad(self.classifier,True)                                
        self.classification = classification
  
  def forward(self, x):
    torch.cuda.empty_cache()

    x = self.features_extractor(x)
    torch.cuda.empty_cache()

    x = self.flatten(x)
    if self.classification:
      
      x= self.classifier(x)
      
    return x



class MSN(nn.Module):
  
  """
  Takes a chunked video of shape (n_chunks, frames, frame_height, frame_width, channels)
  
  Outputs a tensor of shape (n_chunks, 200) containing chunk feature vector
  """
  
  def __init__(self,spatial,temporal,spatial2,temporal2,features_extraction_output):
    super().__init__()
#     weights = torch.load(path_to_weights_dir+"/model_temporal.pt")
#     keys = ['classifier.0.weight', 'classifier.0.bias', 'classifier.3.weight', 'classifier.3.bias', 'classifier.6.weight','classifier.6.bias']
#     for key in keys:
#     weights.pop(key)  ## We only need Feature extractors weights
    
    self.cnn_temp_coarse = temporal2
#     self.cnn_temp_fine   = temporal2
    self.cnn_spat_coarse = spatial
#     self.cnn_spat_fine   = spatial
    self.conv1 = nn.Sequential(nn.Conv1d(1, 64, 3, stride=2),
                                 nn.ReLU(),
                                 nn.MaxPool1d(3,2),
                                 nn.Dropout(0.5),
                                 nn.Conv1d(64, 20, 3,stride=2),
                                 nn.ReLU(),
                                 nn.Conv1d(20, 10, 3,stride=2),
                                 nn.ReLU())
    
    self.fin_vec = nn.Sequential(nn.Linear(in_features =114550 , out_features = 2048),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(in_features =2048 , out_features = 1024),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(in_features = 1024 , out_features= 200)
                                )
   
                                 
    self.flatten = Flatten()
    
  def forward(self,x):
    
    x1 = (self.cnn_spat_coarse(x[1]))
    x2 = (self.cnn_temp_coarse(x[2]))
#     x3 = (self.cnn_spat_fine(x[3]))
#     x4 = (self.cnn_temp_fine(x[4]))
    o = torch.cat((x1, x2), dim=1)
    o = o.view(o.shape[0] , 1 , o.shape[1])
    
    o = self.conv1(o)
    torch.cuda.empty_cache()
    o = self.flatten(o)
    torch.cuda.empty_cache()
    o = self.fin_vec(o)
    torch.cuda.empty_cache()
    
    return o




class bi_lstm(nn.Module):

  def __init__(self,msn,batch_size,hidden_size=60,num_layers = 1):
    super().__init__()
    self.msn_feature_extractor = msn
    self.lstm = nn.LSTM(200, hidden_size= hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True )

#     self.lstm.cuda()
    
    # h_0, c_0 of shape (num_layers * num_directions, batch, hidden_size)
    
    self.hidden_vect_1 = hidden_vect_1 = (Variable(torch.zeros(2*num_layers, batch_size, hidden_size).cuda()),
                                          Variable(torch.zeros(2*num_layers, batch_size, hidden_size).cuda()))
    self.forward_classifier = nn.Linear(in_features = 60, out_features = 6)
    self.backward_classifier = nn.Linear(in_features = 60, out_features = 6)

  def forward(self,x):
    torch.cuda.empty_cache()

    x = self.msn_feature_extractor(x)   ## Shape = (n_chunks, 200)
    torch.cuda.empty_cache()

    x = x.view(-1, 1, 200)
    
    # output of shape (seq_len, batch, num_directions * hidden_size)
    output, hidden = self.lstm( x, self.hidden_vect_1    )
    torch.cuda.empty_cache()

    x_forward, x_backward = torch.split(output,60,dim=2)
  
  
    x_forward= (self.forward_classifier(x_forward))
    torch.cuda.empty_cache()

    x_backward = (self.backward_classifier(x_backward))
    torch.cuda.empty_cache()

    x = (x_forward+x_backward) /2
    

    return (x)

  
  
  
#####
# Manual LSTM Cells
#####
class Model(nn.Module):
      
  """
  Takes a chunked video of shape (n_chunks, frames, channels, frame_height, frame_width)
                         
  Outputs 2 tensors of shape (n_chunks, 2) labels containing the action prediction of chunks
  using backward and forward lstm
  """
  
  def __init__(self,msn):
    super().__init__()
    self.msn_feature_extractor = msn
    self.lstm_cell = nn.LSTM(input_size = 200, hidden_size = 60,num_layers=1, batch_first=False,bidirectional= True)
    self.classifier = nn.Linear(in_features = 60, out_features = 6)
    self.fin = nn.Softmax(dim = 1)
    
    
  def forward(self,x, get_logits = True):
    x = self.msn_feature_extractor(x)   ## Shape = (n_chunks, 200)
    x = x.view(-1, 1, 200)
    
    hx = torch.randn(1, 60)                ## Handle shape
    cx = torch.randn(1, 60)                ## Handle shape
    forward_outputs = []
    print(x.shape)
    # for chunk in range(x.shape[0]):
    hx, cx = self.lstm_cell(x, (hx, cx))
    forward_outputs.append(hx)
      
    hx = torch.randn(1, 60)                ## Handle shape
    cx = torch.randn(1, 60)                ## Handle shape   
    backward_outputs = []
    # for chunk in reversed(range(x.shape[0])):
    hx, cx = self.lstm_cell(x, (hx, cx))
    backward_outputs.append(hx)
      
    forward_outputs = torch.FloatTensor(forward_outputs)      ## Handle Shape
    backward_outputs = torch.FloatTensor(backward_outpus)     ## Handle Shape
    
    pre_logits_forw = self.classifier(forward_outputs)
    pre_logits_back = self.classifier(backward_outputs)
    
    if get_logits:
      logits_forw = self.fin(pre_logits_forw)
      logits_back = self.fin(pre_logits_back)
      return torch.mean(torch.stack([logits_forw, logits_back])) 
    
    return pre_logits_forw, pre_logits_back   