import Models
import FullModelFunctions


def inferOnVideo(path_to_video):

  path_to_weights = 'drive/My Drive/MERL Dataset/'

  torch.cuda.empty_cache()
  model1 = backbone_feature_extractor( features_extraction_output = 32768,classification = False)
  model2 = backbone_feature_extractor( features_extraction_output = 150528,classification = False)
  model3 = backbone_feature_extractor( features_extraction_output = 32768,classification = False)
  model4 = backbone_feature_extractor( features_extraction_output = 150528,classification = False)
  model1, model2 = prepare_models(model1, model2, model3, model4,path_to_weights)
  torch.cuda.empty_cache()
  msn = MSN(spatial =model1, temporal=model2, spatial2=model3, temporal2=model4,        
            features_extraction_output=183296)
  msbl_model = bi_lstm(msn,batch_size = 20,num_layers = 1)
  torch.cuda.empty_cache()
  weights = torch.load(path_to_weights + '/Full Model2.pi')
  torch.cuda.empty_cache()
  msbl_model.load_state_dict(weights)
  del weights 
  torch.cuda.empty_cache()
  msbl_model.to(device)
  classification_model = msbl_model
  video, length = load_video_MS(path_to_video, half = False)
  dropped_batch_lenght = 20 - (length % 20)
  video = list(video)
  for i in range(dropped_batch_lenght):
        (video).append(video[len(video)-1])
  test_dataloader = get_data_loader(video, batch_size = 20, with_labels = False, labels = None)
  predictions = []
  probability = nn.Softmax()
  for x1,x2,x3,x4 in test_dataloader:
      with torch.no_grad():
        x = {1:x1.cuda() , 2:x2.cuda() , 3:x3.cuda() , 4:x4.cuda()}
        y_pred = classification_model.forward(x)
        del x
        torch.cuda.empty_cache()
        y_pred = y_pred.view(-1,6)
        y_pred = probability(y_pred)
        y_pred = y_pred.max(1)[1]
        y_pred = y_pred.cpu()
        torch.cuda.empty_cache()
        y_pred = list(chain.from_iterable(repeat(n, 7) for n in y_pred))
        predictions.append((y_pred))
  predictions = np.array(predictions)
  predictions = list(predictions.reshape(predictions.shape[0]*predictions.shape[1]))
  for i in range(dropped_batch_lenght):
        predictions = predictions[:-7]

  return predictions



