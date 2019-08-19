########################################################
# For Loading /  Processing / Training Full MSBL models#
########################################################

def get_data_MS(videos_path, label_path, scaling_factor=0.5, sample_labels=True):

  # data  , length= load_video_MS(videos_path,scaling_factor )
  print('Loading ', videos_path)
  data = np.load(videos_path, allow_pickle=True)
  length = len(data) * 12
  # print('loaded ',path)
  # labels = np.load(labels_path,allow_pickle= True)
  labels = get_video_label_MS(label_path, length, half=True)
  if sample_labels:
    # Since each chunk of video is 6 Frames, we sample the Labels at 6 labels per sample
    labels = labels[[i for i in range(0, len(labels), 6)]]
  ## Transform the Data
  labels = torch.tensor(labels)
  print(len(labels))
  count = len(data)
  # print(labels.shape)
  # print(data.shape)
  ## Sometime labels lenght might not be equal frames lenght,
  while (len(labels) > len(data)):
    labels = labels[:-1]
    print('removed from labels')
  while (len(labels) < len(data)):
    data = data[:-1]
    print('removed from Data')
  return data, labels, count


def get_video_label_MS(label_path, video_lenght, half=True, data_from_numpy=False):

  label_data = scipy.io.loadmat(label_path)
  # Initially each frame is label 0
  label = [0 for i in range(video_lenght)]
  label = np.array(label)

  for category_number in range(5):
      # Each video chunk in the same class defined as in labels
      for video_chunk in label_data['tlabs'][category_number][0]:
        label[video_chunk[0]:video_chunk[1]] = category_number+1

  label = np.array(label)
  if half:
      label = label[[i for i in range(1, video_lenght, 2)]]

  return (label)


def load_video_MS(video_path, resize_scale=0.5, half=True):

          video = []
          print('Loading ', video_path)

          cap = cv2.VideoCapture(video_path)
          length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
          frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
          frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
          print('Frame width: '+str(frame_width) +
                "   frame_height: "+str(frame_height))
          width, height = int(
              frame_width * resize_scale), int(frame_height*resize_scale)
          print('resizing_width: '+str(width) +
                '    resizing_heighth: '+str(height))
          # taking the Background from 2nd second
          cap.set(cv2.CAP_PROP_POS_MSEC, 2*1000)
          reval, background = cap.read()
          # Preprocess the Background
          background = cv2.resize(background, (width, height))
          background = cv2.GaussianBlur(background, (21, 21), 0)
          background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
          # Get back to the 0 second
          cap.set(cv2.CAP_PROP_POS_MSEC, 0 * 1000)
          while True:
            #                 video.append(img)
              chunk = []
              for i in range(0, 6):
                  if half:
                    # This is neglected because we load at 15 FPS , and video is stored at 30 FPS
                    reval, img = cap.read()
                  reval, img = cap.read()

                  if not reval:
                      # End of frames
                      if i != 0:
                        while i < 6:
                          chunk.append(chunk[len(chunk)-1])
                          i += 1
                          print(len(chunk))
                        print('END')
                        chunk = np.array(chunk)
                        video.append(process_chunk(chunk, background))
                      video = np.array(video)
                      cap.release()
                      del cap
                      print(length)
                      return video, length

                  img = cv2.resize(img, (width, height))
                  img = cv2.GaussianBlur(img, (3, 3), 0)
                  # Here we store video chunk of 6 Frames and then pass it to get_stacked_pixel_trajectory
                  chunk.append(img)
              chunk = np.array(chunk)
              video.append(process_chunk(chunk, background))
          video = np.array(video)
          cap.release()
          del cap
          print(length)
          return video, length


def process_chunk(chunk, background):
  img1 = cv2.resize(chunk[2], (256, 256))
  img2 = get_stacked_pixel_trajectory(chunk, 2)
  img3 = cv2.resize(subtract_background(chunk[2], background), (256, 256))
  img4 = get_stacked_pixel_trajectory(
      subtract_background(chunk, background, List=True), 2, rgb=True)
  chunk = [img1, img2, img3, img4]
  return chunk


def get_data_loader(video, labels, batch_size, with_labels=True):
  x1 = [torch.tensor(video[i][0]) for i in range(len(video))]
  x1 = [Image.fromarray(i.numpy()) for i in x1]
  x1 = [prepare(i) for i in x1]
  x1 = torch.stack(x1, dim=0)

  x2 = [torch.tensor(video[i][1]) for i in range(len(video))]
  x2 = [Image.fromarray(i.numpy()) for i in x2]
  x2 = [prepare(i) for i in x2]
  x2 = torch.stack(x2, dim=0)

  x3 = [torch.tensor(video[i][2]) for i in range(len(video))]
  x3 = [Image.fromarray(i.numpy()) for i in x3]
  x3 = [prepare(i) for i in x3]
  x3 = torch.stack(x3, dim=0)

  x4 = [torch.tensor(video[i][3]) for i in range(len(video))]
  x4 = [Image.fromarray(i.numpy()) for i in x4]
  x4 = [prepare(i) for i in x4]
  x4 = torch.stack(x4, dim=0)

  my_dataset = utils.TensorDataset(
      x1, x2, x3, x4, labels)  # create your datset
  dataloader = utils.DataLoader(
      my_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
  return dataloader



def model_fit_MS(data_path, label_path,  no_training_subjects, no_validation_subjects, batch_size,classification_model,save_path  ,
          optimizer, no_epochs=5 ,fps=15 ,criterion =  nn.CrossEntropyLoss()):
    '''
    This function performs the following
    1- load the Dataset (call get_data) according to the Type of model and Type of Data set (NUMPY of Video)
    2- Construct the Data Loader 
    3- Perform training step (call train function)
    4- Perform validation step
    Parameters:
    from_numpy (boolean) if true, the data is loaded from numpy arrays
    '''
    val_acc = -999
    history = []
    validation_subjects = no_training_subjects+1 + no_validation_subjects
    session_id, subject_id = 1,1
    for i in range(no_epochs):
        print('Training Epoch {} ...'.format(i+1))
        running_loss,train_correct,valid_correct,val_loss = 0,0,0,0
        
        train_count,valid_count = 0,0
        while subject_id < no_training_subjects+1:

            # Loading the videos and labels one at a time from the stored Numpy files
            path =  data_path + "/Copy of "+ str(subject_id) + "_" + str(session_id)+ '.npy'
            if  os.path.exists(path): # check whether this file exists in the directory
              labels_path =  label_path + str(subject_id) +"_" + str(session_id)+'_label.mat'
              data , labels, count = get_data_MS(path,labels_path )
              train_count += count
             
             
              train_dataloader = get_data_loader(data,labels,batch_size=batch_size)
              # del data, labels, my_dataset # To free some CUDA Memory
              # Training Step
              loss , correct = training_batch(classification_model,train_dataloader,optimizer,criterion)
              torch.cuda.empty_cache()
              running_loss +=loss
              train_correct += correct
            else:
                print('No such path:' ,path)
            # Next Video    
            session_id += 1
            if session_id > 3:
              subject_id += 1
              session_id = 1
              if subject_id > 10:
                 torch.save(classification_model.state_dict(), save_path)
         
           
        ## Validation step
        print( " Validating Epoch ", i+1)
        subject_id,session_id = no_training_subjects+1 , 1
        
        while subject_id < validation_subjects:

            # Loading the videos and labels one at a time from the stored Numpy files or from Stored Videos
            path =  data_path + "/Copy of "+ str(subject_id) + "_" + str(session_id)+ '.npy'
            if  os.path.exists(path): # check whether this file exists in the directory
              labels_path =  label_path + str(subject_id) +"_" + str(session_id)+"_label.mat"
              
              data , labels, count = get_data_MS(path,labels_path)
              valid_count += count
              train_dataloader = get_data_loader(data,labels,batch_size=batch_size)
              # del data, labels, my_dataset # To free some CUDA Memory
              # Validation Step
              loss , correct = validation_step(classification_model,train_dataloader,criterion)
              torch.cuda.empty_cache()

              val_loss += loss
              valid_correct += correct
            # Next Video
            session_id += 1
            if session_id > 3:
              subject_id += 1
              session_id = 1
        
        test_accu =  int(train_correct) / train_count
        valid_accu = int(valid_correct) / valid_count

        print('Train correct: ',int(train_correct), ' out of: ',train_count )
        print('Tets correct: ', int(valid_correct), ' out of: ',valid_count)
        print ('Epoch [{}/{}], -T-Loss : {:.6f} , Train_accuracy: {:.4f} ,  Val_loss: {:.6f}  -Val_acc: {:.4f}'.format(
            i+1, no_epochs,running_loss/train_count,test_accu ,val_loss/valid_count, valid_accu))

        if val_acc < valid_accu:
          torch.save(classification_model.state_dict(), save_path)
          print('val_acc has improved from {:.4f} to {:.4f}, model is saved at {}'.format(val_acc , valid_accu,save_path))
          val_acc = valid_accu
        else:
          print('val_acc has not Improved from {:.4f}'.format(val_acc))    
        history.append([i,running_loss/train_count,valid_accu,val_loss/valid_count])
        session_id, subject_id = 1,1
        #accu.append(100 * int(test_correct) / len(X_test))

    return history



def training_batch(classification_model,train_dataloader,optimizer,criterion):
      running_loss,train_correct = 0,0
      probability = nn.Softmax()
      for x1,x2,x3,x4,y_true in train_dataloader:
          y_true= y_true.type(torch.LongTensor)
          y_true = y_true.cuda()
          optimizer.zero_grad()
          x = {1:x1.cuda() , 2:x2.cuda() , 3:x3.cuda() , 4:x4.cuda()}
          y_pred = classification_model(x)
          del x
          torch.cuda.empty_cache()
          y_pred = y_pred.view(-1,6)
          loss = criterion(y_pred, y_true)
          running_loss += loss.item()
          y_pred = probability(y_pred)
          train_correct +=  (y_true == y_pred.max(1)[1]).sum()
          del y_pred
          torch.cuda.empty_cache()
          loss.backward()
          del loss
          optimizer.step()
      return running_loss,train_correct

def validation_step(classification_model,train_dataloader,criterion):
  val_loss,valid_correct = 0,0
  probability = nn.Softmax()
  for x1,x2,x3,x4,y_true in train_dataloader:
      y_true= y_true.type(torch.LongTensor)
      y_true = y_true.cuda()
      
      with torch.no_grad():
        
        x = {1:x1.cuda() , 2:x2.cuda() , 3:x3.cuda() , 4:x4.cuda()}
        
        
        y_pred = classification_model.forward(x)
        del x
        torch.cuda.empty_cache()

        y_pred = y_pred.view(-1,6)
        loss = criterion(y_pred, y_true)
        val_loss += loss.item()
        y_pred = probability(y_pred)
        valid_correct +=  (y_true == y_pred.max(1)[1]).sum()
        torch.cuda.empty_cache()
  return val_loss, valid_correct


