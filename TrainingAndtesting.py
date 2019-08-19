################################################
### For testing and Training separate Models ###
################################################


def model_fit(data_path, label_path, from_numpy, model_type, no_training_subjects, no_validation_subjects,
              batch_size, classification_model, save_path,
              optimizer, no_epochs=5, fps=15, criterion=nn.CrossEntropyLoss()):
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

    if from_numpy:
      data_extension = '.npy'
      label_extension = '_label.npy'

    else:
      data_extension = '_crop.mp4'
      label_extension = '_label.mat'
      if model_type == 3:
        background_sub = True

    for i in range(no_epochs):
        print('Training Epoch {} ...'.format(i+1))
        running_loss, train_correct, valid_correct, val_loss = 0, 0, 0, 0
        session_id, subject_id = 1, 1
        train_count, valid_count = 0, 0
        while subject_id < no_training_subjects+1:

            # Loading the videos and labels one at a time from the stored Numpy files
            path = data_path + "/" + \
                str(subject_id) + "_" + str(session_id) + data_extension
            if os.path.exists(path):  # check whether this file exists in the directory
              labels_path = label_path + \
                  str(subject_id) + "_" + str(session_id)+label_extension

              data, labels, count = get_data(
                  path, labels_path, True, background_sub=background_sub, data_from_numpy=from_numpy)
              train_count += count
              print(data.shape)
              print(labels.shape)
              my_dataset = utils.TensorDataset(
                  data, labels)  # create your datset
              train_dataloader = utils.DataLoader(
                  my_dataset, batch_size=batch_size, shuffle=False)
              del data, labels, my_dataset  # To free some CUDA Memory
              # Training Step
              loss, correct = training_batch(
                  classification_model, train_dataloader, optimizer, criterion)
              running_loss += loss
              train_correct += correct
            else:
                print('No such path:', path)
            # Next Video
            session_id += 1
            if session_id > 3:
              subject_id += 1
              session_id = 1

        ## Validation step
        print(" Validating Epoch ", i+1)
        subject_id, session_id = no_training_subjects+1, 1

        while subject_id < validation_subjects:

            # Loading the videos and labels one at a time from the stored Numpy files or from Stored Videos
            path = data_path + "/" + \
                str(subject_id) + "_" + str(session_id)+"_crop.mp4"
            if os.path.exists(path):  # check whether this file exists in the directory
              labels_path = label_path + \
                  str(subject_id) + "_" + str(session_id)+"_label.mat"

              data, labels, count = get_data(
                  path, labels_path, True, background_sub=background_sub, data_from_numpy=from_numpy)
              valid_count += count
              my_dataset = utils.TensorDataset(
                  data, labels)  # create your datset
              train_dataloader = utils.DataLoader(
                  my_dataset, batch_size=batch_size, shuffle=False)
              del data, labels, my_dataset  # To free some CUDA Memory
              # Validation Step
              loss, correct = validation_step(
                  classification_model, train_dataloader, criterion)
              val_loss += loss
              valid_correct += correct
            # Next Video
            session_id += 1
            if session_id > 3:
              subject_id += 1
              session_id = 1

        test_accu = int(train_correct) / train_count
        valid_accu = int(valid_correct) / valid_count

        print('Train correct: ', int(train_correct), ' out of: ', train_count)
        print('Tets correct: ', int(valid_correct), ' out of: ', valid_count)
        print('Epoch [{}/{}], -T-Loss : {:.6f} , Train_accuracy: {:.4f} ,  Val_loss: {:.6f}  -Val_acc: {:.4f}'.format(
            i+1, no_epochs, running_loss/train_count, test_accu, val_loss/valid_count, valid_accu))

        if val_acc < valid_accu:
          torch.save(classification_model.state_dict(), save_path)
          print('val_acc has improved from {:.4f} to {:.4f}, model is saved at {}'.format(
              val_acc, valid_accu, save_path))
          val_acc = valid_accu
        else:
          print('val_acc has not Improved from {:.4f}'.format(val_acc))
        history.append([i, running_loss/train_count,
                        valid_accu, val_loss/valid_count])

        #accu.append(100 * int(test_correct) / len(X_test))

    return history


def training_batch(classification_model, train_dataloader, optimizer, criterion):
      running_loss, train_correct = 0, 0
      probability = nn.Softmax()
      for batch, y_true in train_dataloader:
          y_true = y_true.type(torch.LongTensor)
          y_true = y_true.cuda()
          batch = (batch.cuda())
          optimizer.zero_grad()
          y_pred = classification_model(batch)
          loss = criterion(y_pred, y_true)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()
          y_pred = probability(y_pred)
          train_correct += (y_true == y_pred.max(1)[1]).sum()
          del loss, batch
          y_pred.cpu()
          y_true.cpu()
          torch.cuda.empty_cache()
      return running_loss, train_correct


def validation_step(classification_model, train_dataloader, criterion):
  val_loss, valid_correct = 0, 0
  for batch, y_true in train_dataloader:
      y_true = y_true.type(torch.LongTensor)
      y_true = y_true.cuda()
      batch = (batch.cuda())
      with torch.no_grad():
        y_pred = classification_model(batch)
        y_pred = y_pred.view(-1, 6)
        valid_correct += (y_true == y_pred.max(1)[1]).sum()
        loss = criterion(y_pred, y_true)
        val_loss += loss.item()
        del batch
        y_pred.cpu()
        y_true.cpu()
        torch.cuda.empty_cache()
  return val_loss, valid_correct


def test(classification_model, model_type, data_path, label_path, data_from_numpy, subject_id, test_subjects, batch_size,
         background_sub=False, criterion=nn.CrossEntropyLoss()):
    '''
    Parameters:
    subject_id: initial subject_id, the starting point
    test_subjects: Number of test subjects
    '''
    probability = nn.Softmax()
    test_count, valid_correct, valid_count = 0, 0, 0
    threshold = subject_id + test_subjects
    session_id = 1
    while subject_id < threshold:

        # Loading the videos and labels one at a time from the stored Numpy files
        # path = data_path + str(subject_id)+"_"+str(session_id)+"_fps_"+str(fps)+'.npy'
       # Loading the videos and labels one at a time from the stored Numpy files or from Stored Videos
        path = data_path + "/" + str(subject_id) + \
            "_" + str(session_id)+"_crop.mp4"
        if os.path.exists(path):  # check whether this file exists in the directory
          labels_path = label_path + \
              str(subject_id) + "_" + str(session_id)+"_label.mat"

          data, labels, count = get_data(
              path, labels_path, True, background_sub=background_sub, data_from_numpy=from_numpy)
          test_count += count
          my_dataset = utils.TensorDataset(data, labels)  # create your datset
          train_dataloader = utils.DataLoader(
              my_dataset, batch_size=batch_size, shuffle=False)
          del data, labels, my_dataset  # To free some CUDA Memory

          score = []
          predictions = []
          for batch, y_true in train_dataloader:
              y_true = y_true.type(torch.LongTensor)
              y_true = y_true.cuda()
              batch = (batch.cuda())
              with torch.no_grad():
                y_pred = classification_model(batch)
                y_pred = y_pred.view(-1, 6)

                valid_correct += (y_true == y_pred.max(1)[1]).sum()
                loss = criterion(y_pred, y_true)
                val_loss += loss.item()
                y_pred = probability(y_pred)
                y_pred = y_pred.cpu().numpy()

                y_pred = list(chain.from_iterable(repeat(n, 6)
                                                  for n in y_pred))
                predictions.append(np.array(y_pred))

          predictions = np.array(predictions)
          predictions = predictions.reshape(
              predictions.shape[0]*predictions.shape[1], 6)
          print(predictions.shape)
          print(len(original_labels))

          while len(original_labels) > len(predictions):
            original_labels = original_labels[:-1]
          while len(original_labels) < len(predictions):
            predictions = predictions[:-1]
          print(len(predictions))
          print(len(original_labels))
          score.append(evaluate_average_percision(
              label_to_one_hot(original_labels).numpy(), predictions, 6))
          del batch

          torch.cuda.empty_cache()
        session_id += 1
        if session_id > 3:
          subject_id += 1
          session_id = 1

    avg_precision = 0
    for i in range(len(score)):
           avg_precision += score[i][2]['micro']
    avg_precision = avg_precision / len(score)

    valid_accu = int(valid_correct) / valid_count
    print('Tess Results: ', int(valid_correct), ' out of: ', valid_count)
    print('Accuracy: ', valid_accu)
    print('Loss', str(val_loss / valid_count))
    print('average precision: ', avg_precision)
    return score
