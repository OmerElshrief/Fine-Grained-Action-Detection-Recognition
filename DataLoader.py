

# def __init__( start_id , threshold_id, videos_path,labels_path):
#   self.subject_id = start_id
#   self.threshold = threshold_id
#   self.videos_path = videos_path
#   self.session_id = 1
#   self.labels_path = labels_path



def write_video(video,out_path, frame_width, frame_height,subject_id, session_id, as_numpy = False):
  if as_numpy:
    path = out_path +str(subject_id) + "_"+str(session_id)+".npy"
    np.save(path,np.array(video),allow_pickle=False)
  else:

    path = out_path +str(subject_id) + "_"+str(session_id)+".avi"
    out = cv2.VideoWriter(path,1, 20.0, frameSize = (frame_width,frame_height),)
    print("Video lenght: ", len(video))
    for frame in video:
      # Write the frame into the file 'output.avi'
      out.write(frame)
    out.release()
    del out
  print('Video of lenght ',str(len(video)),' is saved at ',path)


  
def get_data(videos_path,label_path, sample_labels = True,fps = 15, denoising = False, blur = True, resize = True,
                normalize = False, frame_width = 256, frame_height=256, background_sub = False, object_trajectory = False, 
                data_from_numpy = False):
  """
  To load the Dataset, Dataset  can be loaded from Videos or from Stored Numpy arrays 
  Parameters:
  videos_path: Path of the Dataset
  Label_path: path of the Labels 
  sample_labels: If true, labels will be sampled, label per 6 frames 
  background_sub: If true, frames loaded from videos will be background subtracted (for the Background subtraction models)
  object_trajectory: If true, frames loaded from video will be used to extract the Object trajectory for each 6 frames (for Temporal Models)
  data_from_numpy: determing if the Dataset is loaded from videos or from Stored Numpy arrays
  """

  data  , length= load_video(videos_path,fps = fps, denoising = denoising,blur = blur,resize = resize,normalize = normalize,frame_width = frame_width,
                             frame_height = frame_height,background_sub = background_sub,object_trajectory = object_trajectory,
                             data_from_numpy = data_from_numpy )
  # print('loaded ',path)
  # labels = np.load(labels_path,allow_pickle= True)
  labels = get_video_label(label_path,length,half = True,data_from_numpy=data_from_numpy)
  if not object_trajectory:
    ## We take a sample frame each 6 frames
    print(data.shape)
    data = data[[i for i in range(0,len(data),6)]]
  print(data.shape)
  data = [prepare(i) for i in data]
  data = torch.stack(data)
  print(len(labels))
  if sample_labels:
    # Since each chunk of video is 6 Frames, we sample the Labels at 6 labels per sample
    labels = labels[[i for i in range(0,len(labels),6)]]
  ## Transform the Data
  labels = torch.tensor(labels)
  print(len(labels))
  count =len(data)
  # print(labels.shape)
  # print(data.shape)
  ## Sometime labels lenght might not be equal frames lenght,
  while (len(labels) > len(data)):
    labels = labels[:-1]
    print('removed from labels')
  while (len(labels) < len(data)):
    data = data[:-1]
    print('removed from Data')
  return data , labels, count


def load_video(video_path, fps = 15, denoising = False, blur = True, resize = True, normalize = False, frame_width = 256, frame_height=256, 
               background_sub = False,object_trajectory = False, data_from_numpy = False):
      """
      background_sub: If true, frames loaded from videos will be background subtracted (for the Background subtraction models)
      object_trajectory: If true, frames loaded from video will be used to extract the Object trajectory for each 6 frames
      data_from_numpy: determing if the Dataset is loaded from videos or from Stored Numpy arrays
      """
      video  = []
      print('Loading ',video_path)
      if data_from_numpy:
         video = np.load(path)
         length = len(video)

      else:
          cap = cv2.VideoCapture(video_path)
          length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
          
          # taking the Background from 2nd second 
          cap.set(cv2.CAP_PROP_POS_MSEC,2*1000)
          reval, background = cap.read()
          # Preprocess the Background
          background = cv2.GaussianBlur(background, (3,3), 0) 
          # Get back to the 0 second
          cap.set(cv2.CAP_PROP_POS_MSEC,0)
          while True:
#                 video.append(img)
              if object_trajectory: # If it's desired to Load video frames and get the object trajectory for each 6 consecutive frames
                 chunk = []
                 for i in range(0,6):
                    reval, img = cap.read() # This is neglected because we load at 15 FPS , and video is stored at 30 FPS
                    reval, img = cap.read()

                    if not reval:
                        # End of frames
                        print('END')
                        video = np.array(video)
                        return video , length
                    if background_sub:      # This is needed for the 4th Model, as we subtract the BG and then get OT
                        # Subtract Background
                        img = subtract_background(img,background)
                    img = cv2.resize(img, (512, 512))
                    chunk.append(img) # Here we store video chunk of 6 Frames and then pass it to get_stacked_pixel_trajectory
                 chunk = np.array(chunk)
                 video.append(get_stacked_pixel_trajectory(chunk,2))

              else:

                  reval, img = cap.read() # Video is stored at 30 FPS, we want to load it at 15 FPS so we ignore a frame each iteration
                  reval, img = cap.read() # Faster than cap.set()

                  if not reval:
                      # End of frames
                      break
                  if background_sub:
                    # Subtract Background
                    img = subtract_background(img,background)

                  if resize:
                    img = cv2.resize(img, (frame_width, frame_height)) 
                  if blur:
                    img = cv2.GaussianBlur(img, (21,21), 0) 
                  if normalize:
                    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                  
                  video.append(img)

          video = np.array(video)
          # try:
          #   video = video[[i for i in range(0,len(video),2)]]
          # except IndexError:
            # print('index error')
            # pass
          cap.release()
          del cap
          print(length)
      return video , length

    
def get_video_label(label_path,video_lenght, half = True, data_from_numpy = False):
  """
  Load the labels from matlab files or form Numpy files
  """
  if data_from_numpy :
    labels = np.load(path)

  else:

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
        label = label[[i for i in range(1,video_lenght,2)]]

  return (label)











