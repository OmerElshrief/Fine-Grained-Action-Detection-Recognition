transform = transforms.Compose(
    
    [
#         transforms.ToPILImage(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])


def prepare(img):
  
  # img = cv2.fastNlMeansDenoisingColored(img,None,15,10,7,21)
  # img = cv2.GaussianBlur(img, (3,3), 0) 
  img = transform(img)
  
  # img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  
  return img
  
def get_optical_flow(first_frame,second_frame, rgb ):
        '''
        To extract the Obtical flow between 2 frames 
        '''
        # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
        original = first_frame
        if rgb:
         
          first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
          second_frame = cv2.cvtColor(second_frame, cv2.COLOR_BGR2GRAY)
        else:
           original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        # denoising and bluring
        # first_gray = cv2.fastNlMeansDenoising(first_gray,None,templateWindowSize= 7,h = 10,searchWindowSize  = 21)
        
        # second_gray = cv2.fastNlMeansDenoising(second_gray,None,templateWindowSize= 7,h = 10,searchWindowSize  = 21)
        # Creates an image filled with zero intensities with the same dimensions as the frame
        mask = np.zeros_like(original)
        # Sets image saturation to maximum
        mask[..., 1] = 255
        flow = cv2.calcOpticalFlowFarneback(first_frame, second_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Sets image hue according to the optical flow direction
        mask[..., 0] = angle * 180 / np.pi / 2
        # Sets image value according to the optical flow magnitude (normalized)
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        # Converts HSV to RGB (BGR) color representation
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
        return cv2.resize(rgb, (224, 224))


  
def get_stacked_pixel_trajectory( video1, i,rgb = True):
    '''
    Function to get stacked optical flows between a Middle from video[i] and it's 6 neighbouring frames
    to form the Object trajectory 
    video: chunk of frames [7 frames]
    i : index of the middel frame
    '''
#     print("Extracting Obejct trajectory")
    of1 = get_optical_flow(first_frame = video1[i], second_frame=video1[i-3],rgb = rgb )
    of2 = get_optical_flow(video1[i], video1[i-2],rgb)
    of3 = get_optical_flow(video1[i],video1[i-1],rgb)

    of4 = get_optical_flow(video1[i],video1[i+1],rgb)
    of5 = get_optical_flow(video1[i],video1[i+2],rgb)
    of6 = get_optical_flow(video1[i],video1[i+3],rgb)

    of = np.concatenate((of1, of2, of3, of4, of5, of6), axis = 1)
#     plt.figure()
#    plt.imshow(of)
    return of


  
  
  
def subtract_background(frame,background, List = False):
    
    if List:
      chunk = []
      for img in frame:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # In each iteration, calculate absolute difference between current frame and reference frame
        difference = cv2.absdiff(gray, background)
        # Apply thresholding to eliminate noise
        thresh = cv2.threshold(difference, 15, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
        chunk.append(thresh)
      return chunk

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # In each iteration, calculate absolute difference between current frame and reference frame
    difference = cv2.absdiff(gray, background)
    # Apply thresholding to eliminate noise
    thresh = cv2.threshold(difference, 15, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
    return(thresh)









def label_to_one_hot(labels):
    batch_size = len(labels)
    nb_digits = 6
    # Dummy input that HAS to be 2D for the scatter (you can use view(-1,1) if needed)
    y = torch.Tensor(labels.float()).view(-1,1).long()
    # One hot encoding buffer that you create out of the loop and just keep reusing
    y_onehot = torch.FloatTensor(batch_size, nb_digits)

    # In your for loop
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)

    return y_onehot
      