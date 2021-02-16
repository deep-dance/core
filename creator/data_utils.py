import os
import numpy as np
import json
import mdn


# get motion_db file structure
basedir = '../data/train/video/deep-dance/'
motion_db = {}
dancers = [dancer for dancer in next(os.walk(basedir))[1]]
for dancer in dancers:
    motion_db[dancer] = [folder for folder in next(os.walk(basedir + dancer))[1]]
    
#get all tags 
all_tags = set([tag for folderlist in motion_db.values() for folder in folderlist for tag in folder.split("_")])

def get_training_data(dancers = "all", tags = "all", look_back = 15, target_length = 1, traj = True):
    """loads data 
    
    Args:
      dancers: list of dancers to include, i.e ['maria', 'raymond']
      tags: list of tags to include, i.e ['impro', 'standing'], can also be list of lists 
            if tags should only be included if they occur in combination, i.e ["impro", ["allbody", "uplevel"]]
      lookback: int, length of pose sequences in which data is cut for input into the model
      target_lenght: int, length of target poses the model is supposed to predict
      traj: if False, subtracts the hip trajectory from all other keypoints but not from the hip itself

    Returns:
      X,y i.e input (X),target (y) both numpy arrays for model training
    
    """
    # default gets the data of all dancers
    if dancers is "all":
        dancers = list(motion_db.keys())
        
    # default includes all tags
    if tags == "all":
        tags = all_tags
    
    # collect all file names of relevant data
    include_files = []
    for dancer in dancers:
        try:
            for folder_name in motion_db[dancer]:
                folder_tag_set = set(folder_name.split("_"))
                for tag in tags:
                    if isinstance(tag, str):
                        if tag in folder_tag_set:
                            include_files.append(basedir + dancer + "/" + folder_name + "/" + 'pose3d.npz')
                    else:
                        if set(tag) <= folder_tag_set:
                            include_files.append(basedir + dancer + "/" + folder_name + "/" + 'pose3d.npz')
                            
        except KeyError:
            print("The specified dancer does not exist in the data")
            
    data = []
    for filename in set(include_files):
        keypoints = np.load(filename, allow_pickle=True)
        data.append(keypoints['arr_0'])
        

        
    # create input and target data
    dataX, dataY = [], []
    for dataset in data:
          
        if not traj:
            # substract hip trajectory every but at hip keypoint
            dataset = np.array([np.concatenate(([x[0]], x[1:] - x[0])) for x in dataset])
  
            
        # reshape input to be [samples, features = (keypoints*3dim)] 
        dataset = np.reshape(dataset,  (dataset.shape[0], dataset.shape[1]*dataset.shape[2]))
        
            
            
        for i in range(len(dataset) - look_back - target_length):
            # dataX has dimension [samples, lookback, features = (keypoints*3dim)] 
            a = dataset[i:(i + look_back), :]     
            dataX.append(a)
            # dataY has dimension [samples, features = (keypoints*3dim)] 
            dataY.append(dataset[i + look_back : i + look_back + target_length, :])
            
    return np.array(dataX), np.array(dataY)


def generate_performance(model, initial_positions, steps_limit=100, n_mixtures=3, temp=1.0, sigma_temp=0.0, look_back=10, traj=True):
    """Generates aperformance
    
    Args:
      model: trained model used for prediction
      initial_positions: array of poses to initiate prediction. length has to be equal or greater to expected model input
      steps_limit: duration of generated performance in frames
      n_mixtures: number of mixture parameters in model'l final mdn layer
      temp: the temperature for sampling between mixture components 
      sigma_temp: the temperature for sampling from the normal distribution
      look_back: number of poses the model takes as input
      traj: if False, adds the hip trajectory to all other keypoints but the hip 
           (this reverses the manipulation done in get_training_data() when flag traj=False is set)
    Returns: 
      numpy array with generated pose sequence"""
    time = 0
    steps = 0
    performance = [pose for pose in initial_positions]
    while (steps < steps_limit):
        params = model.predict(np.expand_dims(np.array(performance[-look_back:]), axis=0))
        new_poses = mdn.sample_from_output(params[0], 51, n_mixtures, temp=temp, sigma_temp=sigma_temp)
        for pose in new_poses:
            performance.append(pose)
        steps += len(new_poses)
        
    #reshape performance array
    performance = np.reshape(performance,(np.shape(performance)[0],17,3))
    
    if not traj:
        performance = np.array([np.concatenate(([x[0]], x[1:] + x[0])) for x in performance])
        
    return np.array(performance)

def save_seq_to_json(performance, filename, path_base_dir=os.path.abspath("./")):
    """ converts numpy array to json format as needed by the three.js viewer
        and saves json to file
    
    Args: 
      performance: nummpy array of generated performance
      file: filename to which json is saved
      path: path to base directory
    """

    jsonData = {}
    jsonData['bones'] = {
        'rightLeg': [0,1,2,3],
        'leftLeg' : [0,4,5,6],
        'spine': [0,7,8],
        'rightArm':[8,14,15,16],
        'leftArm':[8,11,12,13],
        'head':[8,9,10]
    }

    jsonData['frames'] = {}
    frameIdx = 0;
    for frame in performance:
        jsonData['frames'][frameIdx] = []
        for point in frame:
            jsonData['frames'][frameIdx].append([point[0].astype(float), point[1].astype(float), point[2].astype(float)])
        frameIdx += 1
    with open(os.path.join(path_base_dir, filename), 'w') as outfile:
        json.dump(jsonData, outfile)
