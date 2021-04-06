import os
import numpy as np
import json
import mdn
from tensorflow_probability import distributions as tfd
import tensorflow as tf


# get motion_db file structure
basedir = '../data/train/video/deep-dance/'
motion_db = {}
dancers = [dancer for dancer in next(os.walk(basedir))[1]]
for dancer in dancers:
    motion_db[dancer] = [folder for folder in next(os.walk(basedir + dancer))[1]]
    
#bone segments of sceleton 
bones = [
    # Right leg
    (0, 1),
    (1, 2),
    (2, 3),
    # Left leg
    (0, 4),
    (4, 5),
    (5, 6),
    # Torso
    (0, 7),
    (7, 8),
    # Head
    (8, 9),
    (9, 10),
    # Right arm
    (8, 14),
    (14, 15),
    (15, 16),
    # Left arm
    (8, 11),
    (11, 12),
    (12, 13),
]
    
#get all tags 
all_tags = set([tag for folderlist in motion_db.values() for folder in folderlist for tag in folder.split("_")])

def get_training_data(dancers = "all", tags = "all", look_back = 15, target_length = 1, traj = True, normalize_z = True, normalize_body=False, body_segments = np.array([0.09205135, 0.38231316, 0.37099043, 0.09205053, 0.38036615,
        0.37101445, 0.20778206, 0.23425052, 0.0848529 , 0.10083677,
        0.10969228, 0.23822378, 0.19867802, 0.10972143, 0.23854321,
        0.1993194 ])):
    """loads data 
    
    Args:
      dancers: list of dancers to include, i.e ['maria', 'raymond']
      tags: list of tags to include, i.e ['impro', 'standing'], can also be list of lists 
            if tags should only be included if they occur in combination, i.e ["impro", ["allbody", "uplevel"]]
      lookback: int, length of pose sequences in which data is cut for input into the model
      target_lenght: int, length of target poses the model is supposed to predict
      traj: if False, subtracts the hip trajectory from all other keypoints but not from the hip itself
      normalize_body: if True, rescals pose to a predefined body size, given in body_segments
      body_segments: if normalize_body is True, pose is rescaled according to these segment lenghts

    Returns:
      X,y i.e input (X),target (y) both numpy arrays for model training
    
    """
    print("getTragingsData",str(normalize_body)) 
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
        frames = keypoints['arr_0']
        if normalize_z:
            #substract mean from z axis
            frames = np.concatenate((frames[:,:,:2],frames[:,:,2:]-np.mean(np.min(frames[:,:,2], axis=1))), axis=2)
        data.append(frames)
        

        
    # create input and target data
    dataX, dataY = [], []
    for dataset in data:
        
                    
        #rescale pose to a predefined body size
        if normalize_body:
            print("Normalize")
            rescaled_dataset = []
            for pose in dataset:
                pose = normalize_pose(pose, body_segments)
                rescaled_dataset.append(pose)
            dataset = np.asarray(rescaled_dataset)
          
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

def normalize_pose(pose, body_segments):
    p = pose.reshape((17,3))
    count = 0
    new_pose = np.zeros((17,3))
    new_pose[0] = p[0]
    for bone in bones:
        #print("count, bone, p[bone[0]]", count, bone, p[bone[0]])
        direction = np.subtract(p[bone[1]], p[bone[0]])
        direction = direction / np.sqrt(np.sum(direction**2))
        new_pose[bone[1]] = new_pose[bone[0]] + direction * body_segments[count]
        count += 1
        
    return new_pose

def generate_performance(model, initial_positions, steps_limit=100, n_mixtures=3, temp=1.0, sigma_temp=0.0, look_back=10, traj=True, post_rescale=False,body_segments = np.array([0.09205135, 0.38231316, 0.37099043, 0.09205053, 0.38036615,
        0.37101445, 0.20778206, 0.23425052, 0.0848529 , 0.10083677,
        0.10969228, 0.23822378, 0.19867802, 0.10972143, 0.23854321,
        0.1993194 ])):
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
            if not traj:
                posen = np.concatenate(([pose[0]], pose[1:] + pose[0])) 
            if post_rescale:
                pose = normalize_pose(pose, body_segments)
                pose = pose.reshape((51))
            performance.append(pose)
        steps += len(new_poses)
        
    #reshape performance array
    performance = np.reshape(performance,(np.shape(performance)[0],17,3))
    
    #if not traj:
        #performance = np.array([np.concatenate(([x[0]], x[1:] + x[0])) for x in performance])
        
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
        
        
        
        


def custom_mixture_loss_func(output_dim, num_mixes, segs_loss_weight=1.0, floor_err= False, floor_err_weight=1.0, traj = False, tfsegs=None):
    """Construct a  custom loss function for the MDN layer parametrised by number of mixtures.
       Additionally a penalty is added if the segment length deviates from the average segment length in the training data"""
    
    # average segment length of all training data
    if tfsegs is None:
        tfsegs = tf.constant([0.09205135, 0.38231316, 0.37099043, 0.09205053, 0.38036615,
           0.37101445, 0.20778206, 0.23425052, 0.0848529 , 0.10083677,
           0.10969228, 0.23822378, 0.19867802, 0.10972143, 0.23854321,
           0.1993194 ])
    
    # Construct a loss function with the right number of mixtures and outputs
    def mdn_loss_func(y_true, y_pred):
        # Reshape inputs in case this is used in a TimeDistribued layer

        y_pred = tf.reshape(y_pred, [-1, (2 * num_mixes * output_dim) + num_mixes], name='reshape_ypreds')
        y_true = tf.reshape(y_true, [-1, output_dim], name='reshape_ytrue')
        # Split the inputs into paramaters
        out_mu, out_sigma, out_pi = tf.split(y_pred, num_or_size_splits=[num_mixes * output_dim,
                                                                         num_mixes * output_dim,
                                                                         num_mixes],
                                             axis=-1, name='mdn_coef_split')

        # Construct the mixture models
        cat = tfd.Categorical(logits=out_pi)
        component_splits = [output_dim] * num_mixes
        mus = tf.split(out_mu, num_or_size_splits=component_splits, axis=1)
        sigs = tf.split(out_sigma, num_or_size_splits=component_splits, axis=1)
        coll = [tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
                in zip(mus, sigs)]
        mixture = tfd.Mixture(cat=cat, components=coll)
        
        #part 1: negative entropy of model prediction
        loss = mixture.log_prob(y_true)
        loss = tf.negative(loss)
        loss = tf.reduce_mean(loss)
        
        #part 2: error from segment lengths 
        # calculate means for keypoints
        kp_predicted_means = mixture.mean()
        kp_predicted_means = tf.reshape(kp_predicted_means, [-1,17,3])
        if not traj:
            t1 = kp_predicted_means[:,:1,:]
            t2 = kp_predicted_means[:,1:,:] + t1
            kp_predicted_means = tf.concat([t1, t2], 1)
        # calculate average segment lengths       
        segments = tf.map_fn(fn=lambda el: tf.convert_to_tensor([el[bone[1]]-el[bone[0]] for bone in bones]),
                                 elems=kp_predicted_means, fn_output_signature = tf.TensorSpec((16,3), tf.float32))
        segment_lengths = tf.math.reduce_euclidean_norm(segments, 2)
        loss_segs_len_err = segs_loss_weight*tf.math.reduce_euclidean_norm(segment_lengths - tfsegs,1)
        loss_segs_len_err = tf.reduce_mean(loss_segs_len_err)
        loss = loss + loss_segs_len_err
        
        if floor_err:
            #part3: penalty for leaving the ground
            loss_floor_err = tf.reduce_min(kp_predicted_means[:,:,2], axis=1)
            loss_floor_err = floor_err_weight*tf.math.reduce_euclidean_norm(loss_floor_err)
            #add losses
            loss = loss +  loss_floor_err
            
        return loss

    # Actually return the loss function
    with tf.name_scope('MDN'):
        return mdn_loss_func
