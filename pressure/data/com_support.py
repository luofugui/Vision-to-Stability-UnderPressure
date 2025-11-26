import numpy as np
from typing import Tuple

class ViconCoM:
    def __init__(self):
        """
        Initialize the Center of Mass calculator for a 15-segment Vicon Plug-in-Gait model.
        """
        # Segment mass fractions (one entry per segment in segment_joints below)
        self.mass_percentages = np.array([
            0.142,  # Pelvis
            0.100,  # Femur_R
            0.100,  # Femur_L
            0.0465, # Tibia_R
            0.0465, # Tibia_L
            0.0145, # Foot_R
            0.0145, # Foot_L
            0.028,  # Humerus_R
            0.028,  # Humerus_L
            0.016,  # Radius_R
            0.016,  # Radius_L
            0.006,  # Hand_R
            0.006,  # Hand_L
            0.355,  # Thorax
            0.081   # Head
        ])

        # Corresponding CoM location (fraction of distance from start-joint to end-joint)
        self.com_percentages = np.array([
            0.895,  # Pelvis
            0.567,  # Femur_R
            0.567,  # Femur_L
            0.567,  # Tibia_R
            0.567,  # Tibia_L
            0.500,  # Foot_R
            0.500,  # Foot_L
            0.564,  # Humerus_R
            0.564,  # Humerus_L
            0.570,  # Radius_R
            0.570,  # Radius_L
            0.6205, # Hand_R
            0.6205, # Hand_L
            0.630,  # Thorax
            0.500   # Head (approximation)
        ])

        # Dictionary of segments = (start_joint_idx, end_joint_idx) in your 17-joint PiG scheme
        self.segment_joints = {
            "Pelvis":    [12, 13],
            "Femur_R":   [6, 7],
            "Femur_L":   [9, 10],
            "Tibia_R":   [7, 8],
            "Tibia_L":   [10, 11],
            "Foot_R":    [8, 8],   # approximate foot center
            "Foot_L":    [11, 11], # approximate foot center
            "Humerus_R": [0, 1],
            "Humerus_L": [3, 4],
            "Radius_R":  [1, 2],
            "Radius_L":  [4, 5],
            "Hand_R":    [2, 2],   # approximate hand center
            "Hand_L":    [5, 5],   # approximate hand center
            "Thorax":    [14, 16],
            "Head":      [16, 16]  # approximate head center
        }

    def center_data_around_pelvis(self, data):
        """
        Subtract the pelvis (joint 12) x,y,z from every joint so that pelvis is at (0,0,0).
        data shape: (N, 17, 4) or (N, 17, 3).  We only shift the first 3 coords.
        """
        for f in range(data.shape[0]):
            pelvis_xyz = data[f, 12, :3]
            data[f, :, :3] -= pelvis_xyz
        return data

    def compute_segment_com(self, joints):
        """
        Compute each segment's CoM for every frame.

        joints: (N, 17, 4) => x,y,z + confidence in last channel
        Returns:
            segment_coms: (N, 15, 3)
            segment_conf: (N, 15) each segment 1 if valid, else 0
        """
        N = joints.shape[0]
        n_segments = len(self.segment_joints)

        segment_coms = np.zeros((N, n_segments, 3))
        segment_conf = np.zeros((N, n_segments))

        # Loop over each segment
        for i, (seg_name, (j1, j2)) in enumerate(self.segment_joints.items()):
            joint_start = joints[:, j1, :3]  # (N,3)
            joint_end   = joints[:, j2, :3]

            alpha = self.com_percentages[i]
            segment_coms[:, i, :] = joint_start + alpha * (joint_end - joint_start)

            # Confidence is 1 only if *both* joints exist (i.e., confidence=1 in the last channel)
            seg_conf = joints[:, j1, 3] * joints[:, j2, 3]  # elementwise product
            segment_conf[:, i] = seg_conf

        return segment_coms, segment_conf

    def compute_total_com(self, joints):
        """
        Compute total CoM frame by frame, returning (N,4) => [x,y,z,confidence].
        Confidence is the fraction of total body mass that was valid for that frame.
        """
        # 1) Optionally recenter data about the pelvis:
        #    (If you want to do it outside, just comment this out.)
        # joints = self.center_data_around_pelvis(joints.copy())

        # 2) Segment-level CoMs and confidence
        segment_coms, segment_conf = self.compute_segment_com(joints)

        # 3) Weighted by segment mass, zero if missing
        mass_masked = self.mass_percentages[None, :] * segment_conf  # shape (N,15)
        weighted_coms = segment_coms * mass_masked[..., None]        # shape (N,15,3)

        # 4) Sum up the valid segment masses per frame
        total_mass_per_frame = np.sum(mass_masked, axis=1, keepdims=True)
        # Avoid dividing by zero
        total_mass_per_frame = np.where(total_mass_per_frame == 0, 1.0, total_mass_per_frame)

        # 5) Total COM = sum of [segment COM * segment mass] / sum of segment masses
        total_com_xyz = np.sum(weighted_coms, axis=1) / total_mass_per_frame  # (N,3)

        # 6) Let "confidence" = fraction of total mass that is present
        #    i.e. total_mass_per_frame / sum_of_all_segments
        sum_of_all_segments = np.sum(self.mass_percentages)  # e.g. ~1.392
        com_confidence = (np.sum(mass_masked, axis=1) / sum_of_all_segments)  # shape (N,)

        # 7) Package into [x,y,z,conf]
        total_com = np.concatenate([total_com_xyz, com_confidence[:, None]], axis=1)  # (N,4)
        return total_com
    
class CenterOfMass:
    def __init__(self, dims=3):
        """
        Initialize Center of Mass calculator with BODY25 model constants
        
        Args:
            dims: Number of position dimensions (2 for x,y or 3 for x,y,z)
        """
        if dims not in [2, 3]:
            raise ValueError("dims must be either 2 or 3")
            
        self.dims = dims
        
        # Mass and length percentages from MATLAB code
        self.mass_percentages = np.array([
            6.94, 43.46, 2.71, 1.62, 0.61, 2.71, 1.62, 0.61, 
            14.16, 4.33, 1.37, 14.16, 4.33, 1.37
        ]) / 100.0
        
        self.length_percentages = np.array([
            50.02, 43.20, 57.72, 45.74, 79.00, 57.72, 45.74, 79.00,
            40.95, 43.95, 44.15, 40.95, 43.95, 44.15
        ]) / 100.0
        
        self.num_segments = 14
    
    def calculate_segment_confidence(self, conf: np.ndarray):
        """
        Calculate confidence for each body segment based on contributing joints.

        Args:
            joints: Array of shape (N, 25, 4), where the last dimension includes 
                    x, y, z coordinates and confidence.

        Returns:
            segment_confidence: Array of shape (N, 14), with confidence values for each segment.
        """
        N = conf.shape[0]
        segment_confidence = np.zeros((N, 14))


        # Define confidence for each segment based on joint indices and operations
        segment_confidence[:, 0] = conf[:, 0]
        segment_confidence[:, 1] = (conf[:, 8] * conf[:, 2] * conf[:, 5]) ** (1/3)
        segment_confidence[:, 2] = (conf[:, 3] * conf[:, 2]) ** (1/2)
        segment_confidence[:, 3] = (conf[:, 4] * conf[:, 3]) ** (1/2)
        segment_confidence[:, 4] = conf[:, 4]
        segment_confidence[:, 5] = (conf[:, 6] * conf[:, 5]) ** (1/2)
        segment_confidence[:, 6] = (conf[:, 7] * conf[:, 6]) ** (1/2)
        segment_confidence[:, 7] = conf[:, 7]
        segment_confidence[:, 8] = (conf[:, 10] * conf[:, 9]) ** (1/2)
        segment_confidence[:, 9] = (conf[:, 11] * conf[:, 10]) ** (1/2)
        segment_confidence[:, 10] = (conf[:, 22] * conf[:, 11]) ** (1/2)
        segment_confidence[:, 11] = (conf[:, 13] * conf[:, 12]) ** (1/2)
        segment_confidence[:, 12] = (conf[:, 14] * conf[:, 13]) ** (1/2)
        segment_confidence[:, 13] = (conf[:, 19] * conf[:, 14]) ** (1/2)

        return segment_confidence
 
    def calculate_com_locations(self, joints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate CoM locations for each body segment
        
        Args:
            joints: Array of shape (N, 25, dims+1) where:
                   N is number of frames
                   25 is number of joints
                   dims+1 includes x,y(,z) and confidence
        
        Returns:
            com_locations: Array of shape (N, 14, dims+1)
            segment_mask: Array of shape (N, 14)
        """
        N = joints.shape[0]
        com_locations = np.zeros((N, self.num_segments, self.dims))
        
        joint_confidences = joints[..., -1]
        joints = joints[..., :-1]  # Remove confidence values
      
        # Using nose location for head
        com_locations[:,0,:] = joints[:,0,:]  # nose
        # Calculate segmented CoM based on length percentages and joint coordinates
        A = (joints[:,2,:] + joints[:,5,:]) / 2  # shoulder midpoint (top of thorax)
        com_locations[:,1,:] = ((joints[:,8,:] - A) * self.length_percentages[1]) + A  # thorax
        com_locations[:,2,:] = ((joints[:,3,:] - joints[:,2,:]) * self.length_percentages[2]) + joints[:,2,:]  # upper right arm
        com_locations[:,3,:] = ((joints[:,4,:] - joints[:,3,:]) * self.length_percentages[3]) + joints[:,3,:]  # lower right arm
       
        com_locations[:,4,:] = joints[:,4,:]  # wrist as hand location
        com_locations[:,5,:] = ((joints[:,6,:] - joints[:,5,:]) * self.length_percentages[5]) + joints[:,5,:]  # upper left arm
        com_locations[:,6,:] = ((joints[:,7,:] - joints[:,6,:]) * self.length_percentages[5]) + joints[:,5,:]  # lower left arm
        
        com_locations[:,7,:] = joints[:,7,:]  # left wrist as hand location
        com_locations[:,8,:] = ((joints[:,10,:] - joints[:,9,:]) * self.length_percentages[8]) + joints[:,9,:]  # right thigh
        com_locations[:,9,:] = ((joints[:,11,:] - joints[:,10,:]) * self.length_percentages[9]) + joints[:,10,:]  # right shin
        com_locations[:,10,:] = ((joints[:,22,:] - joints[:,11,:]) * self.length_percentages[10]) + joints[:,11,:]  # right foot
        com_locations[:,11,:] = ((joints[:,13,:] - joints[:,12,:]) * self.length_percentages[11]) + joints[:,12,:]  # left thigh
        com_locations[:,12,:] = ((joints[:,14,:] - joints[:,13,:]) * self.length_percentages[12]) + joints[:,13,:]  # left shin
        com_locations[:,13,:] = ((joints[:,19,:] - joints[:,14,:]) * self.length_percentages[13]) + joints[:,14,:]  # left foot
           
        # Create segment mask based on confidence values
        segment_confidence = self.calculate_segment_confidence(joint_confidences)
        return com_locations, segment_confidence 
    
    def calculate_total_com(self, joints):
        """
        Calculate total center of mass from joint positions
        
        Args:
            joints: Array of shape (N, 25, dims+1) 
                   where dims+1 includes x,y(,z) and confidence
        
        Returns:
            com_pose: Array of shape (N, dims) containing total CoM positions
            com_mask: Array of shape (N,) indicating frames with all segments
            segment_mask: Array of shape (N, 14) containing segment validity masks
        """
        # Calculate CoM locations for each segment
        com_locations, segment_confidence = self.calculate_com_locations(joints)

        # Calculate weighted confidences using mass percentages
        weighted_confidences = self.mass_percentages[None, :] * segment_confidence
        com_confidence = np.sum(weighted_confidences, axis=1) / np.sum(self.mass_percentages)        
        
        # # Convert boolean mask to float and handle missing segments
        # segment_mask = (segment_mask > 0.0).astype(float)
       
        # Calculate masked mass percentages 
        mass_percentages_masked = self.mass_percentages[None, :] * segment_confidence
        sum_masked = np.sum(mass_percentages_masked, axis=1, keepdims=True)
        sum_masked = np.where(sum_masked == 0, 1.0, sum_masked)  # Avoid division by zero
        mass_percentages_rescaled = mass_percentages_masked / sum_masked

        # Calculate weighted sum of CoM positions
        com_weighted_sum = np.sum(
            com_locations[..., :self.dims] * mass_percentages_rescaled[..., None],
            axis=1
        )
        
        return com_weighted_sum, com_confidence
        
