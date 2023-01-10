import numpy as np
import jittor as jt

class GetNearC2W:
    def __init__(self, args):
        super(GetNearC2W, self).__init__()
        self.near_c2w_type = args.near_c2w_type
        self.near_c2w_rot = args.near_c2w_rot
        self.near_c2w_trans = args.near_c2w_trans
        
        self.smoothing_rate = args.smoothing_rate
        self.smoothing_step_size = args.smoothing_step_size

    
    def __call__(self, c2w, all_poses=None,j=None,iter_=None):
        assert (c2w.shape == (3,4))
        
        if self.near_c2w_type == 'rot_from_origin':
            return self.rot_from_origin(c2w,iter_)
    
    def rot_from_origin(self, c2w,iter_=None):
        rot = c2w[:3,:3]
        pos = c2w[:3,-1:]
        rot_mat = self.get_rotation_matrix(iter_)
        pos = jt.matmul(rot_mat, pos)
        rot = jt.matmul(rot_mat, rot)
        c2w = jt.concat((rot, pos), -1)
        return c2w

    def get_rotation_matrix(self,iter_=None):
        rotation = self.near_c2w_rot

        phi = (rotation*(np.pi / 180.))
        x = np.random.uniform(-phi, phi)
        y = np.random.uniform(-phi, phi)
        z = np.random.uniform(-phi, phi)
        
        rot_x = jt.float32([
                    [1,0,0],
                    [0,np.cos(x),-np.sin(x)],
                    [0,np.sin(x), np.cos(x)]
                    ])
        rot_y = jt.float32([
                    [np.cos(y),0,-np.sin(y)],
                    [0,1,0],
                    [np.sin(y),0, np.cos(y)]
                    ])
        rot_z = jt.float32([
                    [np.cos(z),-np.sin(z),0],
                    [np.sin(z),np.cos(z),0],
                    [0,0,1],
                    ])
        rot_mat = jt.matmul(rot_x, jt.matmul(rot_y, rot_z))
        return rot_mat

