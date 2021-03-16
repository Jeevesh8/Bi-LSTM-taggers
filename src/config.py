config = {
    'max_length' : 512,
    'class_names' : ['O', 'B-C', 'B-P', 'I-C', 'I-P'],
    'ft_data_folders' : ['/content/change-my-view-modes/v2.0/data/'],
    'max_tree_size' : 10,
    'max_labelled_users_per_tree':10,
    'batch_size' : 4,
    'd_model': 512,
    
    'init_alphas': [0.0, 0.0, 0.0, -10000., -10000.],
    'transition_init': [[1.0, -10000., -10000., 1.0, 1.0],
                        [1.0, -10000., -10000., 1.0, 1.0],
                        [1.0, -10000., -10000., 1.0, 1.0],
                        [-10000., 1.0, -10000., 1.0, -10000.],
                        [-10000., -10000., 1.0, -10000., 1.0]],
    
    'scale_factors':[1.0, 10.0, 10.0, 1.0, 1.0], 
}