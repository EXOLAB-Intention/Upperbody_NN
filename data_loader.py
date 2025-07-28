from functions import *

def load_processed_data(folder_path='./data_250602/'):
    all_trial_data = load_all_trials_from_all_files_with_transition(folder_path, pattern='cropped_*.h5')

    for trial in all_trial_data:
        if 'test' not in trial['file']: 
            trial['label'] = generate_fixed_sequence_labels_with_transition(trial)
        else:
            test_seq = ['Stand','Sit','Stand','Sit','Stand','Sit','Stand','Walk','Stand','Walk','Stand','Walk','Stand']
            trial['label'] = generate_fixed_sequence_labels_with_transition(trial, test_seq=test_seq)
        trial['regression_label'] = generate_regression_label(trial['label'])
        convert_labels_to_int(trial, phase_to_int_transition)
        add_velocity(trial, ts=0.01)
    all_trial_data_processed2 = process_all_emg(all_trial_data, lp_cutoff=5, norm_method='max')
    all_trial_data_processed = add_sagittal_angle(
        all_trial_data_processed2, 
        angle_keys=['elbowR','elbowL','shoulderR','shoulderL','trunk','elbowR_vel','elbowL_vel','shoulderR_vel','shoulderL_vel','trunk_vel']
    )
    return all_trial_data_processed