%% General Experiment (OK버튼만 사용한 경우) 
clear all;  close all;  clc;

%% Sensor Attachment Information
%
% [EMG]
% EMG L1: Left Rectus Abdominis (복직근)
% EMG L2: Left External Oblique (외복사근)
% EMG L3: Left Trapezius (승모근)
% EMG L4: Left Posterior Deltoid (후면 삼각근)
% 
% EMG R1: Right Rectus Abdominis (복직근)
% EMG R2: Right External Oblique (외복사근)
% EMG R3: Right Trapezius (승모근)
% EMG R4: Right Posterior Deltoid (후면 삼각근)
% 
% 
% [IMU]
% IMU5: Pelvis
% IMU6: Trunk 
% IMU7: Right Upperarm (RS)
% IMU8: Right Forearm (RE)
% IMU9: Left Upperarm (LS)
% IMU10: Left Forearm (LE)

    
%% Load the h5 file
folderPath = '../OriginalData/250813';          % Change
h5fileName = 'Stand-to-Sit.h5';                 % Change
filename = fullfile(folderPath, h5fileName);

time = '/Sensor/Time/time';

emgL1 = '/Sensor/EMG/emgL1';
emgL2 = '/Sensor/EMG/emgL2';
emgL3 = '/Sensor/EMG/emgL3';
emgL4 = '/Sensor/EMG/emgL4';
emgR1 = '/Sensor/EMG/emgR1';
emgR2 = '/Sensor/EMG/emgR2';
emgR3 = '/Sensor/EMG/emgR3';
emgR4 = '/Sensor/EMG/emgR4';

imu5 = '/Sensor/IMU/imu5';
imu6 = '/Sensor/IMU/imu6';
imu7 = '/Sensor/IMU/imu7';
imu8 = '/Sensor/IMU/imu8';
imu9 = '/Sensor/IMU/imu9';
imu10 = '/Sensor/IMU/imu10';

button_ok = '/Controller/button_ok';
button_b = '/Controller/button_b';

data_time = h5read(filename, time);

data_emgL1 = (h5read(filename, emgL1));
data_emgL2 = (h5read(filename, emgL2));
data_emgL3 = (h5read(filename, emgL3));
data_emgL4 = (h5read(filename, emgL4));
data_emgR1 = (h5read(filename, emgR1));
data_emgR2 = (h5read(filename, emgR2));
data_emgR3 = (h5read(filename, emgR3));
data_emgR4 = (h5read(filename, emgR4));

data_imu5 = (h5read(filename, imu5))';
data_imu6 = (h5read(filename, imu6))';
data_imu7 = (h5read(filename, imu7))';
data_imu8 = (h5read(filename, imu8))';
data_imu9 = (h5read(filename, imu9))';
data_imu10 = (h5read(filename, imu10))';

data_button_ok = (h5read(filename, button_ok));
data_button_b  = (h5read(filename, button_b));

data_label_ok = strcmp(data_button_ok, 'TRUE');
data_label_b  = strcmp(data_button_b, 'TRUE');    


%%
rising_edge_ok = uint8([0; diff(data_label_ok) == 1]);
falling_edge_ok = uint8([0; diff(data_label_ok) == -1]);
rising_edge_b  = uint8([0; diff(data_label_b)  == 1]);

rising_idx_ok = find(rising_edge_ok == true);
falling_idx_ok = find(falling_edge_ok == true);
rising_idx_b  = find(rising_edge_b  == true);

ok_risings  = find(rising_edge_ok);
ok_fallings = find(falling_edge_ok);
b_risings = find(rising_edge_b);


%% 데이터 자르기 
cropped_APA_data = {};
APA_time = 1000;        % msec
Ts = 10;                % msec

for count = 1:size(ok_risings,1)
    start_APA_idx = ok_risings(count);
    stop_APA_idx = start_APA_idx + APA_time/Ts - 1;
    
    % --- APA --- %
    cropped_APA.time = data_time(start_APA_idx:stop_APA_idx,1);
    cropped_APA.imu5 = data_imu5(start_APA_idx:stop_APA_idx,:);
    cropped_APA.imu6 = data_imu6(start_APA_idx:stop_APA_idx,:);
    cropped_APA.imu7 = data_imu7(start_APA_idx:stop_APA_idx,:);
    cropped_APA.imu8 = data_imu8(start_APA_idx:stop_APA_idx,:);
    cropped_APA.imu9 = data_imu9(start_APA_idx:stop_APA_idx,:);
    cropped_APA.imu10 = data_imu10(start_APA_idx:stop_APA_idx,:);

    cropped_APA.emgL1 = data_emgL1(start_APA_idx:stop_APA_idx,1);
    cropped_APA.emgL2 = data_emgL2(start_APA_idx:stop_APA_idx,1);
    cropped_APA.emgL3 = data_emgL3(start_APA_idx:stop_APA_idx,1);
    cropped_APA.emgL4 = data_emgL4(start_APA_idx:stop_APA_idx,1);
    cropped_APA.emgR1 = data_emgR1(start_APA_idx:stop_APA_idx,1);
    cropped_APA.emgR2 = data_emgR2(start_APA_idx:stop_APA_idx,1);
    cropped_APA.emgR3 = data_emgR3(start_APA_idx:stop_APA_idx,1);
    cropped_APA.emgR4 = data_emgR4(start_APA_idx:stop_APA_idx,1);
    
    cropped_APA.rising_ok = rising_edge_ok(start_APA_idx:stop_APA_idx,1);
    cropped_APA.falling_ok = falling_edge_ok(start_APA_idx:stop_APA_idx,1);
    cropped_APA.button_b = rising_edge_b(start_APA_idx:stop_APA_idx,1);
    
    cropped_APA_data{end+1} = cropped_APA;
end
disp('Finish Appending');



%% For APA
cropped_APA_filename = ['cropped_APA_', h5fileName];
for i = 1:length(cropped_APA_data)
    trial = cropped_APA_data{i};
    group_prefix = sprintf('/trial_%d', i);

    fields = fieldnames(trial);

    for j = 1:numel(fields)
        name = fields{j};
        path = sprintf('%s/%s', group_prefix, name);
        data = trial.(name);

        % h5create는 처음에만 실행 (존재하면 skip 또는 try-catch)
        if ~isfile(cropped_APA_filename) || ~h5exists(cropped_APA_filename, path)
            h5create(cropped_APA_filename, path, size(data), 'Datatype', class(data));
        end

        % 데이터 저장
        h5write(cropped_APA_filename, path, data);
    end
end
disp('APA File is created');


