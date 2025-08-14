%% Reset ALL
clear all;
close all;
clc;

%% User Selection
button_ON = 1;  

imu1_ON = 0;
imu2_ON = 0;
imu3_ON = 0;
imu4_ON = 0;
imu5_ON = 1;
imu6_ON = 1;
imu7_ON = 1;
imu8_ON = 1;
imu9_ON = 1;
imu10_ON = 1;
imu_num = imu1_ON + imu2_ON + imu3_ON + imu4_ON + imu5_ON + imu6_ON + imu7_ON + imu8_ON + imu9_ON + imu10_ON;

%% Load the h5 file
filename = 'final.h5';

time = '/Sensor/Time/time';
emgL1 = '/Sensor/EMG/emgL1';
emgL2 = '/Sensor/EMG/emgL2';
emgL3 = '/Sensor/EMG/emgL3';
emgL4 = '/Sensor/EMG/emgL4';
emgR1 = '/Sensor/EMG/emgR1';
emgR2 = '/Sensor/EMG/emgR2';
emgR3 = '/Sensor/EMG/emgR3';
emgR4 = '/Sensor/EMG/emgR4';

if imu1_ON == 1
    imu1 = '/Sensor/IMU/imu1';
end
if imu2_ON == 1
    imu2 = '/Sensor/IMU/imu2';
end
if imu3_ON == 1
    imu3 = '/Sensor/IMU/imu3';
end
if imu4_ON == 1
    imu4 = '/Sensor/IMU/imu4';
end
if imu5_ON == 1
    imu5 = '/Sensor/IMU/imu5';
end
if imu6_ON == 1
    imu6 = '/Sensor/IMU/imu6';
end
if imu7_ON == 1
    imu7 = '/Sensor/IMU/imu7';
end
if imu8_ON == 1
    imu8 = '/Sensor/IMU/imu8';
end
if imu9_ON == 1
    imu9 = '/Sensor/IMU/imu9';
end
if imu10_ON == 1
    imu10 = '/Sensor/IMU/imu10';
end

button_A = '/Controller/button_a';
button_B = '/Controller/button_b';
button_OK = '/Controller/button_ok';


data_time = (h5read(filename, time))';
data_emgL1 = (h5read(filename, emgL1))';
data_emgL2 = (h5read(filename, emgL2))';
data_emgL3 = (h5read(filename, emgL3))';
data_emgL4 = (h5read(filename, emgL4))';
data_emgR1 = (h5read(filename, emgR1))';
data_emgR2 = (h5read(filename, emgR2))';
data_emgR3 = (h5read(filename, emgR3))';
data_emgR4 = (h5read(filename, emgR4))';

if imu1_ON == 1
    data_imu1 = (h5read(filename, imu1))';
end
if imu2_ON == 1
    data_imu2 = (h5read(filename, imu2))';
end
if imu3_ON == 1
    data_imu3 = (h5read(filename, imu3))';
end
if imu4_ON == 1
    data_imu4 = (h5read(filename, imu4))';
end
if imu5_ON == 1
    data_imu5 = (h5read(filename, imu5))';
end
if imu6_ON == 1
    data_imu6 = (h5read(filename, imu6))';
end
if imu7_ON == 1
    data_imu7 = (h5read(filename, imu7))';
end
if imu8_ON == 1
    data_imu8 = (h5read(filename, imu8))';
end
if imu9_ON == 1
    data_imu9 = (h5read(filename, imu9))';
end
if imu10_ON == 1
    data_imu10 = (h5read(filename, imu10))';
end

data_buttonA = (h5read(filename, button_A))';
data_buttonB = (h5read(filename, button_B))';
data_buttonOK = (h5read(filename, button_OK))';
data_buttonA = strcmp(data_buttonA, 'TRUE');
data_buttonB = strcmp(data_buttonB, 'TRUE');
data_buttonOK = strcmp(data_buttonOK, 'TRUE');


%% Plot IMU

figure(1);
imuCnt = 1;

if imu1_ON == 1
    subplot(imu_num,1,imuCnt);
    plot(data_time, data_imu1);
    hold on;
    if button_ON == 1
        plot(data_time, data_buttonA, 'r', 'LineWidth', 2);
        plot(data_time, data_buttonB, 'b', 'LineWidth', 2);
        plot(data_time, data_buttonOK, 'k', 'LineWidth', 2);
    end
    legend("IMU1");
    imuCnt = imuCnt + 1;
end

if imu2_ON == 1
    subplot(imu_num,1,imuCnt);
    plot(data_time, data_imu2);
    hold on;
    if button_ON == 1
        plot(data_time, data_buttonA, 'r', 'LineWidth', 2);
        plot(data_time, data_buttonB, 'b', 'LineWidth', 2);
        plot(data_time, data_buttonOK, 'k', 'LineWidth', 2);
    end
    legend("IMU2");
    imuCnt = imuCnt + 1;
end

if imu3_ON == 1
    subplot(imu_num,1,imuCnt);
    plot(data_time, data_imu3);
    hold on;
    if button_ON == 1
        plot(data_time, data_buttonA, 'r', 'LineWidth', 2);
        plot(data_time, data_buttonB, 'b', 'LineWidth', 2);
        plot(data_time, data_buttonOK, 'k', 'LineWidth', 2);
    end
    legend("IMU3");
    imuCnt = imuCnt + 1;
end

if imu4_ON == 1
    subplot(imu_num,1,imuCnt);
    plot(data_time, data_imu4);
    hold on;
    if button_ON == 1
        plot(data_time, data_buttonA, 'r', 'LineWidth', 2);
        plot(data_time, data_buttonB, 'b', 'LineWidth', 2);
        plot(data_time, data_buttonOK, 'k', 'LineWidth', 2);
    end
    legend("IMU4");
    imuCnt = imuCnt + 1;
end

if imu5_ON == 1
    subplot(imu_num,1,imuCnt);
    plot(data_time, data_imu5);
    hold on;
    if button_ON == 1
        plot(data_time, data_buttonA, 'r', 'LineWidth', 2);
        plot(data_time, data_buttonB, 'b', 'LineWidth', 2);
        plot(data_time, data_buttonOK, 'k', 'LineWidth', 2);
    end
    legend("IMU5");
    imuCnt = imuCnt + 1;
end

if imu6_ON == 1
    subplot(imu_num,1,imuCnt);
    plot(data_time, data_imu6);
    hold on;
    if button_ON == 1
        plot(data_time, data_buttonA, 'r', 'LineWidth', 2);
        plot(data_time, data_buttonB, 'b', 'LineWidth', 2);
        plot(data_time, data_buttonOK, 'k', 'LineWidth', 2);
    end
    legend("IMU6");
    imuCnt = imuCnt + 1;
end

if imu7_ON == 1
    subplot(imu_num,1,imuCnt);
    plot(data_time, data_imu7);
    hold on;
    if button_ON == 1
        plot(data_time, data_buttonA, 'r', 'LineWidth', 2);
        plot(data_time, data_buttonB, 'b', 'LineWidth', 2);
        plot(data_time, data_buttonOK, 'k', 'LineWidth', 2);
    end
    legend("IMU7");
    imuCnt = imuCnt + 1;
end

if imu8_ON == 1
    subplot(imu_num,1,imuCnt);
    plot(data_time, data_imu8);
    hold on;
    if button_ON == 1
        plot(data_time, data_buttonA, 'r', 'LineWidth', 2);
        plot(data_time, data_buttonB, 'b', 'LineWidth', 2);
        plot(data_time, data_buttonOK, 'k', 'LineWidth', 2);
    end
    legend("IMU8");
    imuCnt = imuCnt + 1;
end

if imu9_ON == 1
    subplot(imu_num,1,imuCnt);
    plot(data_time, data_imu9);
    hold on;
    if button_ON == 1
        plot(data_time, data_buttonA, 'r', 'LineWidth', 2);
        plot(data_time, data_buttonB, 'b', 'LineWidth', 2);
        plot(data_time, data_buttonOK, 'k', 'LineWidth', 2);
    end
    legend("IMU9");
    imuCnt = imuCnt + 1;
end

if imu10_ON == 1
    subplot(imu_num,1,imuCnt);
    plot(data_time, data_imu10);
    hold on;
    if button_ON == 1
        plot(data_time, data_buttonA, 'r', 'LineWidth', 2);
        plot(data_time, data_buttonB, 'b', 'LineWidth', 2);
        plot(data_time, data_buttonOK, 'k', 'LineWidth', 2);
    end
    legend("IMU10");
end

xlabel("Time[msec]");



%% Plot EMG
if button_ON == 1
    data_buttonA = data_buttonA * 0.2; 
    data_buttonB = data_buttonB * 0.2; 
    data_buttonOK = data_buttonOK * 0.2; 
end

figure(2);
subplot(8,1,1);
plot(data_time, data_emgL1);
hold on;
if button_ON == 1
    plot(data_time, data_buttonA, 'r', 'LineWidth', 2);
    plot(data_time, data_buttonB, 'b', 'LineWidth', 2);
    plot(data_time, data_buttonOK, 'k', 'LineWidth', 2);
end
legend("EMGL1");
title("EMG results");

subplot(8,1,2);
plot(data_time, data_emgL2);
hold on;
if button_ON == 1
    plot(data_time, data_buttonA, 'r', 'LineWidth', 2);
    plot(data_time, data_buttonB, 'b', 'LineWidth', 2);
    plot(data_time, data_buttonOK, 'k', 'LineWidth', 2);
end
legend("EMGL2");

subplot(8,1,3);
plot(data_time, data_emgL3);
hold on;
if button_ON == 1
    plot(data_time, data_buttonA, 'r', 'LineWidth', 2);
    plot(data_time, data_buttonB, 'b', 'LineWidth', 2);
    plot(data_time, data_buttonOK, 'k', 'LineWidth', 2);
end
legend("EMGL3");

subplot(8,1,4);
plot(data_time, data_emgL4);
hold on;
if button_ON == 1
    plot(data_time, data_buttonA, 'r', 'LineWidth', 2);
    plot(data_time, data_buttonB, 'b', 'LineWidth', 2);
    plot(data_time, data_buttonOK, 'k', 'LineWidth', 2);
end
legend("EMGL4");

subplot(8,1,5);
plot(data_time, data_emgR1);
hold on;
if button_ON == 1
    plot(data_time, data_buttonA, 'r', 'LineWidth', 2);
    plot(data_time, data_buttonB, 'b', 'LineWidth', 2);
    plot(data_time, data_buttonOK, 'k', 'LineWidth', 2);
end
legend("EMGR1");

subplot(8,1,6);
plot(data_time, data_emgR2);
hold on;
if button_ON == 1
    plot(data_time, data_buttonA, 'r', 'LineWidth', 2);
    plot(data_time, data_buttonB, 'b', 'LineWidth', 2);
    plot(data_time, data_buttonOK, 'k', 'LineWidth', 2);
end
legend("EMGR2");

subplot(8,1,7);
plot(data_time, data_emgR3);
hold on;
if button_ON == 1
    plot(data_time, data_buttonA, 'r', 'LineWidth', 2);
    plot(data_time, data_buttonB, 'b', 'LineWidth', 2);
    plot(data_time, data_buttonOK, 'k', 'LineWidth', 2);
end
legend("EMGR3");

subplot(8,1,8);
plot(data_time, data_emgR4);
hold on;
if button_ON == 1
    plot(data_time, data_buttonA, 'r', 'LineWidth', 2);
    plot(data_time, data_buttonB, 'b', 'LineWidth', 2);
    plot(data_time, data_buttonOK, 'k', 'LineWidth', 2);
end
legend("EMGR4");
xlabel("Time[msec]");


%% Plot Time 
figure(3);
plot(diff(data_time));
title("data check");




