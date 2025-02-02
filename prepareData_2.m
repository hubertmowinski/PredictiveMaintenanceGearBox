function sData = prepareData_2(data)
%% 
%
%   Clean up the signals in the data:
%       Vibration signal - remove the 1st 10 seconds of data and any duplicte times.
%       Tacho - remove 1st 10 seconds, find rising edge times

%The first 10 seconds of the simulation contains data where the
%transmission system is starting up; for analysis we discard this data.
Vibration = data.Vibration{1};
idx = Vibration.Time >= seconds(10);
Vibration = Vibration(idx,:);
Vibration.Time = Vibration.Time - Vibration.Time(1);

%The tacho signal contains pulses for the rotations of the drive and load
%shafts. Later we use time synchronous averaging on the vibration data and
%that requires the times of shaft rotations, the following code discards
%the first 10 seconds of the tacho data and finds the shaft rotation times
%in TachoPulses. 
Tacho = data.Tacho{1};


% idx = diff(Tacho.Data(:,2)) > 0.5;
% tachoPulses = Tacho.Time(find(idx)+1)-Tacho.Time(1);


idx = Tacho.Time >= seconds(10);
Tacho = Tacho(idx,:);
Tacho.Time = Tacho.Time - Tacho.Time(1);

AngularVelocity = data.AngularVelocity{1};
idx = AngularVelocity.Time >= seconds(10);
AngularVelocity = AngularVelocity(idx,:);
AngularVelocity.Time = AngularVelocity.Time - AngularVelocity.Time(1);

Gear = data.Gear{1};
idx = Gear.Time >= seconds(10);
Gear = Gear(idx,:);
Gear.Time = Gear.Time - Gear.Time(1);

Torque = data.Torque{1};
idx = Torque.Time >= seconds(10);
Torque = Torque(idx,:);
Torque.Time = Torque.Time - Torque.Time(1);

Brake = data.Brake{1};
idx = Brake.Time >= seconds(10);
Brake = Brake(idx,:);
Brake.Time = Brake.Time - Brake.Time(1);

%The Simulink.SimulationInput.Variables property contains the values of the
%fault parameters used for the simulation, these values allow us to create
%fault labels for each ensemble member. 
vars = data.SimulationInput{1}.Variables;
sF = false; sV = false; sT = false;
idx = strcmp({vars.Name},'SDrift');
if any(idx)
    sF = abs(vars(idx).Value) > 0.01; %Small drift values are not faults
end
idx = strcmp({vars.Name},'ShaftWear');
if any(idx)
    sV = vars(idx).Value < 0;
end
idx = strcmp({vars.Name},'ToothFaultGain');
if any(idx)    
    sT = abs(vars(idx).Value) < 0.1; %Small tooth fault values are not faults
end
FaultCode = sF+2*sV+4*sT; %A fault code to capture different fault conditions

%Collect processed data into a table
sData = table({Vibration},{Tacho},{AngularVelocity},{Gear},{Torque},{Brake},sF,sV,sT,FaultCode, ...
        'VariableNames',{'Vibration','Tacho','AngularVelocity','Gear','Torque','Brake','SensorDrift','ShaftWear','ToothFault','FaultCode'});
end