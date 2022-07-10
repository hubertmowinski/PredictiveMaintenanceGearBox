function sData = limit_data(data)
%PREPARE_TACHO similar to preparaData for Vibration

%   This function takes only data that is >= 10 seconds because of 
% starting
%$data.Tacho{1}.Time =round(data.Tacho{1}.Time, 4);
data.Tacho{1}.Time = data.Tacho{1}.Time + seconds(10000)
data.Tacho{1}.Time = dateround(data.Tacho{1}.Time);
[~,ia] = unique(data.Tacho{1}.Time, "stable");
data.Tacho{1}.Time = data.Tacho{1}.Time - seconds(10000)
Tacho = data.Tacho{1}(ia,:);
%Tacho = unique(data.Tacho{1}, "rows", "stable");
%Tacho = data.Tacho{1};
%Tacho = Tacho(1:30000,:);

data.Tacho{1}
%data.Vibration{1}.Time = round(data.Vibration{1}.Time, 4);
%Vibration = unique(data.Vibration{1},"stable");
data.Vibration{1}.Time = data.Vibration{1}.Time + seconds(10000)
data.Vibration{1}.Time = dateround(data.Vibration{1}.Time);
[~,ia] = unique(data.Vibration{1}.Time, "stable");
data.Vibration{1}.Time = data.Vibration{1}.Time - seconds(10000)
Vibration = data.Vibration{1}(ia,:);
% Vibration = data.Vibration{1};
% Vibration = Vibration(1:30000,:);

sData = table({Vibration}, {Tacho}, 'VariableNames',{'Vibration', 'Tacho'});
end

