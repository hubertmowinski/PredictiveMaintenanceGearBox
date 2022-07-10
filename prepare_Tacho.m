function sData = prepare_Tacho(data)
%PREPARE_TACHO similar to preparaData for Vibration
%   This function takes only data that is >= 10 seconds because of 
% starting
Tacho = data.Tacho{1};
Tacho = Tacho(1:30000,:);

Vibration = data.Vibration{1};
Vibration = Vibration(1:30000,:);

sData = table({Tacho}, 'VariableNames',{'Vibration', 'Tacho'});
end

