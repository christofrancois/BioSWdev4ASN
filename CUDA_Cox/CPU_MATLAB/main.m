% This is the main m file calculating effective connectivity map of the
% network by importing the .rst data file using the res_to_struct.m and
% applying the cox method on it using cox.m function. 
% The user might be aware of the number of neurons in network as well as
% the duration of the recording. 

m = 4; % number of neurons 
duration = 25; % duration of recording 
data_path = '' % path to the .rst data file 
save_path = '' 
global ET; % The struct value containing all the runtime info for benchmarking 
ET = struct ; 
ET.total = 0 ; % total runtime duration
ET.coxes_all = 0 ; % sum of runtime duration of all cox methods 
ET.Zs_all = 0; % sum of runtime duration of Z value calculation 
ET.hessian_all = 0 ; % sum of runtime duration of all hessian calculation 
ET.coxes = zeros(1,m); % every individual cox method runtime duration 
ET.Zs = zeros(1,m);  % every individual Z value functions runtime duration 
ET.hessian = zeros (m,1); % % every individual hessian functions runtime duration 
ET.hessian_sum = zeros(m,1); 

spike_struct = res_to_struct (data_path, m); % converting the .res data to matlab struct 
spike_num_avg = 0; % average number of spike per neuron
for i = 1:m 
    spike_num_avg = spike_num_avg + length(spike_struct(i).Target);
end
spike_num_avg = spike_num_avg/m 
t1 = timetic; % timetic variables are for benchmarking 
tic(t1)
betahats = zeros ( m,m) ; % Matrix of betahat values in the network
n = (m*3)-1; 
betacis = zeros ( m,n) ; % Matrix of confidence interval of betahat values in the network


%%%%% applying cox method on the data 
for i = 1:m
    if i ==1 
    t2 = timetic; 
    tic(t2) 
    [temp1 ,temp2] = cox (spike_struct(i).Target , spike_struct(i).Ref ,i) ; 
    ET.coxes(i) = toc(t2);
    betahats (2:m,1) = temp1;
    betacis(2:m,1:2) = temp2 ;
    elseif i == m
     t2 = timetic; 
     tic (t2)
     [temp1,temp2 ] = cox (spike_struct(i).Target , spike_struct(i).Ref ,i)  ;
     ET.coxes(i) = toc(t2);
     betahats (1:(m-1),m) = temp1 ; 
     betacis(1:(m-1),n-1:n) = temp2 ;
    else
        t2 = timetic; 
        tic(t2)
     [temp1,temp2] = cox (spike_struct(i).Target , spike_struct(i).Ref ,i)     ;
     ET.coxes(i) = toc (t2); 
     betahats (1:i-1 , i )  = temp1 (1:i-1,1) ;
     betahats (i+1:m ,i ) = temp1 (i:end,1) ; 
     ind = 2*(i)+i-2;
     betacis  (1:i-1,ind : ind+1) =  temp2 (1:i-1,:);
     betacis  (i+1:end,ind : ind+1) = temp2 (i:end , : );
    end
disp (['Neuron ' num2str(i) ' completed']);
end

r = betahats'
c = betacis'
p = 1 ;
from = [];
to = [];
thickness = [];
for i = 1:m
    for j = 1:m
        if (c(p+1,j) > 0 && c(p,j)>0)||(c(p+1,j) < 0 && c(p,j)<0)
    from = [from; j];
    to = [to;i];
    thickness = [thickness;r(i,j)] ;
        end
    end
     p = p + 3 ; 
end

ET.total = toc(t1);
ET.hessian_sum = sum(ET.hessian,2);
ET.coxes_all = sum(ET.coxes);
ET.Zs_all = sum(ET.Zs);
ET.hessian_all = sum (ET.hessian_sum);
save([save_path 'CG_' experiment_type '_' filename(strfind(filename,'_')+1:end) '.mat'],'ET'); % save the runtim info variable as mat file
to_save = {'nurons' 'duration' 'spikes' 'total' 'coxes' 'Zs' 'hessians'}; % name of the fields to be saved in excel file
if ~exist([save_path 'CG_' experiment_type '.xls'] , 'file')
    to_save(end+1,:) = {m duration spike_num_avg ET.total ET.coxes_all ET.Zs_all ET.hessian_all};
    xlswrite([save_path 'CG_' experiment_type '.xls'],to_save); % save the excel  file 
else
    xls_cur = num2cell(xlsread([save_path 'CG_' experiment_type '.xls']));
    [rows,~] = size(xls_cur);
    to_save(end+1:end+rows,:) = xls_cur ;
    to_save(end+1,:) = {m duration spike_num_avg ET.total ET.coxes_all ET.Zs_all ET.hessian_all};
    xlswrite([save_path 'CG_' experiment_type '.xls'],to_save);% save the excel  file 
end
beep % notification sound 