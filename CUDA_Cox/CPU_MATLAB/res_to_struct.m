function out_struct = res_to_struct (path,n)
% res_to_struct function is used for converting the ELIF data in .rst
% format to a struct in order to making the process of applying cox method
% on it more convenient. An ELIF model neuron can be simulated using the
% software from the following website:
% http://www.tech.plymouth.ac.uk/infovis 
s = ['a = load (''',path,''', ''-ascii'');']; % load the file 
eval(s) 
T = length(a) / n ;  % find the array length of each spike train 
a=reshape(a,T,n); 
out_struct = struct ; % initializing the empty struct
for i = 1:n
   q = num2str(i);
   s= ['temp',q, '= find (a(:,', q , '));'] ; % finding the non-zero elements 
   eval (s);   
end
for i = 1:n
    s = ['out_struct(i).Target = temp', num2str(i),';']; % filling the Target part of the each struct dedicated for each neuron
    eval (s) 
     % The length of reference spike trains varies. follwoing lines defines 
     % the length of the longest reference spike train (maximum).
    l = ['maximum = max ([']; 
    for p = 1:n  
        if p~=i
           l = [l [' length(temp', num2str(p),  ')']];
           
        end
    end
    l = [l ']);'];
    eval (l)
    nn = 1 ; 
    % Filling the "reference" section of the struct for each neuron in the
    % network 
   for j = 1:n
       if j~=i
           js = num2str(j);
           nns = num2str(nn);
           s = ['difference = maximum - length(temp', js,');'];
           eval (s);
     s = ['temp(:,',nns,') = [temp', js,';zeros(difference,1)];'];
     eval (s);
     nn = nn + 1 ;
       end
   end
    out_struct(i).Ref = temp;
   clear temp ;
end
