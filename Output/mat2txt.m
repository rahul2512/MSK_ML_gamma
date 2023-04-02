clear
%load('JA.mat');
load('JRF_Braced.mat');
D = data;
S = {'Subject1' 'Subject2' 'Subject3' 'Subject4' 'Subject5' 'Subject6' 'Subject7' 'Subject8' 'Subject9' 'Subject10' 'Subject11' 'Subject12' 'Subject13' 'Subject14' 'Subject15' 'Subject16'};
R = {'RGF_1' 'RGF_2' 'RGF_3'};
%LIC RIC LASI surely gone
%I am just using
for i =1:16
    for j = 1:3
        T = [];
        for l = 1:length(label)
        T = [T  D.(S{i}).(R{j}).(label{l})];
%        filename = append('MA_' , S{i} ,  '_' ,  R{j} , '.txt')
        filename = append('Braced_JRF_' , S{i} ,  '_' ,  R{j} , '.txt')
        writematrix(T, filename)
        end
    end
end