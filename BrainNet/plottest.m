% 画脑区节点图
% 用来生成edge文件
clear all
load('Net.mat')
idx{1,1}=[30,51];    
idx{1,2}=[15,38];
idx{1,3}=[4,40];
idx{1,4}=[18,19];
idx{1,5}=[33,47];
idx{1,6}=[1,26];
idx{1,7}=[4,49];
idx{1,8}=[7,18];
idx{1,9}=[4,26];
idx{1,10}=[11,27];
idx{1,11}=[1,42];
idx{1,12}=[4,31];
idx{1,13}=[4,25];
idx{1,14}=[4,50];
idx{1,15}=[1,24];
idx{1,16}=[7,17];

net=zeros(116,116);
for i=1:length(idx)
    id=idx{1,i};
    for j=1:size(id,1)
        net(id(j,1),id(j,2))=Net(id(j,1),id(j,2));
        net(id(j,2),id(j,1))=Net(id(j,2),id(j,1));
    end
end
X=net;
dlmwrite('6Repeat.edge', X, 'precision', '%5f', 'delimiter', '\t')     % 将文件保存到ADNC.edge文件
