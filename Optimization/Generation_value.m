%配合遗传算法GA.m使用
%输入种群（元胞数组形式，每个自变量的染色体为一列矩阵），计算该种群各项参数
%返回eval为每个染色体的适应度，x为每个染色体表现型，Q为累积概率

function [eval,x,Q] = Generation_value(population)
global UB; global LB; global m; global len_m; global f;
%% 将种群染色体编码从二进制转换为十进制
x = {};
for i = 1:len_m,%逐列转化
    x{i} = bin2dec(num2str(population{1,i}));
end
x = cell2mat(x);
len_x = size(x,1);
%% 计算染色体表现型
x = repmat(LB',len_x,1) + x.*repmat(((UB-LB)./(2.^m-1))',len_x,1);
%% 计算种群每个染色体适应度
eval = [];
for i = 1:len_x,
    eval(i,1) = f(x(i,:));
end
%% 种群每个染色体被复制率
P_copy = eval/sum(eval);
%% 累积概率
Q = cumsum(P_copy);
