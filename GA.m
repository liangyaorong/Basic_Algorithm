%遗传算法
%作者：梁耀荣
%日期：2016/06/28
%操作：修改目标函数与约束范围
%结果：目标函数全局最大值及对应自变量取值

clear; clc; close all
global UB; global LB; global m; global len_m; global f
%% 定义目标函数（求最大值）
f = @(x)21.5+x(1)*sin(4*pi*x(1))+x(2)*sin(20*pi*x(2));
%f = @(x) (sin(x(1))+cos(x(2)));
%% 初始约束范围及参数
UB = [12.1;5.8];                                                            %上界（按列输入）
LB = [-3;4.1];                                                              %下界（按列输入）
precision = 4;                                                              %要求精度小数点后4位
m = ceil(log2((UB-LB)*10^precision));                                       %计算所需染色体数
len_m = size(m,1);
popsize = 100;                                                              %初始种群大小
cross_percent = 0.5;                                                        %杂交概率
mutation_percent = 0.001;                                                   %突变概率
MaxG = 500;                                                                 %最大进化代数
%% 初始化种群染色体
population = round(rand(popsize,sum(m)));                                   %矩阵形式
%% 计算该种群各项指标（适应度，表现型，累积概率）
population = mat2cell(population,popsize,m');                               %元胞数组形式，便于转化
[eval,x,Q] = Generation_value(population);
%% 记录数据
y = []; evl_mean = []; max_X = [];
y(1) = max(eval);                                                           %记录该代适应度最大值
eval_mean(1) = mean(eval);                                                  %记录该代适应度均值
max_pos = find(eval == max(eval));
max_x(1,:) = x(max_pos(1),:);                                               %记录该代最大适应度对应表现型                                                 
%% 循坏进化 
for G = 1:MaxG,
    %% 复制
    population = cell2mat(population);                                      %转回矩阵形式便于复制，交配及变异
    new_population = zeros(popsize,sum(m));
    for i = 1:popsize,
        copy_randnum = rand();
        copy_pos = find(Q >= copy_randnum);                                 %确定要复制的染色体位置
        new_population(i,:) = population(copy_pos(1),:);                    %复制
    end
    %% 交配（单点交配）
    cross_randnum = rand(popsize,1);
    cross_pos = find(cross_randnum <= cross_percent);                       %确定需要交配的染色体
    cross_number = size(cross_pos,1);
    if rem(cross_number,2)==1,                                              %确保参与交配的数目为双
        cross_number =cross_number-1;
    end
    for i = 1:2:cross_number,                                               %两两配对
        cross_point = randi(sum(m));                                        %确定染色体开始互换的节点
        gene = new_population(cross_pos(i),cross_point:end);                %交配
        new_population(cross_pos(i),cross_point:end) = new_population(cross_pos(i+1),cross_point:end);
        new_population(cross_pos(i+1),cross_point:end) = gene;
    end
    %% 突变
    mutation_randnum = rand(popsize,sum(m));
    [mutationX,mutationY] = find(mutation_randnum <= mutation_percent);     %确定突变的基因
    new_population(mutationX,mutationY) = 1-new_population(mutationX,mutationY);%突变（0-1反转）   
    %% 新一代
    new_population = mat2cell(new_population,popsize,m');
    [eval,x,Q] = Generation_value(new_population);                          %计算新一代数据   
    %% 记录数据
    y(G) = max(eval);
    eval_mean(G) = mean(eval);
    max_pos = find(eval == max(eval));
    max_x(G,:) = x(max_pos(1),:);

    population = new_population;   
end
%% 画图
figure
plot(1:length(y),y,1:length(eval_mean),eval_mean)

max_y = max(y)
x_pos = find(y==max_y);
max_x = max_x(x_pos(1),:)
