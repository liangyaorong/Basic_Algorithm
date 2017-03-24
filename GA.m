%�Ŵ��㷨
%���ߣ���ҫ��
%���ڣ�2016/06/28
%�������޸�Ŀ�꺯����Լ����Χ
%�����Ŀ�꺯��ȫ�����ֵ����Ӧ�Ա���ȡֵ

clear; clc; close all
global UB; global LB; global m; global len_m; global f
%% ����Ŀ�꺯���������ֵ��
f = @(x)21.5+x(1)*sin(4*pi*x(1))+x(2)*sin(20*pi*x(2));
%f = @(x) (sin(x(1))+cos(x(2)));
%% ��ʼԼ����Χ������
UB = [12.1;5.8];                                                            %�Ͻ磨�������룩
LB = [-3;4.1];                                                              %�½磨�������룩
precision = 4;                                                              %Ҫ�󾫶�С�����4λ
m = ceil(log2((UB-LB)*10^precision));                                       %��������Ⱦɫ����
len_m = size(m,1);
popsize = 100;                                                              %��ʼ��Ⱥ��С
cross_percent = 0.5;                                                        %�ӽ�����
mutation_percent = 0.001;                                                   %ͻ�����
MaxG = 500;                                                                 %����������
%% ��ʼ����ȺȾɫ��
population = round(rand(popsize,sum(m)));                                   %������ʽ
%% �������Ⱥ����ָ�꣨��Ӧ�ȣ������ͣ��ۻ����ʣ�
population = mat2cell(population,popsize,m');                               %Ԫ��������ʽ������ת��
[eval,x,Q] = Generation_value(population);
%% ��¼����
y = []; evl_mean = []; max_X = [];
y(1) = max(eval);                                                           %��¼�ô���Ӧ�����ֵ
eval_mean(1) = mean(eval);                                                  %��¼�ô���Ӧ�Ⱦ�ֵ
max_pos = find(eval == max(eval));
max_x(1,:) = x(max_pos(1),:);                                               %��¼�ô������Ӧ�ȶ�Ӧ������                                                 
%% ѭ������ 
for G = 1:MaxG,
    %% ����
    population = cell2mat(population);                                      %ת�ؾ�����ʽ���ڸ��ƣ����估����
    new_population = zeros(popsize,sum(m));
    for i = 1:popsize,
        copy_randnum = rand();
        copy_pos = find(Q >= copy_randnum);                                 %ȷ��Ҫ���Ƶ�Ⱦɫ��λ��
        new_population(i,:) = population(copy_pos(1),:);                    %����
    end
    %% ���䣨���㽻�䣩
    cross_randnum = rand(popsize,1);
    cross_pos = find(cross_randnum <= cross_percent);                       %ȷ����Ҫ�����Ⱦɫ��
    cross_number = size(cross_pos,1);
    if rem(cross_number,2)==1,                                              %ȷ�����뽻�����ĿΪ˫
        cross_number =cross_number-1;
    end
    for i = 1:2:cross_number,                                               %�������
        cross_point = randi(sum(m));                                        %ȷ��Ⱦɫ�忪ʼ�����Ľڵ�
        gene = new_population(cross_pos(i),cross_point:end);                %����
        new_population(cross_pos(i),cross_point:end) = new_population(cross_pos(i+1),cross_point:end);
        new_population(cross_pos(i+1),cross_point:end) = gene;
    end
    %% ͻ��
    mutation_randnum = rand(popsize,sum(m));
    [mutationX,mutationY] = find(mutation_randnum <= mutation_percent);     %ȷ��ͻ��Ļ���
    new_population(mutationX,mutationY) = 1-new_population(mutationX,mutationY);%ͻ�䣨0-1��ת��   
    %% ��һ��
    new_population = mat2cell(new_population,popsize,m');
    [eval,x,Q] = Generation_value(new_population);                          %������һ������   
    %% ��¼����
    y(G) = max(eval);
    eval_mean(G) = mean(eval);
    max_pos = find(eval == max(eval));
    max_x(G,:) = x(max_pos(1),:);

    population = new_population;   
end
%% ��ͼ
figure
plot(1:length(y),y,1:length(eval_mean),eval_mean)

max_y = max(y)
x_pos = find(y==max_y);
max_x = max_x(x_pos(1),:)
