%ģ���˻��㷨
%���ߣ���ҫ��
%���ڣ�2016/06/29
%�������޸�Ŀ�꺯����Լ����Χ
%�����Ŀ�꺯��ȫ����Сֵ����Ӧ�Ա���ȡֵ

clear; clc; close all;
%% ����Ŀ�꺯��������Сֵ��
f = @(x)-(21.5+x(1)*sin(4*pi*x(1))+x(2)*sin(20*pi*x(2)));
%% ��ʼԼ������������
UB = [12.1;5.8];                                                            %�Ͻ磨�������룩
LB = [-3;4.1];                                                              %�½磨�������룩
r = 0.99;                                                                   %��������
T = 1000;                                                                   %��ʼ�¶�
T_min = 0.001;                                                              %ֹͣ�¶�
Markov_length = 100;                                                        %ÿ���¶��Ŷ�����
p = 0.1;                                                                    %�Ŷ�ǿ�ȣ�p*���䳤�ȣ��Ŷ�ǿ����Ҫ�㹻�󣬲������������ֲ���С
                                                                            %�����Ž��ܸ����½��������ֲ���С�����������������и������������ȫ����С
Num_x = size(UB,1);
%% ��ʼ��
x(:,1) = LB+(UB-LB).*rand(Num_x,1);                                         %�����������ֵ
y(1) = f(x(:,1));
%% ģ���˻�
i = 1;
while T>T_min,
    best_x = x(:,i);
    best_y = y(i);
    %% ͬһ�¶��ڶ���Ŷ�
    for j = 1:Markov_length,      
        new_x = best_x+p*rand()*sign(unifrnd(-1,1,Num_x,1)).*(UB-LB);       %��������Ŷ���ע�⣡ÿ�������ƶ�����������ģ�����ͬ��ͬ��
        if norm(double(new_x>UB))>0,                                        %ȷ�������Ա�������Լ��������
            new_x = UB;
        elseif norm(double(new_x<LB))>0,
            new_x = LB;
        end
        new_y = f(new_x);
        dy =new_y - best_y;
        if dy < 0 || exp(-dy/T) > rand(),                                   %����ǧ��ע������б�ʽ��-dy������Сֵ�ģ��������ֵ��Ӧ����dy
            best_y = new_y;
            best_x = new_x;            
        end
    end
    %% ��¼���ݣ�����
    y(i+1) = best_y;
    x(:,i+1) = best_x;
    i = i+1;
    T = T*r;
end
%% ��ͼ
plot(1:length(y),y);

min_y = min(y)
pos = find(y ==min_y);
min_x = x(:,pos(1))
