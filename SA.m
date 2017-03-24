%模拟退火算法
%作者：梁耀荣
%日期：2016/06/29
%操作：修改目标函数与约束范围
%结果：目标函数全局最小值及对应自变量取值

clear; clc; close all;
%% 定义目标函数（求最小值）
f = @(x)-(21.5+x(1)*sin(4*pi*x(1))+x(2)*sin(20*pi*x(2)));
%% 初始约束条件及参数
UB = [12.1;5.8];                                                            %上界（按列输入）
LB = [-3;4.1];                                                              %下界（按列输入）
r = 0.99;                                                                   %降温速率
T = 1000;                                                                   %初始温度
T_min = 0.001;                                                              %停止温度
Markov_length = 100;                                                        %每个温度扰动次数
p = 0.1;                                                                    %扰动强度（p*区间长度）扰动强度需要足够大，才能让它跳出局部最小
                                                                            %且随着接受概率下降，跳出局部最小后不容易跳出来，能有跟多机会收敛到全局最小
Num_x = size(UB,1);
%% 初始化
x(:,1) = LB+(UB-LB).*rand(Num_x,1);                                         %产生变量随机值
y(1) = f(x(:,1));
%% 模拟退火
i = 1;
while T>T_min,
    best_x = x(:,i);
    best_y = y(i);
    %% 同一温度内多次扰动
    for j = 1:Markov_length,      
        new_x = best_x+p*rand()*sign(unifrnd(-1,1,Num_x,1)).*(UB-LB);       %产生随机扰动。注意！每个变量移动方向都是随机的，不能同增同减
        if norm(double(new_x>UB))>0,                                        %确保所有自变量都在约束区间内
            new_x = UB;
        elseif norm(double(new_x<LB))>0,
            new_x = LB;
        end
        new_y = f(new_x);
        dy =new_y - best_y;
        if dy < 0 || exp(-dy/T) > rand(),                                   %！！千万注意概率判别公式中-dy是求最小值的，若求最大值，应该是dy
            best_y = new_y;
            best_x = new_x;            
        end
    end
    %% 记录数据，降温
    y(i+1) = best_y;
    x(:,i+1) = best_x;
    i = i+1;
    T = T*r;
end
%% 画图
plot(1:length(y),y);

min_y = min(y)
pos = find(y ==min_y);
min_x = x(:,pos(1))
