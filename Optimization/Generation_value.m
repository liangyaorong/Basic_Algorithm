%����Ŵ��㷨GA.mʹ��
%������Ⱥ��Ԫ��������ʽ��ÿ���Ա�����Ⱦɫ��Ϊһ�о��󣩣��������Ⱥ�������
%����evalΪÿ��Ⱦɫ�����Ӧ�ȣ�xΪÿ��Ⱦɫ������ͣ�QΪ�ۻ�����

function [eval,x,Q] = Generation_value(population)
global UB; global LB; global m; global len_m; global f;
%% ����ȺȾɫ�����Ӷ�����ת��Ϊʮ����
x = {};
for i = 1:len_m,%����ת��
    x{i} = bin2dec(num2str(population{1,i}));
end
x = cell2mat(x);
len_x = size(x,1);
%% ����Ⱦɫ�������
x = repmat(LB',len_x,1) + x.*repmat(((UB-LB)./(2.^m-1))',len_x,1);
%% ������Ⱥÿ��Ⱦɫ����Ӧ��
eval = [];
for i = 1:len_x,
    eval(i,1) = f(x(i,:));
end
%% ��Ⱥÿ��Ⱦɫ�屻������
P_copy = eval/sum(eval);
%% �ۻ�����
Q = cumsum(P_copy);
