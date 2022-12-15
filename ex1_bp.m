% ex1_bp.m
clc;clear;close all;

%% 参数初始化
l = 0.009;    % 学习率
alfa = 0.05;  % 动量因子

cells1 = 20;  % 隐层神经元个数
cells2 = 10;

w1 = rand(cells1,3);       % 随机赋值第一层连接权系数 [20  3]
w2 = rand(cells2,cells1);  % 随机赋值第二层连接权系数 [10  20]
w3 = rand(1,cells2);       % 随机赋值第三层连接权系数 [1   10]

yw1 = rand(cells1,1);  % 随机赋值第一层输出阈值 [20  1]
yw2 = rand(cells2,1);  % 随机赋值第二层输出阈值 [10  1]
yw3 = rand;            % 随机赋值第三层输出阈值 [1]

ts = 0.001;
%%%%%%%%%%%%%%%%%%%%%%%%%%
n = 500;  % 样本数
% n = 1000;  % 样本数
%%%%%%%%%%%%%%%%%%%%%%%%%%
yn = rand(1,n);  % 随机赋值输出(预测）
y  = rand(1,n);  % 随机赋值输出(真实）

counts = 1;  % 计数值初始化

x = [0,0,0]';  % 输入

u_1 = 0;  % 上一时刻的输入
y_1 = 0;  % 上一时刻的输出
y_2 = 0;  % 上上一时刻的输出

times = 300;  % 训练轮数
e = zeros(1,times);  % 均方差初始值设为0

%% 学习过程
for i = 1:times  % 学习轮数
    ei = 0;
    for a = 1:n  % 样本数
        time(a) = a*ts;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%
        % 系统输入
%         u(a) = 0.50*cos(6*pi*a*ts);
        u(a) = -0.75*sin(12*pi*a*ts);
        %%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % 系统真实模型
        y(a) = (y_1 - y_2) / sqrt(1 + y_1^2) + u_1^3;
        
        net1 = w1*x - yw1;     % 第一层网络的输入 [20, 1]
        out1 = logsig(net1);   % 第一层网路的输出 [20, 1]
        net2 = w2*out1 - yw2;  % 第二层网络的输入 [10, 20]*[20 ,1]=[10, 1]
        out2 = logsig(net2);   % 第二层网络的输出 [10, 1]
        net3 = w3*out2 - yw3;  % 第三层网络的输出 [1]
        yn(a)= net3;           % 第三层网络的输出 [1]
        
        det3 = y(a) - yn(a);  % 计算偏差 [1]
        det2 = (det3 *w3) * out2 * (1-out2);  % ([1, 10]'*[10 ,1])*[10, 1] = [10, 1]
        det1 = (det2'*w2) * out1 * (1-out1); % [20, 1]
        
        w1 = w1 + det1*x'*l;       % [20, 2]
        w2 = w2 + (det2*out1')*l;  % [10, 20]
        w3 = w3 + (det3*out2')*l;  % [1, 10]
        
        yw1 = yw1 - det1*l;
        yw2 = yw2 - det2*l;
        yw3 = yw3 - det3*l;
        
        ei = ei + det3^2 / 2;
        e(i) = ei;
        
        % 更新输入
        x(1) = u(a);
        x(2) = y(a);
        x(3) = y_1;
        
        y_2 = y_1;
        u_1 = u(a);
        y_1 = y(a);
        
    end  % 结束一次样本遍历
    
    if ei < 0.008
        break;
    end
    counts = counts + 1;
end  % 结束学习

%% 绘图
figure(1);
subplot(2,1,1);
plot(time,y,'b-',time,yn,'r-');
legend('真实模型输出y', 'BP网络输出y')
grid on
title('输出响应');
xlabel('t');
ylabel('y');

counts = counts - 1;
if (counts < times)
    count = 1:counts;
    sum = counts;
else
    count = 1:times;
    sum = times;
end

subplot(2,1,2);
plot(count,e(1:sum),'r-');
grid on;
title('学习曲线');
xlabel('迭代次数');
ylabel('MSE');

% END