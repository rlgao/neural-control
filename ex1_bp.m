clc;clear;close all;

% function DualModelIden
%% 参数初始化
l = 0.009; % 学习率
alfa = 0.05; % 动量因子

cells1 = 20; % 隐层神经元个数
cells2 = 10;

w1 = rand(cells1,2);       % 随机赋值第一层连接权系数 [20 ,2]
w2 = rand(cells2,cells1);  % 随机赋值第二层连接权系数 [10, 20]
w3 = rand(1,cells2);       % 随机赋值第三层连接权系数 [1, 10]

yw1 = rand(cells1,1);  % 随机赋值第一层输出阈值 [20, 1]
yw2 = rand(cells2,1);  % 随机赋值第二层输出阈值 [10, 1]
yw3 = rand;            % 随机赋值第三层输出阈值 [1]

ts = 0.001; % 样本
n  = 1000;  % 样本数
yn = rand(1,n);  % 随机赋值输出(预测）
y  = rand(1,n);  % 随机赋值输出(真实）

counts = 1;  % 计数值初始化

x = [0,0]';  % 输入

u_1 = 0;  % 上一时刻的输入
y_1 = 0;  % 上一时刻的输出

times = 300; % 训练轮数
e = zeros(1,times); % 均方差初始值设为0

%% 学习过程
for i = 1:times % 学习轮数
    ei = 0;
    for a = 1:n % 样本数
        time(a) = a*ts;
        u(a) = 0.50*sin(3*2*pi*a*ts);
        y(a) = u_1^3 + y_1 / (1+y_1^2);
    
        net1=w1*x-yw1; % 第一层网络的输入 [20, 1]
        out1=logsig(net1); % 第一层网路的输出 [20, 1]
        net2=w2*out1-yw2; % 第二层网络的输入 [10, 20]*[20 ,1]=[10, 1]
        out2=logsig(net2); % 第二层网络的输出 [10, 1]
        net3=w3*out2-yw3; % 第三层网络的输出 [1]
        yn(a)=net3; % 第三层网络的输出 [1]
        
        det3=y(a)-yn(a); % 计算偏差 [1]
        det2=((det3*(w3))*out2)*(1-out2); % ([1, 10]'*[10 ,1])*[10, 1] = [10, 1]
        det1=((det2'*(w2))*out1)*(1-out1); % [20, 1]
     
        w1=w1+det1*x'*l; % [20, 2]
        w2=w2+(det2*out1')*l; % [10, 20]
        w3=w3+(det3*out2')*l; % [1, 10]
        
        yw1=-det1*l+yw1;
        yw2=-det2*l+yw2;
        yw3=-det3*l+yw3;
        
        ei=ei+det3^2/2;
        e(i)=ei;      
        
        x(1)=u(a); % 更新输入
        x(2)=y(a);
    
        u_1=u(a);
        y_1=y(a);
    end % 结束一次样本遍历
    
    if ei<0.008
        break;
    end
    counts=counts+1;
end  % 结束学习

%% 计算学习的曲线
x = [0,0]'; % 输入
yn_test = rand(1,n); % 随机赋值输出(预测）
y_test = rand(1,n); % 随机赋值输出(真实）
u_1 = 0; % 上一时刻的输入
y_1 = 0; % 上一时刻的输出
ts=0.1; % 样本
n = 1000; % 样本数
for a=1:n
    u(a)=sin(2*pi*a*ts/25) + sin(2*pi*a*ts/10);
    y_test(a)=u_1^3+y_1/(1+y_1^2);
 
    net1=w1*x-yw1;
    out1=logsig(net1);
    net2=w2*out1-yw2;
    out2=logsig(net2);
    net3=w3*out2-yw3;
    yn_test(a)=net3;
    
	x(1)=u(a);
	x(2)=y_test(a);
    
	u_1=u(a);
	y_1=y_test(a);
end
%% 绘图
figure(1);
subplot(2,1,1);
plot(time,y,'b-',time,yn,'r-');
legend('true', 'forecast')
grid on
title('BP学习方法逼近y=0.5*(1+cos(x))');
xlabel('x轴');
ylabel('y=0.5*(1+cos(x))');

if (counts<times)
    count=1:counts;
    sum=counts;
else 
    count=1:times;
    sum=times;
end

subplot(2,1,2);
plot(count,e(1:sum));
grid on;
title('BP算法学习曲线');
xlabel('迭代次数');
ylabel('Mean-Square-Error');

figure(2);
plot(time,y_test,'b-',time,yn_test,'r-');
legend('true', 'forecast')
grid on
title('BP学习方法逼近y=sin(2*pi*k/25) + sin(2*pi*k/10)');
xlabel('x轴');
ylabel('y=sin(2*pi*k/25) + sin(2*pi*k/10)');
% return
