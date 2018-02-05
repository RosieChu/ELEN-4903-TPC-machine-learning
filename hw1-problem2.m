% train data
x = '/Users/zym/Documents/tpc ml/hw/hw1-data/X_train.csv';
X_train = csvread(x);
y = '/Users/zym/Documents/tpc ml/hw/hw1-data/Y_train.csv';
Y_train = csvread(y);

[U,S,V] = svd(X_train,0);

w_RR= zeros(7,350);
df_lambda = zeros(1,5001);

for lambda = 0:5000 
    w_RR(:,lambda+1) = inv(lambda*eye(7)+X_train'*X_train)*X_train'*Y_train;
    for j = 1:7
        df_lambda(1,lambda+1) =  df_lambda(1,lambda+1) + S(j,j)^2/(lambda+S(j,j)^2);
    end
end

figure(1);
for i = 1:7
    plot(df_lambda,w_RR(i,:),'LineWidth',1.5);
    hold on;
end

title('Ridge Regression');
legend('Dim1','Dim2','Dim3','Dim4','Dim5','Dim6','Dim7','Location','southwest');

xlabel('df(¦Ë)');
ylabel('w_R_R');

saveas(gcf,'/Users/zym/Documents/tpc ml/hw/2_1.png');

% test data
x_test = '/Users/zym/Documents/tpc ml/hw/hw1-data/X_test.csv';
X_test = csvread(x_test);
y_test = '/Users/zym/Documents/tpc ml/hw/hw1-data/Y_test.csv';
Y_test = csvread(y_test);

w_RR= zeros(7,51);
Y = zeros(42,51);
rmse = zeros(1,51);
Y_sum = 0;

for lambda = 0:50 
    w_RR(:,lambda+1) = inv(lambda*eye(7)+X_train'*X_train)*X_train'*Y_train;
end    
Y = X_test * w_RR;

for lambda = 0:50 
    temp = 0;
    for i = 1:42
        temp = temp + (Y_test(i,1)-Y(i,lambda+1))^2;
    end 
    rmse(1,lambda+1) = sqrt(temp/42); 
end



lambda = 0:50;
figure(2);
plot(lambda,rmse,'LineWidth',1.5);

xlabel('¦Ë');
ylabel('RMSE');

title('Root Mean Squared Error');
saveas(gcf,'/Users/zym/Documents/tpc ml/hw/2_3.png');

%difffernt p
w_RR_1 = zeros(7,501);
w_RR_2 = zeros(13,501);
w_RR_3 = zeros(19,501);
X_train_2 = zeros(350,13);
X_train_3 = zeros(350,19);
X_test_2 = zeros(42,13);
X_test_3 = zeros(42,19);
Y = zeros(42,501);
rmse1 = zeros(1,501);
rmse2 = zeros(1,501);
rmse3 = zeros(1,501);

% update X_train
X_train_2(1:350,1:7) = X_train;
X_train_3(1:350,1:7) = X_train;
for i = 8:13
   X_train_2(:,i) = X_train(:,i-7).^2; 
end
X_train_3(:,8:13) = X_train_2(:,8:13);
for i = 14:19
    X_train_3(:,i) = X_train(:,i-13).^3;
end
% update X_test
X_test_2(1:42,1:7) = X_test;
X_test_3(1:42,1:7) = X_test;
for i = 8:13
   X_test_2(:,i) = X_test(:,i-7).^2; 
end
X_test_3(:,8:13) = X_test_2(:,8:13);
for i = 14:19
    X_test_3(:,i) = X_test(:,i-13).^3;
end


for lambda = 0:500 
    w_RR_1(:,lambda+1) = inv(lambda*eye(7)+X_train'*X_train)*X_train'*Y_train;
    w_RR_2(:,lambda+1) = inv(lambda*eye(13)+X_train_2'*X_train_2)*X_train_2'*Y_train;
    w_RR_3(:,lambda+1) = inv(lambda*eye(19)+X_train_3'*X_train_3)*X_train_3'*Y_train;
end  

Y1 = X_test * w_RR_1;
Y2 = X_test_2 * w_RR_2;
Y3 = X_test_3 * w_RR_3;

for lambda = 0:500 
    temp1 = 0;
    temp2 = 0;
    temp3 = 0;
    for i = 1:42
        temp1 = temp1 + (Y_test(i,1)-Y1(i,lambda+1))^2;
        temp2 = temp2 + (Y_test(i,1)-Y2(i,lambda+1))^2;
        temp3 = temp3 + (Y_test(i,1)-Y3(i,lambda+1))^2;
    end 
    rmse1(1,lambda+1) = sqrt(temp1/42); 
    rmse2(1,lambda+1) = sqrt(temp2/42); 
    rmse3(1,lambda+1) = sqrt(temp3/42); 
end


lambda = 0:500;
figure(3);
plot(lambda,rmse1,'LineWidth',1.5);
hold on;
plot(lambda,rmse2,'LineWidth',1.5);
plot(lambda,rmse3,'LineWidth',1.5);

% find the minimum value when p = 2 and p = 3.
[y1,x1] = findpeaks(-rmse2);
y1 = -y1;
[y2,x2] = findpeaks(-rmse3);
y2 = -y2;

legend('RMSE1(p=1)','RMSE2(p=2)','RMSE3(p=3)','Location','southeast');
title('Root Mean Squared Error');
xlabel('¦Ë');
ylabel('RMSE');
saveas(gcf,'/Users/zym/Documents/tpc ml/hw/2_4.png');
