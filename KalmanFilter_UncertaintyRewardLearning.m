%% README
%firt we plot the different learning paradigms using Rescola-Wagner (RW)
%rule 
%the next part is implementation of Kalman filter method to explain
%blocking and unblocking in conditioning. 'Uncertainty and Learning',
%'Peter Dayan Angela J Yu', 2002
%%
clear all ;
clc ;
%%
N = 10000 ; % number of trials
LR = 0.25 ; % learning rate
p = 0.5 ; % pretrain rate 
n_u = 1 ; % number of stimuli
w = zeros(n_u , N) ; % weights
r = zeros(1 , N) ; % reward
u = zeros(n_u , N) ; % input stimuli
%% extinction
N = 10000 ; % number of trials
LR = 0.001 ; % learning rate
p = 0.5 ; % pretrain rate 
n_u = 1 ; % number of stimuli
w = zeros(n_u , N) ; % weights
r = zeros(1 , N) ; % reward
r(1 , 1:N*p) = 1 ;
u = ones(n_u , N) ; % input stimuli
for i = 1:N-1
    w(: , i+1) = w(: , i) + LR.*(r(1 , i) - w(: , i)*u(: , i)).*u(: , i) ;
end
scatter((1:1:N) , w , 2 , 'filled') ;
xlabel('trials') ;
ylabel('wheght') ;
title('extinction') ;
%% partial
N = 10000 ; % number of trials
LR = 0.001 ; % learning rate
p = 0.5 ; % pretrain rate 
n_u = 1 ; % number of stimuli
w = zeros(n_u , N) ; % weights
r = ones(1 , N) ; % reward
u = ones(n_u , N) ; % input stimuli
per = 0.1 ; % persentsge
per = randperm(N , N*per) ;
r(per) = 0 ;
for i = 1:N-1
    w(: , i+1) = w(: , i) + LR.*(r(1 , i) - w(: , i)*u(: , i)).*u(: , i) ;
end
scatter((1:1:N) , w , 2 , 'filled') ;
xlabel('trials') ;
ylabel('wheght') ;
title('partial') ;
%% blocking
N = 10000 ; % number of trials
LR = 0.001 ; % learning rate
p = 0.8 ; % pretrain rate 
n_u = 2 ; % number of stimuli
w = zeros(n_u , N) ; % weights
r = ones(1 , N) ; % reward
u = ones(n_u , N) ; % input stimuli
u(2 , 1:N*p) = 0 ;
for i = 1:N-1
    w(: , i+1) = w(: , i) + LR.*(r(1 , i) - w(: , i)'*u(: , i)).*u(: , i) ;
end
scatter((1:1:N) , w(1 , :) , 2 , 'filled') ;
xlabel('trials') ;
ylabel('wheght') ;
title('blocking') ;
hold on ;
scatter((1:1:N) , w(2 , :) , 2 , 'filled') ;
legwend('w_1' , 'w_2') ;
%% inhibitory
N = 100000 ; % number of trials
LR = 0.001 ; % learning rate
p = 0.5 ; % pretrain rate 
n_u = 2 ; % number of stimuli
w = zeros(n_u , N) ; % weights
r = zeros(1 , N) ; % reward
u = ones(n_u , N) ; % input stimuli
per = 0.8 ; % persentsge
per = randperm(N , N*per) ;
u(2 , per) = 0 ;
r(1 , per) = 1 ;
for i = 1:N-1
    w(: , i+1) = w(: , i) + LR.*(r(1 , i) - w(: , i)'*u(: , i)).*u(: , i) ;
end
scatter((1:1:N) , w(1 , :) , 2 , 'filled') ;
xlabel('trials') ;
ylabel('wheght') ;
title('inhibitory') ;
hold on ;
scatter((1:1:N) , w(2 , :) , 2 , 'filled') ;
legwend('w_1' , 'w_2') ;
%% overshadow
N = 20000 ; % number of trials
LR = 0.001 ; % learning rate
p = 0.5 ; % pretrain rate 
n_u = 2 ; % number of stimuli
w = zeros(n_u , N) ; % weights
w(1 , 1) = 0.05 ;
r = ones(1 , N) ; % reward
r(1 , 1:N*p) = 1 ;
u = ones(n_u , N) ; % input stimuli
for i = 1:N-1
    w(: , i+1) = w(: , i) + LR.*(r(1 , i) - w(: , i)'*u(: , i)).*u(: , i) ;
end
scatter((1:1:N) , w(1 , :) , 2 , 'filled') ;
xlabel('trials') ;
ylabel('wheght') ;
title('overshadow') ;
hold on ;
scatter((1:1:N) , w(2 , :) , 2 , 'filled') ;
legend('w_1' , 'w_2') ;
%% kalman filter
%finding weights
N = 20 ;
n_u = 2 ;
sigma = sqrt(0.1) ;
w = ones(n_u , N) ;
v = normrnd(0 , sigma , n_u , N) ;
for t = 1:N-1
    w(: , t+1) = w(: , t) + v(: , t) ;
end
w = 2*w/max(max(abs(w))) ;
W = 0 ;
for i = 1:N
    W = W + v(: , i)*v(: , i)' ;
end
W = W./N ;
a = reshape(w(: , 1:N/5) , [2*N/5 , 1]) ;
a1 = reshape(w(: , 1:4*N/5) , [2*4*N/5 , 1]) ;
b1 = normpdf((-5:0.2:5) , mean(a1) , std(a1)) ;
b = normpdf((-5:0.2:5) , mean(a) , std(a)) ;
plot(500*b+N/5 , (-5:0.2:5)) ;
hold on ;
plot(500*b1+4*N/5 , (-5:0.2:5)) ;
%figure ;
plot(w(1 , :)) ; 
hold on ;
plot(w(2 , :)) ;
legend('pdf1' , 'pdf2' , 'w_1' , 'w_2') ;
%% finding the inpiut and reward
% blocking
ta = sqrt(0.005) ;
etha = normrnd(0 , ta , N , 1) ;
V = etha'*etha/N ;
x = ones(N , n_u) ;
x(1:N/2 , 2) = 0 ;
r = diag(x*w) + etha ;
%% blocking
w_hat = zeros(n_u , N1) ;
covar = 0.64*ones(n_u , N1) ;
covar_ = 0.64*ones(n_u , N1) ;
g = [] ;
for t = 1:N1-1
    C = x(t , :) ;
    G = (covar(t) + W)/(covar(t) + W + ta^2) ;
    %covar(t+1) = covar_(t+1)-G*C*covar_(t+1) ;
    covar(t+1) = ((ta^2)*(covar(t) + W))/(covar(t) + ta^2 + W) ;
    w_hat(t+1) = w_hat(t) + G*(1 - w_hat(t)) ;
end
figure ;
plot(w_hat) ;
xlim([1 N1]) ;
ylim([0 1.5]) ;
title('mean') ;
xlabel('time') ;
ylabel('emstimated weight') ;
figure ;
plot(covar) ;
xlim([1 N1]) ;
title('variance') ;
xlabel('time') ;
ylabel('emstimated uncertainty') ;
w_hat_1 = zeros(n_u , N1) ;
w_hat_1(1 , 1) = w_hat(1 , end) ;
covar2 = 0.64*ones(1 , N1) ;
covar2_ = 0.64*ones(1 , N1) ;
covar1 = covar(1 , end)*ones(1 , N1) ;
covar1_ = covar(1 , end)*ones(1 , N1) ;
%%
covar1 = covar(end)*ones(1 , N1) ;
covar2 = covar ;
for t = 1:N1-1
    C = x(t , :) ;
    G1 = (covar1(t) + W(1 , 1))/(covar1(t) + covar2(t) + W(1 , 1) + W(2 , 2) + ta^2) ;
    G2 = (covar1(t) + W(2 , 2))/(covar1(t) + covar2(t) + W(1 , 1) + W(2 , 2) + ta^2) ;
    covar1_(t+1) = covar1(t) + W(1 , 1) ;
    covar2_(t+1) = covar2(t) + W(2 , 2) ;
    %covar1(t+1) = covar1_(t+1)-G1*covar1_(t+1) ;
    %covar2(t+1) = covar2_(t+1)-G2*covar2_(t+1) ;
    w_hat_1(1 , t+1) = w_hat_1(1 , t) + G1*(1 + 0.05 - w_hat_1(1 , t) - w_hat_1(2 , t)) ;
    w_hat_1(2 , t+1) = w_hat_1(2 , t) + G2*(1 + 0.05 - w_hat_1(1 , t) - w_hat_1(2 , t)) ;
end
figure ;
plot(w_hat_1(1 , :)) ;
hold on ;
plot(w_hat_1(2 , :)) ;
xlim([1 N1]) ;
ylim([0 1.5]) ;
title('mean') ;
xlabel('time') ;
ylabel('emstimated weight') ;
figure ;
plot(covar1) ;
hold on ;
plot(covar2) ;
xlim([1 N1]) ;
title('variance') ;
xlabel('time') ;
ylabel('emstimated uncertainty') ;
%%
covar_final1 = [covar , covar1] ;
covar_final2 = [zeros(1 ,N1-1) , covar , covar(end)] ;
w_hat_final1 = [w_hat , w_hat_1(1 , :)] ;
w_hat_final2 = [zeros(1 , N1-1) , w_hat_1(2 , :) , w_hat_1(2 , end)] ;
subplot(2 , 1 , 1) ;
plot(covar_final1) ;
hold on ;
plot((10:1:20) , [covar , covar(end)] , '--') ;
xline(10) ;
xlim([1  N]) ;
ylim([0 1]) ;
title('variance') ;
xlabel('time') ;
ylabel('emstimated uncertainty') ;
legend('\sigma_1^2' , '\sigma_2^2') ;
subplot(2 , 1 , 2) ;
plot(w_hat_final1) ;
hold on ;
plot(w_hat_final2 , '--') ;
xline(10) ;
xlim([1  N]) ;
title('mean') ;
xlabel('time') ;
ylabel('emstimated weight') ;
legend('w_1' , 'w_2') ;
%% unbolcking
p =1 ;
N = 10 ; 
n_u = 2 ;
ta = sqrt(0.005) ;
sigma = sqrt(0.1) ;
[w_hat_final , covariance_final] = kalman(N , n_u , ta , sigma , p) ;
%% unblocking pdf
k = 0 ;
for i = [1 , 10 , 20]
mean = reshape(w_hat_final(: , i) , [1 , 2]) ;
sigma = reshape(covariance_final(: , : , i) , [2 , 2]) ;
a = -1:0.2:2 ;
[X1,X2] = meshgrid(a , a);
X = [X1(:) X2(:)];
y = mvnpdf(X , mean , sigma) ;
y = reshape(y,length(a),length(a));
k = k + 1 ;
subplot(1 , 3 , k) ;
colormap(gray) ;
contourf(a , a , y) ;
title(['t=' num2str(i)]) ;
xlabel('w_1') ;
ylabel('w_2') ;
end
%%
N = 100 ;
n_u = 1 ;

sigma = sqrt(0.1) ;
w = ones(n_u , N) ;
v = normrnd(0 , sigma , n_u , N) ;

c = zeros(1 , N) ;
c(45) = 1 ;
c(90) = 1 ;
lam = sqrt(25) ;
pi = normrnd(0 , lam , n_u , N) ;

for t = 1:N-1
    w(: , t+1) = w(: , t) + v(: , t) + c(t)*pi(t) ;
end
for i = 1:N
    W = W + v(: , i)*v(: , i)' ;
end
W = W./N ;

ta = sqrt(0.05) ;
etha = normrnd(0 , ta , N , 1) ;
V = etha'*etha/N ;
x = ones(N , n_u) ;
r = diag(x*w) + etha ;
plot((1:1:N) , w , 'o') ;
hold on ;
plot((1:1:N) , r , 'x') ;
xlabel('time') ;

w_hat = zeros(n_u , N) ;
covar = 0.64*ones(n_u , N) ;
covar_ = 0.64*ones(n_u , N) ;
g = [] ;
for t = 1:N-1
    C = x(t , :) ;
    G = (covar(t) + W)/(covar(t) + W + ta^2) ;
    %covar(t+1) = covar_(t+1)-G*C*covar_(t+1) ;
    covar(t+1) = ((ta^2)*(covar(t) + W))/(covar(t) + ta^2 + W) ;
    w_hat(t+1) = w_hat(t) + G*(r(t) - w_hat(t)) ;
end
scatter((1:1:N) , w_hat , 'filled') ;
legend('w' , 'r' , 'predicted w') ;
%% function declaration
function [w_hat_final , covariance_final] = kalman(N , n_u , ta , sigma , p) 
etha = normrnd(0 , ta , 2*N , 1) ;
V = etha'*etha/(2*N) ;

w = ones(n_u , 2*N) ;
v = normrnd(0 , sigma , n_u , 2*N) ;
for t = 1:2*N-1
    w(: , t+1) = w(: , t) + v(: , t) ;
end
w = 2*w/max(max(abs(w))) ;
W = 0 ;
for i = 1:2*N
    W = W + v(: , i)*v(: , i)' ;
end
W = W./(2*N) ;

% first part
covariancei = [0.64 , 0 ; 0 , 0.64] ;
covariance = zeros(2 , 2 , N) ;
covariance(: , : , 1) = covariancei ;
w_hat = zeros(n_u , N) ;
covariance1 = covariance ;
r = ones(1 , N) ;
x = ones(N , n_u) ;
for t = 1:N-1
    covariance1(: , : , t+1) = covariance(: , : , t) + W ;
    C = x(t , :) ;
    G = (reshape(covariance1(: , : , t+1) , [2 , 2])*C')/(C*reshape(covariance1(: , : , t+1) , [2 , 2])*C' + ta^2) ;
    covariance(: , : , t+1) = covariance1(: , : , t+1) - G*C*reshape(covariance1(: , : , t+1) , [2 , 2]) ;
    w_hat(: , t+1) = w_hat(: , t) + G*(r(t) - C*w_hat(: , t)) ;
end

% second part
covariance2 = zeros(2 , 2 , N) ;
covariance2(: , : , 1) = covariance(: , : , end) ;
w_hat2 = zeros(n_u , N) ;
w_hat2(: , 1) = w_hat(: , end) ;
covariance12 = covariance2 ;
r = ones(1 , N) ;
x = ones(N , n_u) ;
x(: , 2) = 0 ;
for t = 1:N-1
    covariance12(: , : , t+1) = covariance2(: , : , t) + W ;
    C = x(t , :) ;
    G = (reshape(covariance12(: , : , t+1) , [2 , 2])*C')/(C*reshape(covariance12(: , : , t+1) , [2 , 2])*C' + ta^2) ;
    covariance2(: , : , t+1) = covariance12(: , : , t+1) - G*C*reshape(covariance12(: , : , t+1) , [2 , 2]) ;
    w_hat2(: , t+1) = w_hat2(: , t) + G*(r(t) - C*w_hat2(: , t)) ;
end

% final part
w_hat_final = [w_hat , w_hat2] ;
covariance_final = cat(3 , covariance , covariance2) ;

% ploting 
if p
    subplot(2 , 1 , 1) ;
    plot(w_hat_final(1 , :)) ;
    hold on ;
    plot(w_hat_final(2 , :)) ;
    xline(N+1 , '--') ;
    xlim([1 2*N]) ;
    ylim([0 1.5]) ;
    title('mean') ;
    xlabel('time') ;
    ylabel('emstimated weight') ;
    legend('w_1' , 'w_2') ;
    subplot(2 , 1 , 2) ;
    plot(reshape(covariance_final(1 , 1 , :) , [1 , 2*N])) ;
    hold on ;
    plot(reshape(covariance_final(2 , 2 , :) , [1 , 2*N])) ;
    plot(reshape(covariance_final(1 , 2 , :) , [1 , 2*N])) ;
    xline(N+1 , '--') ;
    xlim([1 2*N]) ;
    title('variance') ;
    xlabel('time') ;
    ylabel('emstimated uncertainty') ;
    legend('\sigma_1^2' , '\sigma_2^2' , '\sigma_12') ;
end
end