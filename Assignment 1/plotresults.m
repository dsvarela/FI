function plotresults(trajEst,diagP,micPos)
%
% Function that plots the estimated trajectory and microphone positions on
% top of an image of the ground truth.
%
% function SFlabCompEstimGroundTruth(trajEst,micPos)
%
% INPUTS:
% trajEst   -  2X117-vector with the estimated trajectory, first row is
%              X-coordinate, second row is Y-coordinate.
% diagP     -  117X2-vector with the diagonal of the covariance matrix
% micPos    -  2x7-matrix with the microphone positions, first row is
%              X-coordinate, second row is Y-coordinate.
%
%
% Modified from SFlabCompEstimGroundTruth.m by
% Martin Skoglund and Karl Granstr�m
% 2009-03-24
% changes (11-2020): function name and added uncertainty plot

t = linspace(0,2*pi) ;
for j = 1:length(trajEst)
    Px(j,:) = 100*sqrt(diagP(j,1))*cos(t) + trajEst(1,j);
    Py(j,:) = 100*sqrt(diagP(j,2))*sin(t) + trajEst(2,j);
end

P(1,1:117,1:100) = Px;
P(2,1:117,1:100) = Py;

% Rotation matrix
R =[...
   0.999774521898456  -0.000924915849553   0.021214379401393;...
   0.001705399360543  -0.992326335159808  -0.123634688341604;...
   0.021165939046876   0.123642990415857  -0.992101009950745];
% Translation matrix
T =[...
  -0.583118758321543;...
   0.458078043186633;...
   1.412654371966365];
% Focal lengths
f =1.0e+003*[...
   2.389065133857016;...
   2.393358741136121];
% Principal point coordinates
c =1.0e+003*[...
   1.120043265067930;...
   0.861829343480461];

% coordinates in camera frame
g = [R T; 0 0 0 1];
PI0 = [eye(3) zeros(3,1)];
xc =PI0*g*[trajEst; zeros(1,size(trajEst,2)) ; ones(1,size(trajEst,2))];
mc =PI0*g*[micPos; zeros(1,size(micPos,2)) ; ones(1,size(micPos,2))];
Pc = zeros(3,size(P,2),100);
for i = 1:100
    Pc(:,:,i) =PI0*g*[P(:,:,i) ; zeros(1,size(P(:,:,i),2)) ; ones(1,size(P(:,:,i),2))];
end

% normalized coordinates
xn(1,:) = xc(1,:)./xc(3,:);
xn(2,:) = xc(2,:)./xc(3,:);
mn(1,:) = mc(1,:)./mc(3,:);
mn(2,:) = mc(2,:)./mc(3,:);
for i = 1:100
    Pn(1,:,i) = Pc(1,:,i)./Pc(3,:,i);
    Pn(2,:,i) = Pc(2,:,i)./Pc(3,:,i);
end


% pixel coordinates
K = [
    f(1)  0     c(1);
    0     f(2)  c(2);
    0     0     1];
xp = K*[xn ; ones(1,size(trajEst,2))];
mp = K*[mn ; ones(1,size(micPos,2))];
for i = 1:100
    Pp(:,:,i) = K*[Pn(:,:,i) ; ones(1,size(Pn(:,:,i),2))];
end

% Load image of ground truth
I=imread('SFlabGroundTruth.jpg');
I = rgb2gray(I);

% Compute axis min and max
[Iy,Ix] = size(I);
extra = 50;
xmin = min([0 xp(1,:) mp(1,:)])-extra;
xmax = max([Ix xp(1,:) mp(1,:)])+extra;
ymin = min([0 xp(2,:) mp(2,:)])-extra;
ymax = max([Iy xp(2,:) mp(2,:)])+extra;

% Plot
imagesc(I)
colormap(gray)
xlims = xlim;
xmin = xlims(1); xmax = xlims(2); % To change
ylims = ylim;
ymin = ylims(1); ymax = ylims(2); % To change
hold on
plot(xp(1,:),xp(2,:),'r.','markersize',10,'linewidth',2)
plot(mp(1,:),mp(2,:),'b.','markersize',25,'linewidth',2)

for j=1:117
    plot(reshape(Pp(1,j,:),[1 100]),reshape(Pp(2,j,:),[1 100]),'y')
    hold on;
end

set(gca, 'YDir','reverse')
hold off
title('Estimated trajectory in red, microphone positions in blue')
axis image
axis([xmin xmax ymin ymax])