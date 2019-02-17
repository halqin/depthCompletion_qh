clear all
close all
subplot(2,3,1);
x = linspace(0,10);
y1 = sin(x);
plot(x,y1)

subplot(2,3,[5,6]); 
y2 = sin(5*x);
plot(x,y2)