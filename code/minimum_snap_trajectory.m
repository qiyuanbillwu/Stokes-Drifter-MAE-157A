close all
clear

% global constants
g = 9.81;
m = 0.5;
dt = 0.01;

% conditions subject to our disposal, like start position
x0 = 1;
y0 = -1;
z0 = 1;
x1 = -1.5;
y1 = 0;
z1 = 2.5;
x2 = 1;
y2 = 1;
z2 = 1;
vy = 4.0; % moving through the gate at this speed
T = 5; % apply some thrust to maintain orientation

t0 = 0;
t1 = 1.5;
t2 = 3;

% boundary conditions 
% hovering at state 0 and 2, passing through the gate at state 1
r0 = [x0 y0 z0];
v0 = [0 0 0];
a0 = [0 0 0];
j0 = [0 0 0];
r1 = [x1 y1 z1];
v1 = [0 vy 0];
a1 = [T/m 0 -g];
j1 = [0 0 0];
r2 = [x2 y2 z2];
v2 = v0;
a2 = a0;
j2 = j0;

A1 = compute_A(t0, t1);
A2 = compute_A(t1, t2);

b1 = [r0; v0; a0; j0; r1; v1; a1; j1];
b2 = [r1; v1; a1; j1; r2; v2; a2; j2];

a_1 = A1 \ b1;
a_2 = A2 \ b2;

f_r1 = @(t) [t.^0 t.^1 t.^2 t.^3 t.^4 t.^5 t.^6 t.^7] * a_1;
f_r2 = @(t) [t.^0 t.^1 t.^2 t.^3 t.^4 t.^5 t.^6 t.^7] * a_2;

f_v1 = @(t) [t-t t.^0 2*t.^1 3*t.^2 4*t.^3 5*t.^4 6*t.^5 7*t.^6] * a_1;
f_v2 = @(t) [t-t t.^0 2*t.^1 3*t.^2 4*t.^3 5*t.^4 6*t.^5 7*t.^6] * a_2;

f_a1 = @(t) [t-t t-t 2*t.^0 6*t.^1 12*t.^2 20*t.^3 30*t.^4 42*t.^5] * a_1;
f_a2 = @(t) [t-t t-t 2*t.^0 6*t.^1 12*t.^2 20*t.^3 30*t.^4 42*t.^5] * a_2;

f_j1 = @(t) [t-t t-t t-t 6*t.^0 24*t.^1 60*t.^2 120*t.^3 210*t.^4] * a_1;
f_j2 = @(t) [t-t t-t t-t 6*t.^0 24*t.^1 60*t.^2 120*t.^3 210*t.^4] * a_2;

f_s1 = @(t) [t-t t-t t-t t-t 24*t.^0 120*t.^1 360*t.^2 840*t.^3] * a_1;
f_s2 = @(t) [t-t t-t t-t t-t 24*t.^0 120*t.^1 360*t.^2 840*t.^3] * a_2;

time1 = t0:dt:t1;
time2 = t1:dt:t2;

r1 = f_r1(time1');
r2 = f_r2(time2');
v1 = f_v1(time1');
v2 = f_v2(time2');
a1 = f_a1(time1');
a2 = f_a2(time2');
j1 = f_j1(time1');
j2 = f_j2(time2');
s1 = f_s1(time1');
s2 = f_s2(time2');

x1 = r1(:,1);
y1 = r1(:,2);
z1 = r1(:,3);
x2 = r2(:,1);
y2 = r2(:,2);
z2 = r2(:,3);

figure(1)
plot3(x1, y1, z1)
hold on
plot3(x2, y2, z2)
hold off
xlabel('x (m)')
ylabel('y (m)')
zlabel('z (m)')
xlim([-2 2])
ylim([-2 2])
zlim([0 4])
legend('1st leg', '2nd leg')
grid on
title('trajectory')

figure(2)
plot(time1, x1, 'r')
hold on
plot(time1, y1, 'k')
plot(time1, z1, 'b')
plot(time2, x2, 'r')
plot(time2, y2, 'k')
plot(time2, z2, 'b')
hold off
legend('x', 'y', 'z')
xlabel('time (s)')
ylabel('position (m)')
title('position')

figure(3)
plot(time1, v1(:,1), 'r')
hold on
plot(time1, v1(:,2), 'k')
plot(time1, v1(:,3), 'b')
plot(time2, v2(:,1), 'r')
plot(time2, v2(:,2), 'k')
plot(time2, v2(:,3), 'b')
hold off
legend('x', 'y', 'z')
xlabel('time (s)')
ylabel('velocity (m/s)')
title('velocity')

figure(4)
plot(time1, a1(:,1), 'r')
hold on
plot(time1, a1(:,2), 'k')
plot(time1, a1(:,3), 'b')
plot(time2, a2(:,1), 'r')
plot(time2, a2(:,2), 'k')
plot(time2, a2(:,3), 'b')
hold off
legend('x', 'y', 'z')
xlabel('time (s)')
ylabel('acceleration (m/s^2)')
title('acceleration')

figure(5)
plot(time1, j1(:,1), 'r')
hold on
plot(time1, j1(:,2), 'k')
plot(time1, j1(:,3), 'b')
plot(time2, j2(:,1), 'r')
plot(time2, j2(:,2), 'k')
plot(time2, j2(:,3), 'b')
hold off
legend('x', 'y', 'z')
xlabel('time (s)')
ylabel('jerk (m/s^3)')
title('jerk')

figure(6)
plot(time1, s1(:,1), 'r')
hold on
plot(time1, s1(:,2), 'k')
plot(time1, s1(:,3), 'b')
plot(time2, s2(:,1), 'r')
plot(time2, s2(:,2), 'k')
plot(time2, s2(:,3), 'b')
hold off
legend('x', 'y', 'z')
xlabel('time (s)')
ylabel('snap (m/s^4)')
title('snap')

T_over_W_1 = sqrt(a1(:,1).^2 + a1(:,2).^2 + (a1(:,3)+g).^2) / g;
T_over_W_2 = sqrt(a2(:,1).^2 + a2(:,2).^2 + (a2(:,3)+g).^2) / g;

figure(7)
plot(time1, T_over_W_1);
hold on
plot(time2, T_over_W_2);
hold off
xlabel('time (s)')
ylabel('thrust to weight ratio')
legend('1st leg', '2nd leg')
title('thrust to weight')

% time = [time1 time2];
% x = [x1 x2];
% y = [y1 y2];
% z = [z1 z2];
% vx = [v1(:,1) v2(:,1)];
% vy = [v1(:,2) v2(:,2)];
% vz = [v1(:,3) v2(:,3)];
% ax = [a1(:,1) a2(:,1)];
% ay = [a1(:,2) a2(:,2)];
% az = [a1(:,3) a2(:,3)];
% jx = [j1(:,1) j2(:,1)];
% jy = [j1(:,2) j2(:,2)];
% jz = [j1(:,3) j2(:,3)];
% sx = [s1(:,1) s2(:,1)];
% sy = [s1(:,2) s2(:,2)];
% sz = [s1(:,3) s2(:,3)];
