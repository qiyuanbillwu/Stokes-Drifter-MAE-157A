close all
clear

% global constants
g = 9.81;
m = 0.5;

% initial conditions
r0 = [1 -2 2]; % position
v0 = [0 0 0]; % velocity
a0 = [0 0 0]; % acceleration

% thrust at the gate
T = 2;

% final conditions
rf = [0 0 0.5]; % position
af = [T/m 0 -g]; % acceleration
jf = [0 0]; % jerk
zf = 0; % vertical velocity (for z)

% final time
tf = 4;

% trajectories for xy and z are designed separately
% there are 6 known states, so a polynomial of degree 5 is needed 
A_xy = [tf^5 tf^4 tf^3; 20*tf^3 12*tf^2 6*tf; 60*tf^2 24*tf 6];
A_z = [tf^5 tf^4 tf^3; 5*tf^4 4*tf^3 3*tf^2; 20*tf^3 12*tf^2 6*tf];

b_xy = [rf(1:2) - r0(1:2); af(1:2); jf];
b_z = [rf(3) - r0(3); zf; af(3)];
a_xy = A_xy \ b_xy;
a_z = A_z \ b_z;
a = [a_xy a_z];

a_5 = a(1,:);
a_4 = a(2,:);
a_3 = a(3,:);
a_2 = a0;
a_1 = v0;
a_0 = r0;

f_x = @(t) a_5(1)*t.^5 + a_4(1)*t.^4 + a_3(1)*t.^3 + a_2(1)*t.^2 + a_1(1)*t + a_0(1);
f_y = @(t) a_5(2)*t.^5 + a_4(2)*t.^4 + a_3(2)*t.^3 + a_2(2)*t.^2 + a_1(2)*t + a_0(2);
f_z = @(t) a_5(3)*t.^5 + a_4(3)*t.^4 + a_3(3)*t.^3 + a_2(3)*t.^2 + a_1(3)*t + a_0(3);

f_vx = @(t) 5*a_5(1)*t.^4 + 4*a_4(1)*t.^3 + 3*a_3(1)*t.^2 + 2*a_2(1)*t + a_1(1);
f_vy = @(t) 5*a_5(2)*t.^4 + 4*a_4(2)*t.^3 + 3*a_3(2)*t.^2 + 2*a_2(2)*t + a_1(2);
f_vz = @(t) 5*a_5(3)*t.^4 + 4*a_4(3)*t.^3 + 3*a_3(3)*t.^2 + 2*a_2(3)*t + a_1(3);

f_ax = @(t) 20 * a_5(1)*t.^3 + 12 * a_4(1)*t.^2 + 6 * a_3(1)*t + 2 * a_2(1);
f_ay = @(t) 20 * a_5(2)*t.^3 + 12 * a_4(2)*t.^2 + 6 * a_3(2)*t + 2 * a_2(2);
f_az = @(t) 20 * a_5(3)*t.^3 + 12 * a_4(3)*t.^2 + 6 * a_3(3)*t + 2 * a_2(3);

time = 0:0.01:tf;

x = f_x(time);
y = f_y(time);
z = f_z(time);

vx = f_vx(time);
vy = f_vy(time);
vz = f_vz(time);

ax = f_ax(time);
ay = f_ay(time);
az = f_az(time);

figure(1)
plot(time, x)
hold on
plot(time, y)
plot(time, z)
legend('x', 'y', 'z')
xlim([0 tf])
ylim([-2 2])
xlabel('t')
ylabel('position')

figure(2)
plot3(x, y, z)
xlabel('x')
ylabel('y')
zlabel('z')
xlim([-2 2])
ylim([-2 2])
zlim([-2 2])
grid on

figure(3)
plot(time, vx)
hold on
plot(time, vy)
plot(time, vz)
legend('vx', 'vy', 'vz')
xlim([0 tf])
xlabel('t')
ylabel('velocity')

figure(4)
plot(time, ax)
hold on
plot(time, ay)
plot(time, az)
legend('ax', 'ay', 'az')
xlim([0 tf])
xlabel('t')
ylabel('acceleration')

T_over_W = sqrt(ax.^2 + ay.^2 + (az+g).^2) / g;
figure(5)
plot(time, T_over_W)
xlabel('t')
ylabel('thrust to weight ratio')