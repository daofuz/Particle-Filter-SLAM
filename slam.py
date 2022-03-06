import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def bresenham2D(sx, sy, ex, ey):
  '''
  Bresenham's ray tracing algorithm in 2D.
  Inputs:
	  (sx, sy)	start point of ray
	  (ex, ey)	end point of ray
  '''
  sx = int(round(sx))
  sy = int(round(sy))
  ex = int(round(ex))
  ey = int(round(ey))
  dx = abs(ex-sx)
  dy = abs(ey-sy)
  steep = abs(dy)>abs(dx)
  if steep:
    dx,dy = dy,dx # swap

  if dy == 0:
    q = np.zeros((dx+1,1),dtype=int)
  else:
    q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
  if steep:
    if sy <= ey:
      y = np.arange(sy,ey+1)
    else:
      y = np.arange(sy,ey-1,-1)
    if sx <= ex:
      x = sx + np.cumsum(q)
    else:
      x = sx - np.cumsum(q)
  else:
    if sx <= ex:
      x = np.arange(sx,ex+1)
    else:
      x = np.arange(sx,ex-1,-1)
    if sy <= ey:
      y = sy + np.cumsum(q)
    else:
      y = sy - np.cumsum(q)
  return np.vstack((x,y))
def read_data_from_csv(filename):
  '''
  INPUT
  filename        file address

  OUTPUT
  timestamp       timestamp of each observation
  data            a numpy array containing a sensor measurement in each row
  '''
  data_csv = pd.read_csv(filename, header=None)
  data = data_csv.values[:, 1:]
  timestamp = data_csv.values[:, 0]
  return timestamp, data
df_encoder = pd.read_csv("../sensor_data/encoder.csv", header=None)
df_fog = pd.read_csv("../sensor_data/fog.csv", header=None)
R = list(map(float, "0.00130201 0.796097 0.605167 0.999999 -0.000419027 -0.00160026 -0.00102038 0.605169 -0.796097".split()))
R = np.array(R).reshape(3,3)

T = list(map(float, "0.8349 -0.0126869 1.76416".split()))
T = np.array(T)
step = 1000000000

df_tmp = df_encoder[(1544582648800000000<=df_encoder.iloc[:, 0])]
df_tmp = df_tmp[(df_encoder.iloc[:, 0]<=1544583809200000000)]
df_tmp.iloc[:, 1] *= 0.623479*np.pi/4096
df_tmp.iloc[:, 2] *= 0.622806*np.pi/4096
df_fog=df_fog[(1544582648800000000<=df_fog.iloc[:, 0])]
df_fog=df_fog[(df_fog.iloc[:, 0]<=1544583809200000000)]
time_delta=np.diff(df_tmp.iloc[:, 0].to_numpy())/10**9
x_delta=np.diff(df_tmp.iloc[:, 1:].mean(1).to_numpy())
yawdata=df_fog.iloc[:, 3].to_numpy().cumsum()
yawdata=yawdata[::10]
delta_yaw=np.diff(yawdata)
delta_yaw=delta_yaw[:-2]
v=x_delta/time_delta
omega=delta_yaw/time_delta
dfomegav = pd.DataFrame(df_tmp.iloc[1:, 0])
dfomegav['omega']=omega
dfomegav['v']=v
df_tmp = dfomegav[(1544582648800000000<=dfomegav.iloc[:, 0])]
df_tmp =df_tmp[(dfomegav.iloc[:, 0]<=1544582648800000000+step)]
mean = df_tmp.iloc[:, 1:].mean(0)
cov = df_tmp.iloc[:, 1:].cov()
def compute_weight(p):
        corrs = np.zeros(n_particle, int)
        for i, (x,y,theta) in enumerate(p):
            x0,y0 = int(x)+dx,int(y)+dy
            if(x0>=1400 or y0>=1300):
                continue
            grid_tmp = np.zeros((200,200), int)
            for x,y in m_tmp:
                if(x-(x0-100)>=200 or y-(y0-100)>=200):
                    continue
                xs,ys = bresenham2D(100,100,x-(x0-100),y-(y0-100))
                grid_tmp[xs,ys] = -1
            corrs[i] = np.count_nonzero(np.logical_and((gridmap[x0-100:x0+100,y0-100:y0+100]<0), (grid_tmp<0)))

        return corrs / corrs.sum()
def lidar_to_world(p,m):
    m = (np.array([[np.cos(p[2]), -np.sin(p[2])], [np.sin(p[2]), np.cos(p[2])]]) @ m.T).T
    m[:, 0] += p[ 0]+100
    m[:, 1] += p[ 1]+1200
    m = m.astype(int)
    return  sorted(m, key=lambda x: (abs(x[0]-x0), abs(x[1]-y0)), reverse=True)
def condition(x): return x <= 75 and x >=2
lidarind,lidar=read_data_from_csv('../sensor_data/lidar.csv')
n_particle = 50
gridmap = np.zeros((1500,1400))

dx,dy = 100,1200
start = 1544582648800000000
end =   1544583809200000000
n_step = (end - start + step - 1)//step
weights = np.zeros((n_step, n_particle))
weights[0] = np.ones(n_particle) / n_particle
weights[1, :] = weights[0, :].copy()


pa = np.zeros((n_step, n_particle, 3))#paticles
pos = np.zeros((n_step, 3))#postions



for i, s in enumerate(range(start+step, end-step, step)):
    omega,v = np.random.multivariate_normal(mean, cov * np.array([[1,1],[1,1]]), n_particle).T
    pa[i+1, :, 0]  = pa[i, :, 0] + v*np.cos(pa[i, :, 2] + omega)
    pa[i+1, :, 1]  = pa[i, :, 1] + v*np.sin(pa[i, :, 2] + omega)
    pa[i+1, :, 2] = pa[i, :, 2] + omega
    idx = np.searchsorted(lidarind, s + step//2)
    tmp = lidar[idx]
    output = [idx for idx, element in enumerate(tmp) if condition(element)]
    tmp=[element for idx, element in enumerate(tmp) if condition(element)]
    m=np.zeros((len(output),2))
    m[:,0]=tmp*np.cos((np.array(output)*0.666  -5)/180*np.pi)
    m[:,1]=tmp*np.sin((np.array(output)*0.666  -5)/180*np.pi)
    tmp = np.zeros((m.shape[0], 3))
    tmp[:, :2] = m
    m = (R @ (tmp+T).T).T[:, :2]
    if i != 0:
        weights[i+1, :]=compute_weight(pa[i+1])
    pos[i+1,:] = (pa[i+1, :, :] * weights[i+1, :, None]).sum(0)
    x0,y0 = int(pos[i+1, 0])+dx,int(pos[i+1, 1])+dy
    m_tmp = lidar_to_world(pos[i+1],m)

    for x,y in m_tmp:
        xs,ys = bresenham2D(x0,y0,x,y)
        gridmap[xs,ys] -=np.log(4)
        gridmap[x,y] += np.log(4)
    gridmap[gridmap < -5] = -5
    gridmap[gridmap > 5] = 5
    gridmap *= 0.999

    df_tmp = dfomegav[(s<=dfomegav.iloc[:, 0])]
    df_tmp =df_tmp[(dfomegav.iloc[:, 0]<=s+step)]
    mean = df_tmp.iloc[:, 1:].mean(0)
    cov = df_tmp.iloc[:, 1:].cov()

    n_bellow = np.count_nonzero(weights[i+1,:] < 0.1)
    if n_bellow >= 10:
        indices = np.random.choice(n_particle, n_particle, p=weights[i +1, :])
        pa[i+1,:,:] = pa[i+1,indices,:]
        weights[i+1,:] = weights[i+1, indices]
        weights[i+1] /= weights[i+1].sum()
a_file = open("data.txt", "w")
for row in gridmap:
    np.savetxt(a_file, row)
a_file.close()
a_file = open("data_pos.txt", "w")
for row in pos:
    np.savetxt(a_file, row)
a_file.close()
